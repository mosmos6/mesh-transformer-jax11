from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import os
import numpy as np

# Assume these imports are defined properly
from mesh_transformer.util import to_f32, to_bf16, global_norm
from mesh_transformer.layers import EmbeddingShard, TransformerLayerShard, RelativePositionEmbs, ProjectionShard
from mesh_transformer.checkpoint import write_ckpt, read_ckpt
from mesh_transformer.mesh_context_manager import MeshContextManager  # Import from new file


class CausalTransformerShard(nn.Module):
    config: dict
    mesh_manager: object

    def setup(self):
        self.layers = self.config["layers"]
        self.d_model = self.config["d_model"]
        self.n_heads = self.config["n_heads"]
        self.heads_per_shard = self.n_heads // self.config["cores_per_replica"]
        self.transformer_layers = [TransformerLayerShard(config=self.config, mesh_manager=self.mesh_manager) for _ in range(self.layers)]
        #self.embed = nn.Embed(self.config["n_vocab"], self.d_model)
        self.embed = EmbeddingShard(config=self.config)
        self.proj = ProjectionShard(config=self.config)

        if self.config["pe"] == "t5":
            self.rpe = RelativePositionEmbs()
        else:
            self.rpe = None

    def __call__(self, x, mask=0.0):
        
        print(f"Shape of x before embedding: {x.shape}")  # Debug: Check x shape before embedding
        x = self.embed(x)
        print(f"Shape of x after embedding: {x.shape}")  # Debug: Check x shape after embedding

        # Calculate attn_bias
        input_len = x.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard)
        else:
            attn_bias = mask  # If rpe is not used, simply set attn_bias to the mask or 0

        for i, layer in enumerate(self.transformer_layers):
            print(f"Shape of x before layer {i}: {x.shape}")  # Debug: Check x shape before each layer
            x = layer(x, attn_bias)
            print(f"Shape of x after layer {i}: {x.shape}")  # Debug: Check x shape after each layer

        return self.proj(x)

    def eval(self, context, target, z_loss=0., mask=0.0):
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard)
        else:
            attn_bias = mask

        x = self.embed(context)

        for l in self.transformer_layers:
            x = x + l(x, attn_bias)

        shard_start_index = compute_shard_start_index(self.proj.dim_per_shard)
        return self.proj.loss(x, target, shard_start_index, z_loss)

    def loss(self, ctx, tgt, z_loss=False, mask=0.0):
        loss, correct = self.eval(ctx, tgt, float(z_loss), mask=mask)

        return {
            "loss": loss.mean(),
            "last_loss": loss[-1].mean(),
            "all_loss": loss,
            "correct": correct
        }

    def generate_initial(self, context, length, mask=0.0):
        last = context[-1:]
        context = context[:-1]

        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard)
        else:
            attn_bias = mask

        x = self.embed(context)

        states = []
        for i, l in enumerate(self.transformer_layers):
            res, layer_state = l.get_init_decode_state(x, length - 1, attn_bias)
            x = x + res
            states.append(layer_state)

        return self.proj(x), (last.astype(jnp.uint32), states, jax.random.PRNGKey(0))

    def generate_once(self, new_tok, state, mask=0.0):
        input_len = state[0]["v"].shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard)
            attn_bias = attn_bias[:, -1:, :]
        else:
            attn_bias = mask

        x = self.embed(new_tok)

        new_states = []

        for i, (l, s) in enumerate(zip(self.transformer_layers, state)):
            res, layer_state = l.decode_once(s, x, attn_bias)
            x = x + res
            new_states.append(layer_state)

        return self.proj(x), new_states




class CausalTransformer:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]

        dp = jax.device_count() // config["cores_per_replica"]
        mp = config["cores_per_replica"]

        mesh_manager = MeshContextManager(dp, mp)

        def init_fn(rng, x):
            model = CausalTransformerShard(config=config, mesh_manager=mesh_manager)
            return model.init(rng, x)  # This should initialize the model with x

        self.init_shmap = shard_map(
            init_fn,
            in_specs=(P(), P()),
            out_specs=(P(), P()),
            mesh=mesh_manager.get_mesh(),
            check_rep=False
        )

        rng = jax.random.PRNGKey(0)
        sample_input = jnp.zeros((config["seq"], config["per_replica_batch"]), dtype=jnp.uint32)

        print(f"Shape of sample_input before shmap: {sample_input.shape}")  # Debug: Before shmap
        self.state, _ = self.init_shmap(rng, sample_input)
        print(f"Shape of x after init_shmap: {self.state.shape}")  # Debug: After shmap
        
        def train_fn(state, ctx, tgt):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config=config, mesh_manager=mesh_manager)
                out = transformer.loss(x, y, z_loss=True)
                return out["loss"], out["last_loss"]

            def microbatch(old_grad, batch):
                ctx, tgt = batch
                val_grad_fn = jax.value_and_grad(train_loss, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(to_bf16(state["params"]), ctx, tgt)
                new_grad = jax.tree_map(lambda a, b: a + b, old_grad, grad)
                gnorm = global_norm(grad)
                return new_grad, (loss, last_loss, gnorm)

            if ctx.shape[0] == 1:
                val_grad_fn = jax.value_and_grad(train_loss, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(to_bf16(state["params"]), ctx[0], tgt[0])
                gnorm = global_norm(grad)
            else:
                grad, (loss, last_loss, gnorm) = jax.lax.scan(microbatch,
                                                              jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16),
                                                                           state["params"]),
                                                              (ctx, tgt))

            grad_norm_micro = jax.lax.pmean(gnorm, "batch")
            grad = jax.lax.pmean(grad, "batch")
            grad_norm = global_norm(grad)
            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])

            return to_f32(loss), to_f32(last_loss), to_f32(grad_norm), to_f32(grad_norm_micro), {
                "params": optax.apply_updates(state["params"], to_f32(updates)),
                "step": state["step"] + 1,
                "opt_state": new_opt_state
            }

        self.train_shmap = shard_map(
            train_fn,
            in_specs=(P(), P(), P()),
            out_specs=(P(), P()),
            mesh=mesh_manager.get_mesh(),
            check_rep=False
        )

    def train(self, sample):
        obs = jnp.transpose(sample["obs"], (1, 0))
        tgt = jnp.transpose(sample["target"], (1, 0))

        print(f"Shape of obs before train_shmap: {obs.shape}")  # Debug: Before train_shmap
        loss, last_loss, grad_norm, grad_norm_micro, self.state = self.train_shmap(self.state, obs, tgt)
        print(f"Shape of x after train_shmap: {self.state['params'].shape}")  # Debug: After train_shmap
        return loss.mean(), last_loss.mean(), grad_norm.mean(), grad_norm_micro.mean()

    def eval(self, sample):
        def eval_loss(x, y, mask):
            transformer = CausalTransformerShard(self.config)
            return transformer.loss(x, y, mask=mask)

        ctx = jnp.transpose(sample["obs"], (1, 0))
        tgt = jnp.transpose(sample["target"], (1, 0))
        ctx_length = sample.get("ctx_length", np.array([len(sample["obs"][0])] * len(sample["obs"])))
        mask = (jnp.arange(0, len(ctx)) > ctx_length) * -1e10

        return eval_loss(to_bf16(self.state["params"]), ctx, tgt, mask)

    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
        def generate_sample(context, ctx_length, aux):
            transformer = CausalTransformerShard(self.config)
            _, initial_state = transformer.generate_initial(context, ctx_length)

            def generate_scan_fn(carry, sampler_input):
                next_token, decode_state, sample_key = carry
                sample_key, new_key = jax.random.split(sample_key)

                logits, new_state = transformer.generate_once(next_token, decode_state)
                next_token, sample_info = self.config["sampler"](sample_key, logits, sampler_input, **sampler_options)

                output = (next_token, sample_info, logits) if return_logits else (next_token, sample_info)
                new_carry = (next_token, new_state, new_key)
                return new_carry, output

            final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=aux, length=gen_length)
            return final_state, outputs

        generate_fn = nn.compact(generate_sample).apply
        key = jax.random.PRNGKey(0)
        aux = jnp.zeros((ctx.shape[0], gen_length), dtype=jnp.uint32)

        return generate_fn(self.state["params"], key, jnp.transpose(ctx, (1, 0)), ctx_length, aux)

    def move(self):
        self.state = self.move_shmap(self.state, None)
