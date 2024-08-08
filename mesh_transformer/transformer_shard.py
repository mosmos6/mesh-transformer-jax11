from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
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

class CausalTransformerShard:
    def __init__(self, config, mesh_manager):
        self.config = config
        self.mesh_manager = mesh_manager
        self.layers = config["layers"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.heads_per_shard = config["n_heads"] // config["cores_per_replica"]
        self.transformer_layers = [TransformerLayerShard(config, mesh_manager) for i in range(self.layers)]
        self.embed = EmbeddingShard(config["n_vocab"], self.d_model)
        self.proj = ProjectionShard(config)
        self.rpe = None  # Adjust this based on your configuration
        
    def eval(self, context, target, z_loss=0., mask=0.0):
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        else:
            attn_bias = 0

        attn_bias += mask

        x = self.embed(context)

        for l in self.transformer_layers:
            x = x + l(x, attn_bias)

        with self.proj.mesh:
            shard_start_index = jax.lax.axis_index('mp') * self.config["dim_per_shard"]
            return self.proj.loss(x, target, shard_start_index, z_loss)

    def loss(self, ctx, tgt, z_loss=False, mask=0.0):
        loss, correct = self.eval(ctx, tgt, float(z_loss), mask=mask)

        return {
            "loss": loss.mean(),
            "last_loss": loss[-1].mean(),
            "all_loss": loss,
            "correct": correct
        }

    def generate_initial(self, context, length):
        print("Entering CausalTransformerShard generate_initial")
        last = context[-1:]
        context = context[:-1]

        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        else:
            attn_bias = 0

        x = self.embed(context)

        states = []
        for i, l in enumerate(self.transformer_layers):
            print(f"Processing layer {i} in generate_initial")
            res, layer_state = l.get_init_decode_state(x, length - 1, attn_bias)
            x = x + res
            states.append(layer_state)

        return self.proj(x), (last.astype(jnp.uint32), states, jax.random.PRNGKey(0))
        
    def generate_once(self, new_tok, state):
        print("Entering CausalTransformerShard generate_once")
        input_len = state[0]["v"].shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
            attn_bias = attn_bias[:, -1:, :]
        else:
            attn_bias = 0

        x = self.embed(new_tok)

        new_states = []

        for i, (l, s) in enumerate(zip(self.transformer_layers, state)):
            print(f"Processing layer {i} in generate_once")
            res, layer_state = l.decode_once(s, x, attn_bias)
            x = x + res
            new_states.append(layer_state)

        print("CausalTransformerShard generate_once completed")
        return self.proj(x), new_states

class CausalTransformer:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]

        # Define the device mesh
        dp = jax.device_count() // config["cores_per_replica"]
        mp = config["cores_per_replica"]
        devices = mesh_utils.create_device_mesh((dp, mp))
        self.mesh = Mesh(devices, axis_names=('dp', 'mp'))

        def init_fn(rng, x):
            transformer = CausalTransformerShard(config, self.mesh)
            return transformer.init(rng, x)

        self.init_shmap = shard_map(
            init_fn,
            in_specs=(P(), P()),
            out_specs=(P(), P()),
            mesh=self.mesh,
            check_rep=False
        )

        rng = jax.random.PRNGKey(0)
        sample_input = jnp.zeros((config["seq"], config["per_replica_batch"]), dtype=jnp.uint32)
        self.state, _ = self.init_shmap(rng, sample_input)

        def train_fn(state, ctx, tgt):
            def train_loss(params, x, y):
                transformer = CausalTransformerShard(config, self.mesh)
                out = transformer.loss(x, y, z_loss=True)
                return out["loss"], out["last_loss"]

            def microbatch(old_grad, batch):
                ctx, tgt = batch
                val_grad_fn = jax.value_and_grad(train_loss, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(state["params"], ctx, tgt)
                new_grad = jax.tree_map(lambda a, b: a + b, old_grad, grad)
                gnorm = global_norm(grad)
                return new_grad, (loss, last_loss, gnorm)

            if ctx.shape[0] == 1:
                val_grad_fn = jax.value_and_grad(train_loss, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(state["params"], ctx[0], tgt[0])
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

            return loss, last_loss, grad_norm, grad_norm_micro, {
                "params": optax.apply_updates(state["params"], updates),
                "step": state["step"] + 1,
                "opt_state": new_opt_state
            }

        self.train_shmap = shard_map(
            train_fn,
            in_specs=(P(), P(), P()),
            out_specs=(P(), P(), P(), P(), P()),
            mesh=self.mesh,
            check_rep=False
        )

        def eval_fn(state, ctx, tgt, mask):
            transformer = CausalTransformerShard(self.config, self.mesh)
            return transformer.eval(ctx, tgt, mask=mask)

        self.eval_shmap = shard_map(
            eval_fn,
            in_specs=(P(), P(), P(), P()),
            out_specs=(P(),),
            mesh=self.mesh,
            check_rep=False
        )

        def generate_fn(state, key, ctx, ctx_length, aux, sampler_options):
            def generate_sample(params, context, ctx_length, aux):
                transformer = CausalTransformerShard(self.config, self.mesh)
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

            return generate_sample(state["params"], ctx, ctx_length, aux)

        self.generate_shmap = shard_map(
            generate_fn,
            in_specs=(P(), P(), P(), P(), P(), P()),
            out_specs=(P(), P()),
            mesh=self.mesh,
            check_rep=False
        )

        self.move_shmap = shard_map(
            lambda x, _: to_bf16(x),
            in_specs=(P(), P()),
            out_specs=(P(),),
            mesh=self.mesh,
            check_rep=False
        )

    def train(self, sample):
        obs = jnp.transpose(sample["obs"], (1, 0))
        tgt = jnp.transpose(sample["target"], (1, 0))

        loss, last_loss, grad_norm, grad_norm_micro, self.state = self.train_shmap(self.state, obs, tgt)
        return loss.mean(), last_loss.mean(), grad_norm.mean(), grad_norm_micro.mean()

    def eval(self, sample):
        ctx = jnp.transpose(sample["obs"], (1, 0))
        tgt = jnp.transpose(sample["target"], (1, 0))
        ctx_length = sample.get("ctx_length", np.array([len(sample["obs"][0])] * len(sample["obs"])))
        mask = (jnp.arange(0, len(ctx)) > ctx_length) * -1e10

        return self.eval_shmap(self.state, ctx, tgt, mask)

    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
        key = jax.random.PRNGKey(0)
        aux = jnp.zeros((ctx.shape[0], gen_length), dtype=jnp.uint32)

        return self.generate_shmap(self.state, key, jnp.transpose(ctx, (1, 0)), ctx_length, aux, sampler_options)

    def move(self):
        self.state = self.move_shmap(self.state, None)
