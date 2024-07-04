from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import os
import numpy as np

# Assume these imports are defined properly
from mesh_transformer.util import to_f32, to_bf16, global_norm
from mesh_transformer.layers import EmbeddingShard, TransformerLayerShard, RelativePositionEmbs, ProjectionShard
from mesh_transformer.checkpoint import write_ckpt, read_ckpt



class CausalTransformerShard(hk.Module):
    def __init__(self, config):
        super().__init__()
        heads = config["n_heads"]
        shards = config["cores_per_replica"]
        layer_count = config["layers"]

        self.transformer_layers = []
        self.heads = heads
        self.heads_per_shard = heads // shards
        self.embed = EmbeddingShard(config)

        init_scale = 2. / layer_count

        for i in range(layer_count):
            self.transformer_layers.append(TransformerLayerShard(config, name=f"layer_{i}", init_scale=init_scale))

        self.proj = ProjectionShard(config)

        if config["pe"] == "t5":
            self.rpe = RelativePositionEmbs()
        else:
            self.rpe = None

    def eval(self, context, target, z_loss=0., mask=0.0):
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        else:
            attn_bias = 0

        attn_bias += mask

        x = hk.remat(self.embed)(context)

        for l in self.transformer_layers:
            x = x + hk.remat(l)(x, attn_bias)

        with self.proj.mesh:
            shard_start_index = jax.lax.axis_index('mp') * self.proj.dim_per_shard
            return hk.remat(self.proj.loss)(x, target, shard_start_index, z_loss)

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

        print("CausalTransformerShard generate_initial completed")
        return self.proj(x), (last.astype(jnp.uint32), states, hk.next_rng_key())

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

        # Calculate dp and mp
        dp = jax.device_count() // config["cores_per_replica"]
        mp = config["cores_per_replica"]

        self.mesh = Mesh(np.array(jax.devices()).reshape(dp, mp), ("dp", "mp"))

        def init_fn():
            sample = jnp.zeros((config["seq"], config["per_replica_batch"]), dtype=jnp.uint32)
            return CausalTransformerShard(config).init(jax.random.PRNGKey(0), sample, sample)

        with self.mesh:
            self.init_shmap = shard_map(init_fn, in_axes=(), out_axes=(), devices=self.mesh)

        self.state = self.init_shmap()

        def train_shard_fn(state, x, y):
            return CausalTransformerShard(config).train(x, y)

        def eval_shard_fn(state, x, y, z_loss, mask):
            return CausalTransformerShard(config).eval(x, y, z_loss, mask)

        def generate_shard_fn(state, x, gen_length, temperature, top_k, top_p, callback):
            return CausalTransformerShard(config).generate(x, gen_length, temperature, top_k, top_p, callback)

        def move_shard_fn(state, _):
            return jax.tree_map(lambda x: x.astype(jnp.bfloat16), state)

        with self.mesh:
            self.train_shmap = shard_map(train_shard_fn, in_axes=(None, 0, 0), out_axes=(None, 0), devices=self.mesh)
            self.eval_shmap = shard_map(eval_shard_fn, in_axes=(None, 0, 0, None, None), out_axes=(), devices=self.mesh)
            self.generate_shmap = shard_map(generate_shard_fn, in_axes=(None, 0, None, None, None, None), out_axes=(), devices=self.mesh)
            self.move_shmap = shard_map(move_shard_fn, in_axes=(None, None), out_axes=(None), devices=self.mesh)

        self.opt = optax.chain(
            optax.clip_by_global_norm(1),
            optimizer
        )

        print("CausalTransformer initialized.")

    def train(self, sample):
        print("Starting training step...")
        obs = jnp.transpose(sample["obs"], (1, 0))
        tgt = jnp.transpose(sample["target"], (1, 0))

        with self.mesh:
            metrics, state, logits = self.train_shmap(self.state, obs, tgt)

        # Replace params with previous params if NaN loss
        new_params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(jnp.isnan(metrics["loss"]), old, new),
            state["params"],
            self.state["params"],
        )
        state = state.replace(params=new_params)

        self.state = state
        print("Training step completed.")

        return jax.device_get(metrics), logits

    def eval(self, context, target, z_loss=0., mask=0.0):
        print("Starting evaluation...")
        obs = jnp.transpose(context, (1, 0))
        tgt = jnp.transpose(target, (1, 0))

        with self.mesh:
            return self.eval_shmap(self.state, obs, tgt, z_loss, mask)

    def generate(self, context, gen_length, temperature=1.0, top_k=20, top_p=0.9, callback=None):
        print("Starting generation...")
        obs = jnp.transpose(context, (1, 0))

        with self.mesh:
            return self.generate_shmap(self.state, obs, gen_length, temperature, top_k, top_p, callback)

    def move(self):
        print("Moving state to bf16...")
        with self.mesh:
            self.state = self.move_shmap(self.state, None)
        print("State moved to bf16.")
