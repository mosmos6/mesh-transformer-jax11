import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from mesh_transformer.util import f_psum, g_psum, maybe_shard, head_print
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from mesh_transformer.mesh_context_manager import MeshContextManager
from functools import partial
from jax import profiler
import gc
from flax.linen import remat


class ReplicatedLayerNorm(nn.Module):
    offset: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = self.param("scale", nn.initializers.ones, param_shape)
        offset = self.param("offset", nn.initializers.zeros, param_shape)

        print(f"Before g_psum in ReplicatedLayerNorm - scale shape: {scale.shape}, offset shape: {offset.shape}")

        # Replace with g_psum for psum in forward pass only
        scale = g_psum(scale)
        offset = g_psum(offset)

        print(f"After g_psum in ReplicatedLayerNorm - scale shape: {scale.shape}, offset shape: {offset.shape}")

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)


class RMSNorm(nn.Module):
    offset: bool
    elementwise: bool

    @nn.compact
    def __call__(self, x):
        param_shape = (x.shape[-1],) if self.elementwise else ()
        normed = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)

        scale = self.param('scale', nn.initializers.constant(x.shape[-1] ** 0.5), param_shape)
        scale = f_psum(scale)  # Using f_psum
        normed = normed * scale

        if self.offset:
            offset = self.param('offset', nn.initializers.zeros, param_shape)
            offset = f_psum(offset)  # Using f_psum
            normed = normed + offset

        return normed


def getnorm(type):
    if type == "layernorm":
        return ReplicatedLayerNorm()
    if type == "layernorm-desync":
        return nn.LayerNorm()
    elif type == "layernorm-nobias":
        return ReplicatedLayerNorm(offset=False)
    elif type == "rmsnorm":
        return RMSNorm(False, True)
    elif type == "scalenorm":
        return RMSNorm(False, False)
    elif type == "rmsnorm-bias":
        return RMSNorm(True, True)
    elif type == "scalenorm-bias":
        return RMSNorm(True, False)
    else:
        raise Exception("Not implemented")


# Other unchanged classes like RelativePositionEmbs, EmbeddingShard, etc.

class TransformerLayerShard(nn.Module):
    config: dict
    mesh_manager: MeshContextManager
    init_scale: float = 1.0

    def setup(self):
        self.n_heads = self.config["n_heads"]
        self.dim = self.config["d_model"]
        self.shards = self.config["cores_per_replica"]
        self.norm = getnorm(self.config["norm"])
        self.is_rotary = self.config["pe"] == "rotary"

        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.shards == 0

        self.dim_per_head = self.dim // self.n_heads
        self.heads_per_shard = self.n_heads // self.shards
        self.dim_per_shard = self.dim // self.shards
        self.pe_rotary_dims = self.config.get("pe_rotary_dims", self.dim_per_head)

        self.q = nn.Dense(self.n_heads * self.dim_per_head, use_bias=False)
        self.v = nn.Dense(self.n_heads * self.dim_per_head, use_bias=False)
        self.k = nn.Dense(self.n_heads * self.dim_per_head, use_bias=False)
        self.o = nn.Dense(self.dim, use_bias=False, kernel_init=nn.initializers.truncated_normal(stddev=self.init_scale / np.sqrt(self.dim)))
        self.dense_proj = nn.Dense(self.dim * 4)
        self.dense_proj_o = nn.Dense(self.dim, kernel_init=nn.initializers.truncated_normal(stddev=self.init_scale / np.sqrt(self.dim)))

    @nn.remat
    def self_attn(self, q, v, k, attn_bias=None):
        attention_logits = jnp.einsum("thd,Thd->htT", q, k, optimize="optimal")
        attention_weights = jax.nn.softmax(attention_logits)
        attention_output = jnp.einsum("htT,Thd->thd", attention_weights, v)
        return attention_output

    @nn.remat
    def ff(self, x):
        dense_proj = self.dense_proj(x)
        dense_proj = jax.nn.gelu(dense_proj)
        return self.dense_proj_o(dense_proj)

    @nn.compact
    def __call__(self, x, attn_bias, layer_index, state):
        print(f"Before f_psum in TransformerLayerShard - x shape: {x.shape}")

        x = f_psum(x)  # Use f_psum for data parallelism

        print(f"After f_psum in TransformerLayerShard - x shape: {x.shape}")

        # Apply normalization and projections
        x = self.norm(x)
        q, v, k = self.q(x), self.v(x), self.k(x)

        attn_out = self.self_attn(q, v, k, attn_bias)
        attn_out = attn_out.reshape((x.shape[0], x.shape[1], self.n_heads * self.dim_per_head))
        dense_out = self.ff(x)

        pre_result = attn_out + dense_out

        print(f"Before g_psum in TransformerLayerShard remat- pre_result shape: {pre_result.shape}")

        result = g_psum(pre_result)  # Use g_psum for final reduction

        print(f"After g_psum in TransformerLayerShard remat- result shape: {result.shape}")

        return result

    def decode_once(self, decode_state, x, attn_bias):
        print(f"Before f_psum in decode_once - x shape: {x.shape}")

        x = f_psum(x)
        x = self.norm(x)
        print(f"After f_psum in decode_once - x shape: {x.shape}")

        assert x.shape[0] == 1

        q, v, k = self.qvk_proj(x)
        v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[0]

        masked_tokens = length - tokens_decoded
        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask) + attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        combined_output = attn_out + dense_out
        print(f"Combined output before g_psum in decode_once - combined_output shape: {combined_output.shape}")

        final_output = g_psum(combined_output)
        print(f"Final output after g_psum in decode_once - final_output shape: {final_output.shape}")

        return final_output, {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    def get_init_decode_state(self, x, given_length, attn_bias):
        with self.mesh_manager.get_mesh():
            print(f"Before f_psum in get_init_decode_state - x shape: {x.shape}")
            x = f_psum(x)
            x = self.norm(x)
            print(f"After f_psum in get_init_decode_state - x shape: {x.shape}")

        q, v, k = self.qvk_proj(x)

        full_length = x.shape[0]
        masked_tokens = full_length - given_length

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        bias = -1e10 * (1. - causal_mask) - 1e10 * (jnp.arange(0, full_length) < masked_tokens) + attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        return attn_out + dense_out, {"k": k, "v": v, "tokens_decoded": given_length.astype(jnp.uint32)}


class ProjectionShard(nn.Module):
    config: dict

    def setup(self):
        self.dim_per_shard = self.config["dim_per_shard"]
        self.out_dim = self.config["d_model"]
        self.shards = self.config["cores_per_replica"]
        self.mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(self.shards, -1), ("dp", "mp"))
        self.layer_norm = nn.LayerNorm()
        self.dense = nn.Dense(self.out_dim)

    def loss(self, x, target, shard_start_index, z_loss):
        logits = self.forward(x)
        logits = jnp.swapaxes(logits, 0, 1)

        print(f"Before g_psum in ProjectionShard - shard_start_index shape: {shard_start_index.shape}")

        shard_start_index = g_psum(shard_start_index)
        print(f"After g_psum in ProjectionShard - shard_start_index shape: {shard_start_index.shape}")

        predicted_logits = jnp.take_along_axis(logits, target[:, :, None] + shard_start_index, axis=-1)
        exp_logits = jnp.exp(logits - logits.max(axis=-1, keepdims=True))
        print(f"Before g_psum in ProjectionShard - sum_exp_logits shape: {exp_logits.shape}")
        sum_exp_logits = g_psum(exp_logits)
        print(f"After g_psum in ProjectionShard - sum_exp_logits shape: {sum_exp_logits.shape}")

        softmax_logits = predicted_logits - jnp.log(sum_exp_logits)

        z_loss_penalty = z_loss * jnp.square(jnp.log(sum_exp_logits)).mean() if z_loss else 0

        return -(softmax_logits.mean() + z_loss_penalty), jnp.argmax(logits, axis=-1) == target

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense(x)
        return x


class Projection(nn.Module):
    config: dict

    def setup(self):
        self.dim = self.config["n_vocab"]
        self.norm = nn.LayerNorm()
        self.proj = nn.Dense(self.dim)

    def __call__(self, x):
        x = self.norm(x)
        return self.proj(x)

    def loss(self, x, targets, z_loss=1):
        x = self.norm(x)
        logits = self.proj(x)

        logits -= logits.max(-1, keepdims=True)

        gt_onehot = jax.nn.one_hot(targets, self.dim)
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        exp_logits = jnp.exp(logits)

        sum_exp_logits = exp_logits.sum(axis=-1)

        loss = jnp.log(sum_exp_logits) - predicted_logits

        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()
        correct = (0.0 == predicted_logits)
        return loss, correct


def compute_shard_start_index(dim_per_shard):
    return jax.lax.axis_index('mp') * dim_per_shard
