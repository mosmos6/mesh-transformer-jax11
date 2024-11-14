import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from mesh_transformer.util import f_psum, g_psum, f_psum_first, g_psum_first, maybe_shard, head_print
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

        print(f"Before all_gather in ReplicatedLayerNorm - scale shape: {scale.shape}, offset shape: {offset.shape}")

        # Replace with g_psum for psum in forward pass only
        scale = jax.lax.all_gather(scale, "mp")[0]
        offset = jax.lax.all_gather(offset, "mp")[0]

        print(f"After all_gather in ReplicatedLayerNorm - scale shape: {scale.shape}, offset shape: {offset.shape}")

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


class RelativePositionEmbs(nn.Module):
    num_buckets: int
    max_distance: int

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        n = np.maximum(n, 0)
        max_exact = num_buckets // 2
        is_small = (n < max_exact)
        val_if_large = max_exact + (
                np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                np.log(max_distance / max_exact) *
                (num_buckets - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    @nn.compact
    def __call__(self, qlen, klen, heads):
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position, self.num_buckets, self.max_distance)
        relative_attention_bias = self.param('rel_embedding', nn.initializers.truncated_normal(stddev=0.02), [heads, self.num_buckets])
        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(relative_attention_bias.dtype)
        values = jax.lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (((1,), (0,)), ((), ())))
        return values


def fixed_pos_embedding(seq_len, n_heads, dim_per_head):
    theta = jnp.arange(0, dim_per_head // 2, dtype=jnp.float32)
    theta = 1.0 / (10000 ** (2 * theta / dim_per_head))
    
    seq = jnp.arange(seq_len, dtype=jnp.float32)
    theta = seq[:, None] * theta[None, :]
    
    sin = jnp.sin(theta)
    cos = jnp.cos(theta)
    
    # Expand dimensions to match input tensors
    sin = sin[:, None, :].repeat(n_heads, axis=1)  # Shape: (seq_len, n_heads, dim_per_head // 2)
    cos = cos[:, None, :].repeat(n_heads, axis=1)

    sin = sin.repeat(2, axis=-1)  # Shape: (seq_len, n_heads, dim_per_head)
    cos = cos.repeat(2, axis=-1)  # Shape: (seq_len, n_heads, dim_per_head)
    
    # Adjust shapes to match input x (batch size, seq_len, n_heads, dim_per_head)
    sin = sin[:, None, :, :]  # Add batch size dimension, Shape: (seq_len, 1, n_heads, dim_per_head)
    cos = cos[:, None, :, :]  # Add batch size dimension, Shape: (seq_len, 1, n_heads, dim_per_head)
    
    return sin, cos





    

def rotate_every_two(x):
    # Debug: Print the initial shape of x
    print(f"rotate_every_two: Input x shape: {x.shape}")

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # Debug: Print the shapes after selecting even and odd elements
    print(f"rotate_every_two: x1 shape: {x1.shape}, x2 shape: {x2.shape}")

    # Combine x1 and x2 back together after rotation
    x = jnp.stack((-x2, x1), axis=-1)

    # Debug: Print the shape after stacking
    print(f"rotate_every_two: shape after stacking: {x.shape}")

    # Rearrange to original shape with doubled last dimension
    reshaped_x = rearrange(x, '... d j -> ... (d j)')
    print(f"rotate_every_two: Reshaped x shape: {reshaped_x.shape}")

    return reshaped_x




from einops import repeat

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    print(f"apply_rotary_pos_emb: x shape: {x.shape}, sin shape: {sin.shape}, cos shape: {cos.shape}")

    # Split the input tensor into two parts and apply the rotary transformation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    rotated_x1 = (x1 * cos) - (x2 * sin)
    rotated_x2 = (x2 * cos) + (x1 * sin)

    # Efficiently combine back the results
    x_out = jnp.concatenate([rotated_x1[..., None], rotated_x2[..., None]], axis=-1).reshape(x.shape)

    return x_out


class EmbeddingShard(nn.Module):
    config: dict

    def setup(self):
        in_dim = self.config["n_vocab"]
        out_dim = self.config["d_model"]
        shards = self.config["cores_per_replica"]

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.out_dim_per_shard = out_dim // shards

        if self.config["pe"] == "fixed":
            embed_init = nn.initializers.truncated_normal(stddev=0.02)
            self.positional_embeddings = self.param('pos_embs', embed_init, [self.config["seq"], self.out_dim_per_shard])
        else:
            self.positional_embeddings = None

        self.proj = nn.Dense(self.out_dim, kernel_init=nn.initializers.truncated_normal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        shard_start_index = jax.lax.axis_index('mp') * self.in_dim_per_shard

        # Use one-hot encoding for input
        input_onehot = jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        proj_out = self.proj(input_onehot)

        # Sum across all devices
        print(f"Before Sum across all devices - x shape: {x.shape}")
        proj_out = g_psum(proj_out)
        print(f"After Sum across all devices - x shape: {x.shape}")

        

        # Apply positional embeddings if available
        if self.positional_embeddings is not None:
            print(f"Before all_pos_embed with all_gather - x shape: {x.shape}")
            all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'mp')
            print(f"After all_pos_embed with all_gather - x shape: {x.shape}")

            # Flatten and transpose like original GPT-J
            all_pos_embed = jnp.transpose(all_pos_embed, (1, 0, 2)).reshape(self.config["seq"], -1)

            proj_out += all_pos_embed

        return proj_out


class TransformerLayerShard(nn.Module):
    config: dict
    mesh_manager: MeshContextManager
    init_scale: float = 1.0

    def setup(self):
        self.n_heads = self.config["n_heads"]
        self.dim = self.config["d_model"]
        self.dim_per_head = self.dim // self.n_heads
        self.shards = self.config["cores_per_replica"]
        self.norm = getnorm(self.config["norm"])
        self.is_rotary = self.config["pe"] == "rotary"
        self.qvk_proj_layer = nn.Dense(3 * self.n_heads * self.dim_per_head, use_bias=False)


        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.shards == 0

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

    def qvk_proj(self, x):
        # Project x directly into separate q, v, k with shape (seq_len, dim_per_head)
        q = self.q(x)
        v = self.v(x)
        k = self.k(x)
    
        # No need to reshape further since we’re staying with 3D tensors
        return q, v, k



    @nn.compact
    def __call__(self, x, attn_bias, layer_index, state):
        print(f"Before f_psum in TransformerLayerShard - x shape: {x.shape}")

        x = f_psum(x)  # Use f_psum for data parallelism

        print(f"After f_psum in TransformerLayerShard - x shape: {x.shape}")

        # Apply normalization and projections
        x = self.norm(x)
        q, v, k = self.qvk_proj(x)  # Combined q, v, k calculation

        # Setting up causal mask and bias, based on the original approach
        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias  # Add attn_bias if present

        attn_out = self.self_attn(q, v, k, attn_bias)
        # attn_out = attn_out.reshape((x.shape[0], x.shape[1], self.n_heads * self.dim_per_head))
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
        self.dim = self.config["d_model"]
        self.shards = self.config["cores_per_replica"]
        self.dim_per_shard = self.config.get("dim_per_shard", self.dim // self.shards)
        self.out_dim = self.config["d_model"]
        self.mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(self.shards, -1), ("dp", "mp"))
        self.layer_norm = getnorm(self.config["norm"])
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
        print(f"After forward layer_norm - x shape: {x.shape}")
        x = self.dense(x)
        print(f"After forward dense - x shape: {x.shape}")
        return x

    def __call__(self, x):
        print(f"Before forward call - x shape: {x.shape}")
        return self.forward(x)  # Define the forward pass by calling the forward method


class Projection(nn.Module):
    config: dict

    def setup(self):
        self.dim = self.config["n_vocab"]
        self.norm = getnorm(self.config["norm"])
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
