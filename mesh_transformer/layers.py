import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from mesh_transformer.util import f_psum, g_psum, maybe_shard, head_print
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from mesh_transformer.mesh_context_manager import MeshContextManager  # Import from new file
from functools import partial
from jax import profiler





class ReplicatedLayerNorm(nn.Module):
    offset: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = self.param("scale", nn.initializers.ones, param_shape)
        scale = jax.lax.all_gather(scale, "mp")[0]

        offset = self.param("offset", nn.initializers.zeros, param_shape)
        offset = jax.lax.all_gather(offset, "mp")[0]

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
        scale = jax.lax.pmean(scale, "mp")
        normed = normed * scale

        if self.offset:
            offset = self.param('offset', nn.initializers.zeros, param_shape)
            offset = jax.lax.pmean(offset, "mp")
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
    sin = sin[:, None, None, :].repeat(n_heads, axis=1)  # [seq_len, n_heads, 1, dim_per_head // 2]
    cos = cos[:, None, None, :].repeat(n_heads, axis=1)

    sin = sin.repeat(2, axis=-1)  # [seq_len, n_heads, 1, dim_per_head]
    cos = cos.repeat(2, axis=-1)  # [seq_len, n_heads, 1, dim_per_head]
    
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

    # Interleave the rotated tensors to form the output
    x_out = jnp.empty_like(x)
    x_out = x_out.at[..., ::2].set(rotated_x1)
    x_out = x_out.at[..., 1::2].set(rotated_x2)
    
    return x_out





class EmbeddingShard(nn.Module):
    config: dict

    def setup(self):
        in_dim = self.config["n_vocab"]
        out_dim = self.config["d_model"]
        shards = self.config["cores_per_replica"]
        dim_per_head = self.config["d_head"]
        n_heads = self.config["n_heads"]

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

        # Ensure x-shard_start_index doesn't underflow and wrap around
        one_hot_input = jax.nn.one_hot(jnp.clip(x - shard_start_index, 0, self.in_dim_per_shard - 1), self.in_dim_per_shard)
        proj_out = self.proj(one_hot_input)

        proj_out = g_psum(proj_out)

        if self.positional_embeddings is not None:
            all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'mp')
            all_pos_embed = all_pos_embed.reshape(self.config["seq"], -1)
            proj_out += all_pos_embed

        # Reshape to [batch_size, seq_len, n_heads, dim_per_head]
        batch_size, seq_len = x.shape  # should be (1, 2048)
        proj_out = proj_out.reshape((batch_size, seq_len, self.config["n_heads"], self.config["d_head"]))

        
        # Handle rotary embeddings
        if self.config["pe"] == "rotary":
            seq_len = x.shape[0]
            num_heads = self.config["n_heads"]
            dim_per_head = self.config["d_head"]
            pe_rotary_dims = self.config.get("pe_rotary_dims", dim_per_head)

            # Ensure sincos embedding shapes match the expected dimension
            sincos = fixed_pos_embedding(seq_len, self.config["n_heads"], pe_rotary_dims)
            proj_out = apply_rotary_pos_emb(proj_out, sincos)

        return proj_out

class TransformerLayerShard(nn.Module):
    config: dict
    mesh_manager: MeshContextManager
    init_scale: float = 1.0

    def setup(self):
        
        self.n_heads = self.config["n_heads"]  # Store n_heads as an instance attribute
        self.dim = self.config["d_model"]
        self.shards = self.config["cores_per_replica"]
        self.norm = getnorm(self.config["norm"])
        self.is_rotary = self.config["pe"] == "rotary"

        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.shards == 0

        self.dim_per_head = self.dim // self.n_heads  # Calculate and store dim_per_head as an instance attribute
        self.heads_per_shard = self.n_heads // self.shards
        self.dim_per_shard = self.dim // self.shards
        self.pe_rotary_dims = self.config.get("pe_rotary_dims", self.dim_per_head)

        self.q = nn.Dense(self.n_heads * self.dim_per_head, use_bias=False)  # Use instance attributes now
        self.v = nn.Dense(self.n_heads * self.dim_per_head, use_bias=False)
        self.k = nn.Dense(self.n_heads * self.dim_per_head, use_bias=False)
        
        self.o = nn.Dense(self.dim, use_bias=False, kernel_init=nn.initializers.truncated_normal(stddev=self.init_scale / np.sqrt(self.dim)))

        self.dense_proj = nn.Dense(self.dim * 4)
        self.dense_proj_o = nn.Dense(self.dim, kernel_init=nn.initializers.truncated_normal(stddev=self.init_scale / np.sqrt(self.dim)))

    @partial(profiler.annotate_function, name="self_attn")
    def self_attn(self, q, v, k, attn_bias=None):
        print(f"self_attn: Adjusted q shape: {q.shape}, k shape: {k.shape}")  # Debug
    
        # Corrected einsum string with three dimensions
        attention_logits = jnp.einsum("thd,Thd->htT", q, k)
        print(f"self_attn: Attention logits shape: {attention_logits.shape}")  # Debug
    
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_logits)
        print(f"self_attn: Attention weights shape: {attention_weights.shape}")  # Debug
    
        # Compute the final attention output
        attention_output = jnp.einsum("htT,Thd->thd", attention_weights, v)
        print(f"self_attn: Attention output shape: {attention_output.shape}")  # Debug

    
    
        return attention_output


    @partial(profiler.annotate_function, name="ff")
    def ff(self, x):
        print(f"ff: Input shape: {x.shape}")  # Debug: Input to feedforward
        dense_proj = self.dense_proj(x)
        dense_proj = jax.nn.gelu(dense_proj)
        print(f"ff: Output shape: {x.shape}")  # Debug: Output from feedforward
        

        return self.dense_proj_o(dense_proj)

    def qvk_proj(self, x):
        print(f"qvk_proj: Input x shape: {x.shape}")  # Debug: Before qvk_proj
        q = self.q(x).reshape((x.shape[0], self.n_heads, self.dim_per_head))  # (seq, heads, d_head)
        v = self.v(x).reshape((x.shape[0], self.n_heads, self.dim_per_head))
        k = self.k(x).reshape((x.shape[0], self.n_heads, self.dim_per_head))
        print(f"qvk_proj: Output q shape: {q.shape}, v shape: {v.shape}, k shape: {k.shape}")  # Debug: After qvk_proj

        return q, v, k




    @partial(profiler.annotate_function, name="__call__")
    def __call__(self, x, attn_bias):
        print(f"TransformerLayerShard: Input x shape: {x.shape}")  # Debug: Input to layer
        x = jax.lax.psum(x, axis_name='mp')
        x = self.norm(x)
        q, v, k = self.qvk_proj(x)
        attn_out = self.self_attn(q, v, k, attn_bias)
        # Combine heads back into the original dimensionality
        attn_out = attn_out.reshape((x.shape[0], x.shape[1], self.dim))  # seq_len, batch, d_model (4096)
        dense_out = self.ff(x)
        return jax.lax.psum(attn_out + dense_out, axis_name='mp')

    def decode_once(self, decode_state, x, attn_bias):
        x = jax.lax.psum(x, axis_name='mp')
        x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k = self.qvk_proj(x)

        v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[0]

        masked_tokens = length - tokens_decoded

        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        return jax.lax.psum(attn_out + dense_out, axis_name='mp'), {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    def get_init_decode_state(self, x, given_length, attn_bias):
        mesh = self.mesh_manager.get_mesh()
        with mesh:
            x = jax.lax.psum(x, 'mp')
            x = self.norm(x)

        q, v, k = self.qvk_proj(x)

        full_length = x.shape[0]
        masked_tokens = full_length - given_length

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))

        bias = -1e10 * (1. - causal_mask)
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)
    

        return attn_out + dense_out, {"k": k, "v": v, "tokens_decoded": given_length.astype(jnp.uint32)}


class ProjectionShard(nn.Module):
    config: dict

    def setup(self):
        self.dim_per_shard = self.config["dim_per_shard"]
        self.out_dim = self.config["out_dim"]
        self.shards = self.config["cores_per_replica"]
        self.mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(self.shards, -1), ("dp", "mp"))

    def loss(self, x, target, shard_start_index, z_loss):
        logits = self.forward(x)
        logits = jnp.swapaxes(logits, 0, 1)

        shard_start_index = jax.lax.psum(shard_start_index, "mp")
        predicted_logits = jnp.take_along_axis(logits, target[:, :, None] + shard_start_index, axis=-1)
        exp_logits = jnp.exp(logits - logits.max(axis=-1, keepdims=True))
        sum_exp_logits = jax.lax.psum(exp_logits, axis_name="mp")

        softmax_logits = predicted_logits - jnp.log(sum_exp_logits)

        if z_loss:
            z_loss_penalty = z_loss * jnp.square(jnp.log(sum_exp_logits)).mean()
        else:
            z_loss_penalty = 0

        return -(softmax_logits.mean() + z_loss_penalty), jnp.argmax(logits, axis=-1) == target

    def forward(self, x):
        x = nn.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = nn.Dense(self.out_dim)(x)
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
