import jax
import jax.numpy as jnp
import numpy as np
import os
from einops import rearrange, repeat
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from mesh_transformer.util import f_psum, g_psum, maybe_shard, head_print
from mesh_transformer.mesh_context_manager import MeshContextManager  # Import from new file

class ReplicatedLayerNorm:
    def __init__(self, offset=True):
        self.offset = offset

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = jnp.ones(param_shape, inputs.dtype)
        scale = jax.lax.all_gather(scale, "mp")[0]

        offset = jnp.zeros(param_shape, inputs.dtype)
        offset = jax.lax.all_gather(offset, "mp")[0]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)

class RMSNorm:
    def __init__(self, offset, elementwise):
        self.offset = offset
        self.elementwise = elementwise

    def __call__(self, x):
        param_shape = (x.shape[-1],) if self.elementwise else ()
        normed = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)

        scale = jnp.full(param_shape, x.shape[-1] ** 0.5)
        scale = jax.lax.pmean(scale, "mp")
        normed = normed * scale

        if self.offset:
            offset = jnp.zeros(param_shape)
            offset = jax.lax.pmean(offset, "mp")
            normed = normed + offset

        return normed

def getnorm(type):
    if type == "layernorm":
        return ReplicatedLayerNorm()
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

class RelativePositionEmbs:
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

    def __call__(self, qlen, klen, heads, num_buckets):
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        relative_attention_bias = jnp.ones((heads, num_buckets)) * 0.02  # Initialize to some value

        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(relative_attention_bias.dtype)
        values = jax.lax.dot_general(relative_attention_bias, rp_bucket_one_hot,
                                     (((1,), (0,)), ((), ())))
        return values

def fixed_pos_embedding(seq_len, dim):
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))
    position = np.arange(0, seq_len, dtype=np.float32)
    sinusoid_inp = np.einsum('i,j->ij', position, inv_freq)
    sin = np.sin(sinusoid_inp)
    cos = np.cos(sinusoid_inp)
    sin = np.concatenate([sin, sin], axis=-1)
    cos = np.concatenate([cos, cos], axis=-1)
    return jnp.array(sin, dtype=jnp.float32), jnp.array(cos, dtype=jnp.float32)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    seq_len, batch_size, num_heads, head_dim = x.shape
    sin = repeat(sin, 'n d -> n b h d', b=batch_size, h=num_heads)[:, :, :, :head_dim]
    cos = repeat(cos, 'n d -> n b h d', b=batch_size, h=num_heads)[:, :, :, :head_dim]
    return (x * cos) + (rotate_every_two(x) * sin)

class EmbeddingShard:
    def __init__(self, n_vocab, d_model, cores_per_replica, seq_len, pe):
        in_dim = n_vocab
        out_dim = d_model
        shards = cores_per_replica

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.out_dim_per_shard = out_dim // shards

        if pe == "fixed":
            self.positional_embeddings = np.random.normal(0, 0.02, (seq_len, self.out_dim_per_shard))
        else:
            self.positional_embeddings = None

        self.proj = jax.nn.Dense(self.out_dim)

    def __call__(self, x, dtype=jnp.bfloat16):
        shard_start_index = jax.lax.axis_index('mp') * self.in_dim_per_shard

        input_onehot = jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        proj_out = self.proj(input_onehot)

        proj_out = g_psum(proj_out)

        if self.positional_embeddings is not None:
            all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'mp')
            all_pos_embed = jnp.transpose(all_pos_embed, (1, 0, 2)).reshape(-1, self.out_dim_per_shard)
            proj_out += all_pos_embed

        return proj_out

class TransformerLayerShard:
    def __init__(self, config, mesh_manager, init_scale=1.):
        self.config = config
        self.mesh_manager = mesh_manager
        heads = config["n_heads"]
        dim = config["d_model"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])
        self.is_rotary = config["pe"] == "rotary"

        assert dim % heads == 0
        assert heads % shards == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards
        self.pe_rotary_dims = config.get("pe_rotary_dims", self.dim_per_head)

        self.norm = norm

        self.q = jax.nn.Dense(self.dim_per_shard, use_bias=False)
        self.v = jax.nn.Dense(self.dim_per_shard, use_bias=False)
        self.k = jax.nn.Dense(self.dim_per_shard, use_bias=False)

        self.o = jax.nn.Dense(self.dim, use_bias=False, kernel_initializer=jax.nn.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

        self.dense_proj = jax.nn.Dense(self.dim_per_shard * 4)
        self.dense_proj_o = jax.nn.Dense(self.dim, kernel_initializer=jax.nn.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

    def self_attn(self, q, v, k, attn_bias):
        if self.is_rotary:
            print(f"self_attn: q.shape = {q.shape}, k.shape = {k.shape}")
            k_rot = k[:, :, :, :self.pe_rotary_dims]
            k_pass = k[:, :, :, self.pe_rotary_dims:]

            q_rot = q[:, :, :, :self.pe_rotary_dims]
            q_pass = q[:, :, :, self.pe_rotary_dims:]

            seq_len, batch_size, num_heads, _ = k_rot.shape

            sincos = fixed_pos_embedding(seq_len, self.pe_rotary_dims)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)
            k_rot = apply_rotary_pos_emb(k_rot, sincos)

            k = jnp.concatenate([k_rot, k_pass], axis=-1)
            q = jnp.concatenate([q_rot, q_pass], axis=-1)

        attention_logits = jnp.einsum("bthd,bThd->bhtT", q, k)
        print(f"attention_logits shape: {attention_logits.shape}")

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size
        print(f"attention_logits normalized shape: {attention_logits.shape}")

        attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        attention_vec = jnp.einsum("bhtT,bThd->bthd", attention_weights, v).reshape((-1, self.dim_per_shard))

        return self.o(attention_vec)

    def ff(self, x):
        dense_proj = self.dense_proj(x)
        dense_proj = jax.nn.gelu(dense_proj)
        return self.dense_proj_o(dense_proj)

    def qvk_proj(self, x):
        q = self.q(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        v = self.v(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        k = self.k(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))

        return q, v, k

    def __call__(self, x, attn_bias):
        print(f"Available axis names: {self.mesh_manager.get_mesh().axis_names}")  # Debug print
        x = jax.lax.psum(x, axis_name='mp')
        x = self.norm(x)
        q, v, k = self.qvk_proj(x)
        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias
        attn_out = self.self_attn(q, v, k, bias)
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
        mesh = self.mesh_manager.get_mesh()  # Use the already initialized MeshContextManager
        print(f"Mesh devices: {mesh.devices}")
        print(f"Mesh axis names: {mesh.axis_names}")

        with mesh:  # Ensure the mesh context is active
            print("Entering mesh context")
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

def compute_shard_start_index(dim_per_shard):
    return jax.lax.axis_index('mp') * dim_per_shard

class ProjectionShard:
    def __init__(self, config):
        self.dim_per_shard = config["dim_per_shard"]
        self.out_dim = config["out_dim"]
        self.shards = config["cores_per_replica"]
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
        x = jax.nn.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.Dense(self.out_dim)(x)
        return x

class Projection:
    def __init__(self, config):
        self.dim = config["n_vocab"]
        self.norm = jax.nn.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = jax.nn.Dense(self.dim)

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
