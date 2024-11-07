import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
import optax
from typing import NamedTuple
import chex
from jax.sharding import NamedSharding, PartitionSpec as P
from jax import devices
from mesh_transformer.mesh_context_manager import MeshContextManager


def maybe_shard(x, partition_spec):
    try:
        mesh = mesh_manager.get_mesh()
        sharding = NamedSharding(mesh, partition_spec)
        return with_sharding_constraint(x, sharding)
    except ValueError as e:
        print(f"Sharding error: {e}")
        return x


def gpt3_schedule(warmup_steps,
                  total_steps,
                  peak_lr,
                  end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, total_steps) / total_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2

    return sch


def global_norm(updates, use_psum=True):
    pre_sqrt = sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)])
    if use_psum:
        pre_sqrt = jax.lax.psum(pre_sqrt, "mp")
    return jnp.sqrt(pre_sqrt)


class ClipByGlobalNormState(NamedTuple):
    """The `clip_by_global_norm` transformation is stateless."""


def clip_by_global_norm(max_norm, use_psum=True) -> optax.GradientTransformation:
    """Clip updates using their global norm.

    References:
      [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

    Args:
      max_norm: the maximum global norm for an update.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return ClipByGlobalNormState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = global_norm(updates, use_psum=use_psum)
        trigger = g_norm < max_norm
        updates = jax.tree_map(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# Use Optax's built-in additive weight decay
def additive_weight_decay(weight_decay: float = 0.0) -> optax.GradientTransformation:
    return optax.additive_weight_decay(weight_decay)


def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


def to_f16(t):
    return jax.tree_map(lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, t)


# identity in forward pass, psum in backward
@jax.custom_vjp
def f_psum(x):
    axis_name = "mp"  # Define axis_name here to avoid passing as a parameter
    reduce_to_first = False  # Set directly as a boolean
    result = x
    if reduce_to_first:
        result = result[0]
    return result

def f_psum_fwd(x):
    return f_psum(x), None

def f_psum_bwd(_, g):
    axis_name = "mp"  # Define axis_name here for backward calculation
    return jax.lax.psum(g, axis_name),

f_psum.defvjp(f_psum_fwd, f_psum_bwd)

@jax.custom_vjp
def f_psum_first(x):
    return jax.lax.all_gather(x, 'mp')[0]

def f_psum_first_fwd(x):
    return f_psum_first(x), None

def f_psum_first_bwd(_, g):
    axis_name = "mp"
    return jax.lax.psum(g, axis_name),

f_psum_first.defvjp(f_psum_first_fwd, f_psum_first_bwd)

# identity in forward pass, pmean in backward
@jax.custom_vjp
def f_pmean(x):
    return x


def f_pmean_fwd(x):
    return f_psum(x), None


def f_pmean_bwd(_, g):
    return jax.lax.pmean(g, "mp"),


f_pmean.defvjp(f_pmean_fwd, f_pmean_bwd)


# psum in forward pass, identity in backward
@jax.custom_vjp
def g_psum(x):
    axis_name = "mp"  # Define axis_name here to avoid passing as a parameter
    reduce_to_first = False  # Set directly as a boolean
    result = jax.lax.psum(x, axis_name)
    if reduce_to_first:
        result = result[0]  # Narrow to the first index if reduction is desired
    return result

def g_psum_fwd(x):
    return g_psum(x), None

def g_psum_bwd(_, g):
    axis_name = "mp"  # Define axis_name here for backward calculation
    return jax.lax.psum(g, axis_name),

g_psum.defvjp(g_psum_fwd, g_psum_bwd)


@jax.custom_vjp
def g_psum_first(x):
    return jax.lax.all_gather(x, 'mp')[0]

def g_psum_first_fwd(x):
    return g_psum_first(x), None

def g_psum_first_bwd(_, g):
    axis_name = "mp"  # Define axis_name for backward calculation
    return jax.lax.psum(g, axis_name),

g_psum_first.defvjp(g_psum_first_fwd, g_psum_first_bwd)


def shard_axis(x, axis_size, axis_name='mp'):
    assert x.shape[0] % axis_size == 0

    x = x.reshape((axis_size, -1) + x.shape[1:])
    print(f"Before axis_index: x.shape = {x.shape}, axis_name = {axis_name}")

    x = x[jax.lax.axis_index(axis_name)]
    print(f"After axis_index: x.shape = {x.shape}, axis_name = {axis_name}")

    return x



def unshard_axis(x, axis_name='mp'):
    print(f"Before all_gather: x.shape = {x.shape}, axis_name = {axis_name}")
    x = jax.lax.all_gather(x, axis_name)
    print(f"After all_gather: x.shape = {x.shape}, axis_name = {axis_name}")

    x = x.reshape((-1, ) + x.shape[2:])
    print(f"After reshape: x.shape = {x.shape}, axis_name = {axis_name}")

    return x



# print but only on the first node
def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)

# Note Lines below 184 cause crash on TPU
if __name__ == "__main__":
   sch = gpt3_schedule(1_000, 20_000, 1e-4, 1e-5)

   for i in range(150):
      i = i * 200
      print(i, sch(i))
