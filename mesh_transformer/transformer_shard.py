from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import haiku as hk
import jax
import jax.numpy as jnp
import optax

# Assume these imports are defined properly
from mesh_transformer.util import to_f32, to_bf16, global_norm

class CausalTransformer:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]

        # Calculate dp and mp
        dp = jax.device_count() // config["cores_per_replica"]
        mp = config["cores_per_replica"]

        mp_per_host = min(mp, 8)

        assert mp == config["cores_per_replica"]

        # Define the device mesh
        devices = mesh_utils.create_device_mesh((dp, mp))
        mesh = Mesh(devices, axis_names=('dp', 'mp'))

        # Define in_specs and out_specs for each function
        in_specs_init = (P('mp'), P('dp'))
        out_specs_init = P('mp')

        in_specs_eval = (P('mp'), P('dp'), P('dp'), P('dp'))
        out_specs_eval = P('mp', 'dp')

        in_specs_train = (P('mp'), P('dp'), P('dp'))
        out_specs_train = (P('dp'), P('dp'), P('dp'), P('dp'), P('mp'))

        in_specs_generate = (P('mp'), P('dp'), P('dp'), P('dp'), P('dp'), P('dp'))
        out_specs_generate = (P('mp', 'dp'), P('dp'))

        in_specs_move = (P('mp'), P('dp'))
        out_specs_move = P('mp')

        # Create the shard_map functions
        self.init_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_init, out_specs=out_specs_init)(self.init)
        self.eval_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_eval, out_specs=out_specs_eval)(self.eval)
        self.train_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_train, out_specs=out_specs_train)(self.train)
        self.generate_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_generate, out_specs=out_specs_generate)(self.generate)
        self.move_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_move, out_specs=out_specs_move)(lambda x, _: to_bf16(x))

        # Generate PRNG keys using Haiku's key generator
        key = hk.PRNGSequence(42)
        keys = jnp.stack([next(key) for _ in range(mp_per_host)])

        example_shape = (max(dp // jax.process_count(), 1), config["seq"],)
        x = jax.random.uniform(jax.random.PRNGKey(42), example_shape, minval=0, maxval=config["n_vocab"]).astype(jnp.uint32)  # batch, len

        self.gen_length = 1
        self.state = self.init_shmap(keys, x)

        param_count = hk.data_structures.tree_size(self.state['params'])
        print(f"Total parameters: {param_count}")

    def init(self, key, x):
        def train_loss(x, y):
            transformer = CausalTransformerShard(self.config)
            return transformer.loss(x, y)

        param_init_fn = hk.transform(hk.experimental.optimize_rng_use(train_loss)).init
        params = param_init_fn(key, x, x)

        return {
            "params": ("early_cast" in self.config and to_bf16 or to_f32)(params),
            "step": np.array(0),
            "opt_state": self.config["optimizer"].init(params)
        }

    # Define eval, train, and generate methods similarly...

    def write_ckpt(self, path, shard):
        write_ckpt(self.state, path, shard)

    def load_ckpt(self, path):
        self.state = read_ckpt(self.state, path, config["cores_per_replica"])

    def train(self, sample):
        obs = jnp.transpose(sample["obs"], (1, 0, 2))
        target = jnp.transpose(sample["target"], (1, 0, 2))
        loss, last_loss, grad_norm, grad_norm_micro, self.state = self.train_shmap(self.state, obs, target)
        return np.array(loss).mean(), np.array(last_loss).mean(), np.array(grad_norm).mean(), np.array(grad_norm_micro).mean()

    def eval(self, sample):
        if "ctx_length" in sample:
            ctx_length = sample["ctx_length"]
        else:
            ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))
        return self.eval_shmap(self.state, sample["obs"], sample["target"], ctx_length)

    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))
        batch_size = ctx.shape[0]
        aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
        self.gen_length = gen_length
        self.return_logits = return_logits
        keys = jnp.stack([next(key) for _ in range(batch_size)])
        return self.generate_shmap(self.state, keys, ctx, np.array(ctx_length, dtype=np.uint32), aux, sampler_options)
