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

        print(f"CausalTransformerShard initialized with {layer_count} layers, {heads} heads, {shards} shards")

    def eval(self, context, target, z_loss=0., mask=0.0):
        print("Entering CausalTransformerShard eval")
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        else:
            attn_bias = 0

        attn_bias += mask

        x = hk.remat(self.embed)(context)

        for i, l in enumerate(self.transformer_layers):
            print(f"Processing layer {i} in eval")
            x = x + hk.remat(l)(x, attn_bias)

        result = hk.remat(self.proj.loss)(x, target, z_loss)
        print("CausalTransformerShard eval completed")
        return result

    def loss(self, ctx, tgt, z_loss=False, mask=0.0):
        print("Entering CausalTransformerShard loss")
        loss, correct = self.eval(ctx, tgt, float(z_loss), mask=mask)
        print("CausalTransformerShard loss computed")

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

        mp_per_host = min(mp, 8)

        assert mp == config["cores_per_replica"]

        # Define the device mesh
        devices = mesh_utils.create_device_mesh((dp, mp))
        mesh = Mesh(devices, axis_names=('dp', 'mp'))
        self.mesh = mesh  # Store the mesh for context management

        # Define in_specs and out_specs for each function
        in_specs_init = (P('dp', 'mp'), P('dp'))
        out_specs_init = P('dp', 'mp')

        in_specs_eval = (P('dp', 'mp'), P('dp'), P('dp'), P('dp'))
        out_specs_eval = P('dp', 'mp')

        in_specs_train = (P('dp', 'mp'), P('dp'), P('dp'))
        out_specs_train = (P('dp'), P('dp'), P('dp'), P('dp'), P('dp', 'mp'))

        in_specs_generate = (P('dp', 'mp'), P('dp'), P('dp'), P('dp'), P('dp'), P('dp'))
        out_specs_generate = (P('dp', 'mp'), P('dp'))

        in_specs_move = (P('dp', 'mp'), P('dp'))
        out_specs_move = P('dp', 'mp')

        # Create the shard_map functions
        self.init_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_init, out_specs=out_specs_init)(self.init)
        self.eval_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_eval, out_specs=out_specs_eval)(self.eval)
        self.train_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_train, out_specs=out_specs_train)(self.train)
        self.generate_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_generate, out_specs=out_specs_generate)(self.generate)
        self.move_shmap = partial(shard_map, mesh=mesh, in_specs=in_specs_move, out_specs=out_specs_move)(lambda x, _: to_bf16(x))

        # Generate PRNG keys
        key = jax.random.PRNGKey(42)
        total_devices = dp * mp
        keys = jax.random.split(key, total_devices).reshape(dp, mp, 2)

        example_shape = (max(dp // jax.process_count(), 1), config["seq"],)
        x = jax.random.uniform(key, example_shape, minval=0, maxval=config["n_vocab"]).astype(jnp.uint32)  # batch, len

        self.gen_length = 1
        with mesh:
            print("Initializing state with init_shmap...")
            self.state = self.init_shmap(keys, x)
            print("State initialized.")

        param_count = hk.data_structures.tree_size(self.state['params'])
        print(f"Total parameters: {param_count}")

    def init(self, key, x):
        print("Initializing model parameters...")
        def train_loss(x, y):
            transformer = CausalTransformerShard(self.config)
            return transformer.loss(x, y)

        param_init_fn = hk.transform(hk.experimental.optimize_rng_use(train_loss)).init

        # Reshape key to have the correct shape
        key = key[0]
        key = jnp.reshape(key, (2,))

        # Call param_init_fn with the correctly shaped key
        params = param_init_fn(key, x, x)
        print("Model parameters initialized.")

        return {
            "params": ("early_cast" in self.config and to_bf16 or to_f32)(params),
            "step": np.array(0),
            "opt_state": self.config["optimizer"].init(params)
        }

    def write_ckpt(self, path, shard=0):
        print(f"Writing checkpoint to {path}...")
        write_ckpt(self.state, path, shard)
        print("Checkpoint written.")

    def load_ckpt(self, path):
        print(f"Loading checkpoint from {path}...")
        self.state = read_ckpt(self.state, path, self.config["cores_per_replica"])
        print("Checkpoint loaded.")

    def train(self, sample):
        print("Starting training step...")
        obs = jnp.transpose(sample["obs"], (1, 0, 2))
        target = jnp.transpose(sample["target"], (1, 0, 2))
        loss, last_loss, grad_norm, grad_norm_micro, self.state = self.train_shmap(self.state, obs, target)
        print("Training step completed.")
        return np.array(loss).mean(), np.array(last_loss).mean(), np.array(grad_norm).mean(), np.array(grad_norm_micro).mean()

    def eval(self, sample):
        print("Starting evaluation step...")
        if "ctx_length" in sample:
            ctx_length = sample["ctx_length"]
        else:
            ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))
        eval_result = self.eval_shmap(self.state, sample["obs"], sample["target"], ctx_length)
        print("Evaluation step completed.")
        return eval_result

    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
        print("Starting text generation...")
        key = jax.random.PRNGKey(random.randint(0, 2 ** 60))
        batch_size = ctx.shape[0]
        aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
        self.gen_length = gen_length
        self.return_logits = return_logits
        keys = jax.random.split(key, batch_size)
        keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        generated_text = self.generate_shmap(self.state, keys, ctx, np.array(ctx_length, dtype=np.uint32), aux, sampler_options)
        print("Text generation completed.")
        return generated_text
