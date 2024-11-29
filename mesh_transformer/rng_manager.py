import jax
import jax.numpy as jnp


class RNGManager:
    def __init__(self, seed=0):
        """
        Initialize the RNGManager with a base seed.
        """
        self.base_rng = jax.random.PRNGKey(seed)
        self.current_rng = self.base_rng

    def split_keys(self, num_splits):
        """
        Split the current RNG into multiple sub-keys.
        """
        split_keys = jax.random.split(self.current_rng, num_splits)
        self.current_rng = split_keys[0]  # Update current_rng to the first subkey
        return split_keys

    def get_current_key(self):
        """
        Get the current RNG key.
        """
        return self.current_rng

    def reset(self, seed=None):
        """
        Reset the RNG to the initial seed or a new one.
        """
        if seed is not None:
            self.base_rng = jax.random.PRNGKey(seed)
        self.current_rng = self.base_rng
