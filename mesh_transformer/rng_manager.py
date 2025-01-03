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
        # Use self.base_rng to generate the splits
        split_keys = jax.random.split(self.base_rng, num_splits)
        # Update the base_rng to the next key in the sequence
        self.base_rng = jax.random.fold_in(self.base_rng, 1)  # Advance base key  
        print(f"split_keys output shape: {split_keys.shape}")  # Debug
        return split_keys

    def get_current_key(self):
        """
        Get the current RNG key.
        """
        return self.base_rng

    def reset(self, seed=None):
        """
        Reset the RNG to the initial seed or a new one.
        """
        if seed is not None:
            self.base_rng = jax.random.PRNGKey(seed)
        self.current_rng = self.base_rng
