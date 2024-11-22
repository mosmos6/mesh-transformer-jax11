import jax
import jax.numpy as jnp
import numpy as np
import os
from einops import rearrange, repeat
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


class MeshContextManager:
    def __init__(self, dp, mp):
        """
        Initializes the mesh for TPU v2-8 with 2x2x2 topology.
        
        :param dp: Number of data parallel devices.
        :param mp: Number of model parallel devices.
        """
        # Ensure dp * mp == total devices (8 for TPU v2-8)
        assert dp * mp == 8, f"Invalid dp ({dp}) and mp ({mp}) configuration for TPU v2-8."
        
        # Define the 3D mesh: (data parallel, model parallel, intra-core)
        devices = mesh_utils.create_device_mesh((2, 2, 2))
        
        # Map dp and mp to mesh axes
        # Assume dp spans the first dimension and mp spans the second
        self.mesh = Mesh(devices, axis_names=('dp', 'mp', 'core'))

    def get_mesh(self):
        """
        Returns the created mesh.
        """
        return self.mesh

    def __enter__(self):
        """
        Enters the mesh context.
        """
        return self.mesh.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the mesh context.
        """
        return self.mesh.__exit__(exc_type, exc_val, exc_tb)

