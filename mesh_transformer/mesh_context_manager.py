class MeshContextManager:
    def __init__(self, config):
        self.mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(config["cores_per_replica"], -1), ("dp", "mp"))

    def get_mesh(self):
        return self.mesh

    def __enter__(self):
        self.mesh.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mesh.__exit__(exc_type, exc_val, exc_tb)
