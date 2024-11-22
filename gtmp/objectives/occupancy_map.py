
from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable

import jax
import jax.numpy as jnp

from flax import struct


@struct.dataclass
class OccupancyMap:

    map_dim: jax.Array = struct.field(pytree_node=False)
    cmap_dim: Tuple[int] = struct.field(pytree_node=False)
    cell_size: jax.Array = struct.field(pytree_node=False)
    map: jax.Array = None
    coord: jax.Array = struct.field(default=None, pytree_node=False)
    origin: jax.Array = struct.field(default=None, pytree_node=False)
    limits: jax.Array = None
    infinite_cost: bool = struct.field(default=False, pytree_node=False)

    @classmethod
    def create(
        cls,
        limits: jax.Array, 
        cell_size: float,
        origin: Optional[jax.Array] = None,
        infinite_cost: bool = False,
        **kwargs: Any
    ):
        dim = limits.shape[0]
        map_dim = limits[:, 1] - limits[:, 0]
        cmap_dim = (map_dim / cell_size).astype(int)
        meshes = jnp.meshgrid(
            *[jnp.linspace(limits[i, 0], limits[i, 1], cmap_dim[i]) for i in range(dim)],
            indexing='ij'
        )
        coord = jnp.stack(meshes, axis=-1)
        map = jnp.zeros(cmap_dim)
        if origin is None:
            origin = cmap_dim // 2
        cmap_dim = tuple(cmap_dim)
        return cls(map_dim, cmap_dim, cell_size, map, coord, origin, limits, infinite_cost)

    @classmethod
    def from_prob(
        cls,
        prob: jax.Array,
        limits: jax.Array,
        threshold: float = 0.5,
        infinite_cost: bool = False,
        **kwargs: Any
    ):
        dim = limits.shape[0]
        map_dim = limits[:, 1] - limits[:, 0]
        cmap_dim = jnp.array(prob.shape)
        cell_size = map_dim / cmap_dim
        meshes = jnp.meshgrid(
            *[jnp.linspace(limits[i, 0], limits[i, 1], cmap_dim[i]) for i in range(dim)],
            indexing='ij'
        )
        coord = jnp.stack(meshes, axis=-1)
        origin = -limits[:, 0] / cell_size
        cmap_dim = tuple(cmap_dim)
        map = (prob > threshold).astype(int)
        return cls(map_dim, cmap_dim, cell_size, map, coord, origin, limits, infinite_cost)

    def __call__(self, X: jax.Array, **kwargs) -> jax.Array:
        return self.compute_cost(X, **kwargs)

    def set_occupancy(self, X: jnp.array, **kwargs):
        """
        Sets the occupancy grid to 1 at the given locations
        """
        X_dims = X.shape[:-1]
        X = X.reshape(-1, X.shape[-1])
        X_occ = X * (1 / self.cell_size) + self.origin
        X_occ = jnp.floor(X_occ).astype(int)

        # Project out-of-bounds locations to axis
        X_occ = jnp.clip(X_occ, 0, jnp.array(self.map.shape) - 1)

        # set occupancy
        return self.replace(map=self.map.at[tuple(X_occ.T)].set(1))
    
    def set_occupancy_from_field(self, field: Callable, threshold: float = -0.5, **kwargs):
        """
        Sets the occupancy grid to 1 at the given locations
        """
        X = self.coord.reshape(-1, self.coord.shape[-1])
        C = field(X)
        map = (C > threshold).astype(int)
        return self.replace(map=map.reshape(self.cmap_dim))

    def clear(self):
        return self.replace(map=jnp.zeros(self.cmap_dim))

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, horizon, task_spaces, position_dim)
        :return: collision cost on the trajectories
        """
        X_dims = X.shape[:-1]
        X = X.reshape(-1, X.shape[-1])
        X_occ = X * (1 / self.cell_size) + self.origin
        X_occ = jnp.floor(X_occ).astype(int)

        # Project out-of-bounds locations to axis
        X_occ = jnp.clip(X_occ, 0, jnp.array(self.map.shape) - 1)

        # check collisions
        collision_vals = self.map[tuple(X_occ.T)]
        return collision_vals.reshape(X_dims)

    def compute_cost(self, X: jnp.array, **kwargs):
        if self.infinite_cost:
            coll = self.get_collisions(X, **kwargs)
            return jnp.where(coll, jnp.inf, 0)
        else:
            return self.get_collisions(X, **kwargs)

    def compute_distance_impl(self, X: jnp.array, **kwargs):
        return self.get_collisions(X, **kwargs)

    def compute_distances(self, X, **kwargs):
        """
        Computes euclidean distances of X to all points in the occupied grid
        """
        X_dims = X.shape[:-1]
        X = X.reshape(-1, X.shape[-1])
        X_grid_point_idxs = jnp.stack(jnp.nonzero(self.map)).T
        if X_grid_point_idxs.shape[0] == 0:
            return jnp.ones((X.shape[0], 1)) * 1000
        X_grid_point_task_space = (X_grid_point_idxs - self.origin) * self.cell_size
        distances = jnp.linalg.norm(X_grid_point_task_space - X[:, None, :], axis=-1)
        distances = jnp.min(distances, axis=-1)
        return distances.reshape(X_dims)

    def plot(self, ax=None, save_dir=None, filename="obst_map.png"):
        import matplotlib.pyplot as plt
        dim = len(self.map_dim)
        if ax is None:
            if dim == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

        if dim == 2:
            rx, ry = self.map.shape
            x = jnp.linspace(self.limits[0][0], self.limits[0][1], rx)
            y = jnp.linspace(self.limits[1][0], self.limits[1][1], ry)
            ax.contourf(x, y, jnp.clip(self.map.T, 0, 1), 2, cmap='Greys')
        else:
            x, y, z = jnp.indices(jnp.array(self.map.shape) + 1, dtype=float)
            x -= self.origin[0]
            x = x * self.cell_size
            y -= self.origin[1]
            y = y * self.cell_size
            z -= self.origin[2]
            z = z * self.cell_size
            ax.voxels(y, x, z, self.map, facecolors='gray', edgecolor='black', shade=False, alpha=0.05)
