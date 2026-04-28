"""Occupancy grid map for 2D/3D collision checking."""
from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class OccupancyMap(eqx.Module):
    """Discretized occupancy grid for collision checking.

    The grid maps workspace coordinates to binary occupied/free cells.
    Supports both 2D and 3D workspaces.
    """

    map_dim: jax.Array
    cmap_dim: Tuple[int, ...] = eqx.field(static=True)
    cell_size: jax.Array
    map: jax.Array
    coord: jax.Array
    origin: jax.Array
    limits: jax.Array
    infinite_cost: bool = eqx.field(static=True, default=False)

    @classmethod
    def create(
        cls,
        limits: jax.Array,
        cell_size: float,
        origin: Optional[jax.Array] = None,
        infinite_cost: bool = False,
        **kwargs: Any,
    ) -> "OccupancyMap":
        """Create an empty occupancy map from workspace limits.

        Parameters
        ----------
        limits : jax.Array, shape (dim, 2)
            Lower and upper bounds per dimension.
        cell_size : float
            Size of each grid cell.
        origin : jax.Array, optional
            Grid origin in cell coordinates.
        infinite_cost : bool
            If True, return inf for occupied cells instead of 1.
        """
        dim = limits.shape[0]
        map_dim = limits[:, 1] - limits[:, 0]
        cmap_dim = (map_dim / cell_size).astype(int)
        meshes = jnp.meshgrid(
            *[jnp.linspace(limits[i, 0], limits[i, 1], cmap_dim[i]) for i in range(dim)],
            indexing="ij",
        )
        coord = jnp.stack(meshes, axis=-1)
        occ_map = jnp.zeros(cmap_dim)
        if origin is None:
            origin = cmap_dim // 2
        cmap_dim_tuple = tuple(int(c) for c in cmap_dim)
        return cls(
            map_dim=map_dim,
            cmap_dim=cmap_dim_tuple,
            cell_size=cell_size,
            map=occ_map,
            coord=coord,
            origin=origin,
            limits=limits,
            infinite_cost=infinite_cost,
        )

    @classmethod
    def from_prob(
        cls,
        prob: jax.Array,
        limits: jax.Array,
        threshold: float = 0.5,
        infinite_cost: bool = False,
        **kwargs: Any,
    ) -> "OccupancyMap":
        """Create occupancy map from a probability grid.

        Parameters
        ----------
        prob : jax.Array
            Probability values per cell (e.g. from a sensor).
        limits : jax.Array, shape (dim, 2)
            Workspace bounds.
        threshold : float
            Probability above which a cell is considered occupied.
        infinite_cost : bool
            If True, return inf for occupied cells.
        """
        dim = limits.shape[0]
        map_dim = limits[:, 1] - limits[:, 0]
        cmap_dim = jnp.array(prob.shape)
        cell_size = map_dim / cmap_dim
        meshes = jnp.meshgrid(
            *[jnp.linspace(limits[i, 0], limits[i, 1], cmap_dim[i]) for i in range(dim)],
            indexing="ij",
        )
        coord = jnp.stack(meshes, axis=-1)
        origin = -limits[:, 0] / cell_size
        cmap_dim_tuple = tuple(int(c) for c in cmap_dim)
        occ_map = (prob > threshold).astype(int)
        return cls(
            map_dim=map_dim,
            cmap_dim=cmap_dim_tuple,
            cell_size=cell_size,
            map=occ_map,
            coord=coord,
            origin=origin,
            limits=limits,
            infinite_cost=infinite_cost,
        )

    def __call__(self, X: jax.Array, **kwargs) -> jax.Array:
        """Evaluate collision cost at points X."""
        return self.compute_cost(X, **kwargs)

    def set_occupancy(self, X: jax.Array, **kwargs) -> "OccupancyMap":
        """Return a new map with occupancy set at the given workspace points."""
        X_flat = X.reshape(-1, X.shape[-1])
        X_occ = X_flat * (1 / self.cell_size) + self.origin
        X_occ = jnp.floor(X_occ).astype(int)
        X_occ = jnp.clip(X_occ, 0, jnp.array(self.map.shape) - 1)
        new_map = self.map.at[tuple(X_occ.T)].set(1)
        return eqx.tree_at(lambda m: m.map, self, new_map)

    def set_occupancy_from_field(
        self, field: Callable, threshold: float = -0.5, **kwargs
    ) -> "OccupancyMap":
        """Set occupancy from a continuous field function."""
        X = self.coord.reshape(-1, self.coord.shape[-1])
        C = field(X)
        new_map = (C > threshold).astype(int).reshape(self.cmap_dim)
        return eqx.tree_at(lambda m: m.map, self, new_map)

    def clear(self) -> "OccupancyMap":
        """Return a new map with all cells cleared."""
        return eqx.tree_at(lambda m: m.map, self, jnp.zeros(self.cmap_dim))

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        """Check collision for batched points using the occupancy grid.

        Parameters
        ----------
        X : jax.Array, shape (..., dim)
            Points to check.

        Returns
        -------
        jax.Array, shape (...)
            Collision values (0 = free, 1 = occupied).
        """
        X_dims = X.shape[:-1]
        X_flat = X.reshape(-1, X.shape[-1])
        X_occ = X_flat * (1 / self.cell_size) + self.origin
        X_occ = jnp.floor(X_occ).astype(int)
        X_occ = jnp.clip(X_occ, 0, jnp.array(self.map.shape) - 1)
        collision_vals = self.map[tuple(X_occ.T)]
        return collision_vals.reshape(X_dims)

    def compute_cost(self, X: jax.Array, **kwargs) -> jax.Array:
        """Compute collision cost: either binary or infinite."""
        if self.infinite_cost:
            coll = self.get_collisions(X, **kwargs)
            return jnp.where(coll, jnp.inf, 0.0)
        else:
            return self.get_collisions(X, **kwargs)

    def plot(self, ax=None, save_dir=None, filename="obst_map.png"):
        """Visualize the occupancy map."""
        import matplotlib.pyplot as plt

        dim = len(self.map_dim)
        if ax is None:
            if dim == 2:
                _, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")

        if dim == 2:
            rx, ry = self.map.shape
            x = jnp.linspace(self.limits[0][0], self.limits[0][1], rx)
            y = jnp.linspace(self.limits[1][0], self.limits[1][1], ry)
            ax.contourf(x, y, jnp.clip(self.map.T, 0, 1), 2, cmap="Greys")
        else:
            x, y, z = jnp.indices(jnp.array(self.map.shape) + 1, dtype=float)
            x = (x - self.origin[0]) * self.cell_size
            y = (y - self.origin[1]) * self.cell_size
            z = (z - self.origin[2]) * self.cell_size
            ax.voxels(y, x, z, self.map, facecolors="gray", edgecolor="black", shade=False, alpha=0.05)
