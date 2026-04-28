"""Embodiment-aware distance fields for robot collision checking."""
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap, lax

from gtmp.objectives.base import Field
from gtmp.utils import SE3_distance


def interpolate_points(X: jax.Array, num_interpolate: int = 2) -> jax.Array:
    """Interpolate points between consecutive links for finer collision checking.

    Parameters
    ----------
    X : jax.Array, shape (..., num_links, 3)
        Link positions.
    num_interpolate : int
        Number of interpolation points between each pair of links.

    Returns
    -------
    jax.Array
        Original points concatenated with interpolated points.
    """
    link_dim = X.shape[:-1]
    alpha = jnp.linspace(0, 1, num_interpolate + 2)[1 : num_interpolate + 1]
    alpha = alpha.reshape(tuple([1] * len(link_dim) + [-1, 1]))
    X_expanded = X[..., None, :]
    X_diff = jnp.diff(X_expanded, axis=-3)
    X_interp = X_expanded[..., :-1, :, :] + X_diff * alpha
    X_interp = X_interp.reshape(
        tuple(list(link_dim[:-1]) + [-1, num_interpolate, X.shape[-1]])
    )
    points = jnp.concatenate([X_expanded, X_interp], axis=-2)
    return points


class EmbodimentDistanceFieldBase(Field):
    """Base class for embodiment-aware distance fields."""

    link_idxs_for_collision_checking: jax.Array
    interpolate_collision_checking: bool = eqx.field(static=True, default=True)
    num_interpolated_points: int = eqx.field(static=True, default=2)
    collision_margins: float = eqx.field(static=True, default=0.0)
    cutoff_margin: float = eqx.field(static=True, default=0.001)

    def __call__(self, X: jax.Array) -> jax.Array:
        return self.compute_embodiment_signed_distances(X)

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_collisions(self, *args, **kwargs):
        raise NotImplementedError


class CollisionSelfField(EmbodimentDistanceFieldBase):
    """Self-collision distance field based on pairwise link distances."""

    idxs_links_distance_matrix: jax.Array = None

    def compute_embodiment_signed_distances(
        self, X: jax.Array, **kwargs
    ) -> jax.Array:
        """Compute pairwise distances between specified links.

        Parameters
        ----------
        X : jax.Array, shape (..., links, 3)
            Link positions.

        Returns
        -------
        jax.Array
            Minimum distances between specified link pairs.
        """

        dist_mat = jnp.linalg.norm(
            X[..., None, :] - X[..., None, :, :], axis=-1
        )
        idxs_tuple = tuple(zip(*self.idxs_links_distance_matrix))
        distances = dist_mat[..., idxs_tuple[0], idxs_tuple[1]]
        return distances

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        """Check for self-collisions."""
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        return signed_distances < margin


class CollisionObjectBase(EmbodimentDistanceFieldBase):
    """Base class for object collision checking."""

    def compute_embodiment_signed_distances(
        self, X: jax.Array, **kwargs
    ) -> jax.Array:
        return self.object_signed_distances(X, **kwargs)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        """Check for collisions with objects."""
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.object_signed_distances(X, **kwargs)
        collisions = signed_distances < margin
        # Reduce over points and objects
        any_collision = jnp.any(jnp.any(collisions, axis=-1), axis=-1)
        return any_collision

    @abstractmethod
    def object_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError


class CollisionObjectDistanceField(CollisionObjectBase):
    """Object collision field using SDF list."""

    sdf_list: tuple = ()

    def object_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        link_pos = X[..., :3, -1]
        link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]
        sdf = vmap(lambda i: lax.switch(i, self.sdf_list, link_pos))(
            jnp.arange(len(self.sdf_list))
        ).max(axis=0)
        return sdf


class EESE3DistanceField(Field):
    """End-effector SE(3) distance field for goal reaching."""

    target_H: jax.Array
    w_pos: float = eqx.field(static=True, default=1.0)
    w_rot: float = eqx.field(static=True, default=1.0)

    def __call__(self, X: jax.Array) -> jax.Array:
        """Negative distance to target (higher = closer = better)."""
        return -self.compute_ee_distance(X)

    def grad(self, X: jax.Array) -> jax.Array:
        """Compute gradient of the EE distance w.r.t. input."""

        def _cost(X: jax.Array):
            return self.compute_ee_distance(X).sum()


        return jax.grad(_cost)(X)

    def compute_ee_distance(self, X: jax.Array) -> jax.Array:
        """Compute SE(3) distance between X and target transform."""
        return SE3_distance(X, self.target_H, self.w_pos, self.w_rot)
