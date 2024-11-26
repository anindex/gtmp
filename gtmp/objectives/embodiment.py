from abc import ABC, abstractmethod

from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable

import jax
import jax.numpy as jnp
from jax import vmap, lax

from flax import struct

from gtmp.objectives.base import Field


def interpolate_points(X: jax.Array, num_interpolate: int = 2) -> jax.Array:
    #TODO: check this
    link_dim = X.shape[:-1]
    alpha = jnp.linspace(0, 1, num_interpolate + 2)[1:num_interpolate + 1]
    alpha = alpha.reshape(tuple([1] * len(link_dim) + [-1, 1]))  # 1 x 1 x 1 x ... x num_interpolate x 1
    X = X[..., None, :]
    X_diff = jnp.diff(X, axis=-3)  # batch_dim x (num_link - 1) x 1 x 3
    X_interp = X[..., :-1, :, :] + X_diff * alpha  # batch_dim x (num_link - 1) x num_interpolate x 3
    X_interp = X_interp.reshape(tuple(list(link_dim[:-1]) + [-1, num_interpolate, X.shape[-1]]))  # batch_dim x (num_link - 1) * num_interpolate x 3
    points = jnp.concatenate([X, X_interp], axis=-2)  # batch_dim x (num_link + (num_link - 1) * num_interpolate) x 3
    return points


@struct.dataclass
class EmbodimentDistanceFieldBase(Field):

    link_idxs_for_collision_checking: jax.Array = None
    inpterpolate_collision_checking: bool = True
    num_interpolated_points: int = 2
    collision_margins: float = 0.0
    cutoff_margin: float = 0.001

    def __call__(self, X: jax.Array) -> jax.Array:
        return self.compute_embodiment_signed_distances(X)

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_collisions(self, *args, **kwargs):
        raise NotImplementedError


@struct.dataclass
class CollisionSelfField(EmbodimentDistanceFieldBase):

    idxs_links_distance_matrix: jax.Array = None

    def compute_embodiment_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:  # position tensor [batch_dim x links x 3]
        dist_mat = jnp.linalg.norm(X[..., None, :] - X[..., None, :, :], dim=-1)  # batch_dim x links x links

        # select only distances between pairs of specified links
        idxs_links_distance_matrix_tuple = tuple(zip(*self.idxs_links_distance_matrix))
        distances = dist_mat[..., idxs_links_distance_matrix_tuple[0], idxs_links_distance_matrix_tuple[1]]

        return distances


@struct.dataclass
class CollisionObjectBase(EmbodimentDistanceFieldBase):

    def compute_embodiment_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        return self.object_signed_distances(X, **kwargs)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        # position tensor
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.object_signed_distances(X, **kwargs)
        collisions = signed_distances < margin
        # reduce over points (dim -1) and over objects (dim -2)
        any_collision = jnp.any(jnp.any(collisions, axis=-1), axis=-1)
        return any_collision

    @abstractmethod
    def object_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError


@struct.dataclass
class CollisionObjectDistanceField(CollisionObjectBase):

    sdf_list: List[Field] = None

    def object_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        link_pos = X[..., :3, -1]  # batch_dims x links x 3
        link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]  # batch_dims x links x 3
        # link_pos = interpolate_points(link_pos, self.num_interpolated_points) if self.inpterpolate_collision_checking else link_pos
        sdf = vmap(lambda i: lax.switch(i, self.sdf_list, link_pos))(jnp.arange(len(self.sdf_list))).max(axis=0)
        return sdf


@struct.dataclass
class EESE3DistanceField(Field):

    target_H: jax.Array = None
    w_pos: float = 1.0
    w_rot: float = 1.0

    def __call__(self, X: jax.Array) -> jax.Array:
        """Evaluate the field at the given points.

        Args:
            x: A batch of points to evaluate the field at.

        Returns:
            The field evaluated at the given points.
        """
        return -self.compute_ee_distance(X)

    def grad(self, X: jax.Array) -> jax.Array:

        def _cost(X: jax.Array):
            return self.compute_ee_distance(X).sum()

        grad_fn = jax.grad(_cost)(X)
        return grad_fn(X)

    def compute_ee_distance(self, X: jax.Array) -> jax.Array:
        return SE3_distance(X, self.target_H, self.w_pos, self.w_rot)
