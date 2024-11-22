from abc import ABC, abstractmethod

from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

from flax import struct
from jaxlie import SE3, SO3
from gtmp.objectives.base import Field
from gtmp.objectives.occupancy_map import OccupancyMap


@struct.dataclass
class PrimitiveDistanceField(Field):

    def __call__(self, X: jnp.array) -> jnp.array:
        return self.compute_distance(X)

    def compute_distance(self, X: jnp.array) -> jnp.array:
        return self.compute_distance_impl(X)

    @abstractmethod
    def compute_distance_impl(self, X: jnp.array) -> jnp.array:
        raise NotImplementedError()


@struct.dataclass
class PrimitiveSignedDistanceField(PrimitiveDistanceField):

    def __call__(self, X: jnp.array) -> jnp.array:
        return self.compute_signed_distance(X)

    def compute_signed_distance(self, X: jnp.array) -> jnp.array:
        return self.compute_signed_distance_impl(X)

    def compute_signed_distance_impl(self, X: jnp.array) -> jnp.array:
        return -self.compute_distance_impl(X)


@struct.dataclass
class SphereField(PrimitiveSignedDistanceField):
    """
    Sphere field that repels the agent from a sphere as SignedDistanceField.

    Parameters
    ----------
        centers : jax.Array  (num_spheres, dim)
        radii: jax.Array  (num_spheres, )
    """

    centers: jax.Array = None
    radii: jax.Array = None

    def __repr__(self):
        return f"MultiSphereField(centers={self.centers}, radii={self.radii})"

    def compute_distance_impl(self, X: jnp.array) -> jnp.array:
        """ Returns the distance field of X w.r.t. to the circle.
        Parameters
        ----------
            X : torch array (num_input, dim)

        Returns
        -------
            grad_field : torch array (num_input, )
        """
        dim = self.centers.shape[-1]
        distance_to_centers = jnp.linalg.norm(X[..., None, :dim] - self.centers + 1e-10, axis=-1)
        return (distance_to_centers - self.radii).min(axis=-1)

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        dist = self.compute_distance_impl(X, **kwargs)
        return dist < 0

    @staticmethod
    def is_inside(p, center, radius):
        # Check if point p is inside the discretized sphere
        return jnp.linalg.norm(p - center, axis=-1) <= radius


@struct.dataclass
class CylinderField(PrimitiveSignedDistanceField):
    """
    Cylinder field that repels the agent from a cylinder as SignedDistanceField.

    Parameters
    ----------
        p1 : jax.Array  (num_cylinders, 3)
        radii: jax.Array  (num_cylinders, )
        ori: jax.Array  (num_cylinders, 3)
    """

    p1: jax.Array = None
    centers: jax.Array = None
    radii: jax.Array = None
    ori: jax.Array = None
    lengths: jax.Array = None
    scaling: Optional[float] = 1.
    
    @classmethod
    def from_endpoints(cls, p1: jax.Array, p2: jax.Array, radii: jax.Array):
        ori = p2 - p1
        centers = (p1 + p2) / 2
        lengths = jnp.linalg.norm(ori, axis=-1)
        ori = ori / lengths[..., None]
        return cls(p1=p1, centers=centers, radii=radii, ori=ori, lengths=lengths)

    @classmethod
    def from_eulers(cls, centers: jax.Array, eulers: jax.Array, radii: jax.Array, lengths: jax.Array):
        so3 = vmap(SO3.from_rpy_radians, in_axes=(0, 0, 0))(*eulers.T)
        se3 = vmap(SE3.from_rotation_and_translation, in_axes=(0, 0))(so3, centers)
        lengths = lengths[..., None]
        local_z = jnp.concatenate((jnp.zeros((centers.shape[0], 2)), lengths / 2), axis=-1)
        local_mz = jnp.concatenate((jnp.zeros((centers.shape[0], 2)), -lengths / 2), axis=-1)
        p1 = se3.apply(local_z)
        p2 = se3.apply(local_mz)
        return cls.from_endpoints(p1, p2, radii)

    def __repr__(self):
        return f"MultiCylinderField(p1={self.p1}, radii={self.radii}, ori={self.ori})"

    def compute_distance_impl(self, X: jnp.array, **kwargs) -> jnp.array:
        dim = self.p1.shape[-1]
        vec_to_centers = X[..., None, :dim] - self.centers
        # project the vector to the axes
        distance_on_centers = jnp.einsum('...j,...j->...', vec_to_centers, self.ori)
        half_lengths = self.lengths / 2
        clamped_distance_on_centers = jnp.clip(distance_on_centers, -half_lengths, half_lengths)
        closest_point = self.centers + clamped_distance_on_centers[..., None] * self.ori
        radial_vector = X[..., None, :dim] - closest_point
        distance_radial = jnp.linalg.norm(radial_vector, axis=-1) - self.radii[None, ...]
        distance_height = jnp.maximum(jnp.abs(distance_on_centers) - half_lengths, 0.)
        dist = jnp.where(distance_radial > 0, jnp.sqrt(distance_radial ** 2 + distance_height ** 2), distance_radial)
        return dist.min(axis=-1)

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        dist = self.compute_distance_impl(X, **kwargs)
        return dist < 0

    def add_to_occupancy_map(self, obst_map: OccupancyMap) -> OccupancyMap:
        return obst_map

    def render(self, ax, offset=None, ori=None, color='gray', cmap='gray', **kwargs):
        return


@struct.dataclass
class CuboidField(PrimitiveSignedDistanceField):

    centers: jax.Array = None
    ori: SO3 = None
    r: jax.Array = None
    scaling: Optional[float] = 1.

    @classmethod
    def from_eulers(cls, centers: jax.Array, eulers: jax.Array, half_extents: jax.Array):
        so3 = vmap(SO3.from_rpy_radians, in_axes=(0, 0, 0))(*eulers.T)
        return cls(centers=centers, ori=so3, r=half_extents)

    def __repr__(self):
        return f"MultiCuboidField(p1={self.centers}, ori={self.ori}, r={self.r})"

    def compute_distance_impl(self, X: jnp.array, eps=1e-12) -> jnp.array:
        dim = self.centers.shape[-1]
        # rotate the vector to the local frame
        vec_from_centers = self.ori.inverse().apply(X[..., None, :dim] - self.centers)
        dist = jnp.abs(vec_from_centers) - self.r
        outside_dist = jnp.maximum(dist, eps)
        inside_dist = jnp.minimum(jnp.max(dist, axis=-1), 0.0)
        return (jnp.linalg.norm(outside_dist, axis=-1) + inside_dist).min(axis=-1)

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        dist = self.compute_distance_impl(X, **kwargs)
        return dist < 0

    def add_to_occupancy_map(self, obst_map: OccupancyMap) -> OccupancyMap:
        return obst_map

    def render(self, ax, offset=None, ori=None, color='gray', cmap='gray', **kwargs):
        return


@struct.dataclass
class LimitsField(PrimitiveSignedDistanceField):
    """
    SDF that repels the agent to stay away from some limits.

    Parameters
    ----------
        min : jax.Array  (num_limits, dim)
        max: jax.Array  (num_limits, dim)
        beta: jnp.float32
    """

    min: jax.Array = None
    max: jax.Array = None
    beta: Optional[float] = 1.

    def __repr__(self):
        return f"MultiLimitsField(min={self.min}, max={self.max})"

    def compute_distance_impl(self, X: jnp.array) -> jnp.array:
        """ Returns the distance field of X w.r.t. to the limits.
        Parameters
        ----------
            X : torch array (num_input, dim)

        Returns
        -------
            grad_field : torch array (num_input, num_centers)
        """
        dim = self.min.shape[-1]
        distance_to_min = jnp.linalg.norm((X[..., None, :dim] - self.min + 1e-10)[..., None], axis=-1)
        distance_to_max = jnp.linalg.norm((X[..., None, :dim] - self.max + 1e-10)[..., None], axis=-1)
        # Approx. a clip of the costs within the box, i.e. cost<=0, using the softplus function
        dist2bounds = jnp.minimum(distance_to_min, distance_to_max).min(axis=-1).min(axis=-1)
        # check outside the limits
        X_is_inside = (X[..., :dim] >= self.min) & (X[..., :dim] <= self.max)
        X_is_inside = X_is_inside.all(axis=-1)
        return jnp.where(X_is_inside, dist2bounds, -dist2bounds)
    
    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        dim = self.min.shape[-1]
        X_is_inside = (X[..., :dim] >= self.min) & (X[..., :dim] <= self.max)
        return ~X_is_inside.all(axis=-1)

