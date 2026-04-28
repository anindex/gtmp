"""Signed distance field primitives: sphere, cylinder, cuboid, limits."""
from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit, vmap

from jaxlie import SE3, SO3

from gtmp.objectives.base import Field
from gtmp.objectives.occupancy_map import OccupancyMap


class PrimitiveDistanceField(Field):
    """Base class for primitive distance fields."""

    def __call__(self, X: jax.Array) -> jax.Array:
        return self.compute_distance(X)

    def compute_distance(self, X: jax.Array) -> jax.Array:
        return self.compute_distance_impl(X)

    @abstractmethod
    def compute_distance_impl(self, X: jax.Array) -> jax.Array:
        raise NotImplementedError


class PrimitiveSignedDistanceField(PrimitiveDistanceField):
    """Signed distance field (negative = inside obstacle)."""

    def __call__(self, X: jax.Array) -> jax.Array:
        return self.compute_signed_distance(X)

    def compute_signed_distance(self, X: jax.Array) -> jax.Array:
        return self.compute_signed_distance_impl(X)

    def compute_signed_distance_impl(self, X: jax.Array) -> jax.Array:
        return -self.compute_distance_impl(X)


class SphereField(PrimitiveSignedDistanceField):
    """Sphere obstacle field.

    Parameters
    ----------
    centers : jax.Array, shape (num_spheres, dim)
        Sphere center positions.
    radii : jax.Array, shape (num_spheres,)
        Sphere radii.
    """

    centers: jax.Array
    radii: jax.Array

    def __repr__(self):
        return f"SphereField(centers={self.centers}, radii={self.radii})"

    def compute_distance_impl(self, X: jax.Array) -> jax.Array:
        dim = self.centers.shape[-1]
        distance_to_centers = jnp.linalg.norm(
            X[..., None, :dim] - self.centers + 1e-10, axis=-1
        )
        return (distance_to_centers - self.radii).min(axis=-1)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        return self.compute_distance_impl(X, **kwargs) < 0

    @staticmethod
    def is_inside(p, center, radius):
        return jnp.linalg.norm(p - center, axis=-1) <= radius


class CylinderField(PrimitiveSignedDistanceField):
    """Cylinder obstacle field.

    Parameters
    ----------
    p1 : jax.Array, shape (num_cylinders, 3)
        Start endpoints.
    centers : jax.Array, shape (num_cylinders, 3)
        Cylinder centers.
    radii : jax.Array, shape (num_cylinders,)
        Cylinder radii.
    ori : jax.Array, shape (num_cylinders, 3)
        Cylinder axis directions (unit vectors).
    lengths : jax.Array, shape (num_cylinders,)
        Cylinder lengths.
    """

    p1: jax.Array
    centers: jax.Array
    radii: jax.Array
    ori: jax.Array
    lengths: jax.Array

    @classmethod
    def from_endpoints(cls, p1: jax.Array, p2: jax.Array, radii: jax.Array):
        """Create cylinder field from endpoint pairs."""
        ori = p2 - p1
        centers = (p1 + p2) / 2
        lengths = jnp.linalg.norm(ori, axis=-1)
        ori = ori / lengths[..., None]
        return cls(p1=p1, centers=centers, radii=radii, ori=ori, lengths=lengths)

    @classmethod
    def from_eulers(
        cls,
        centers: jax.Array,
        eulers: jax.Array,
        radii: jax.Array,
        lengths: jax.Array,
    ):
        """Create cylinder field from center + euler angles."""
        so3 = vmap(SO3.from_rpy_radians, in_axes=(0, 0, 0))(*eulers.T)
        se3 = vmap(SE3.from_rotation_and_translation, in_axes=(0, 0))(so3, centers)
        lengths_col = lengths[..., None]
        local_z = jnp.concatenate(
            (jnp.zeros((centers.shape[0], 2)), lengths_col / 2), axis=-1
        )
        local_mz = jnp.concatenate(
            (jnp.zeros((centers.shape[0], 2)), -lengths_col / 2), axis=-1
        )
        p1 = se3.apply(local_z)
        p2 = se3.apply(local_mz)
        return cls.from_endpoints(p1, p2, radii)

    def __repr__(self):
        return f"CylinderField(centers={self.centers}, radii={self.radii})"

    def compute_distance_impl(self, X: jax.Array, **kwargs) -> jax.Array:
        dim = self.p1.shape[-1]
        vec_to_centers = X[..., None, :dim] - self.centers
        distance_on_centers = jnp.einsum("...j,...j->...", vec_to_centers, self.ori)
        half_lengths = self.lengths / 2
        clamped = jnp.clip(distance_on_centers, -half_lengths, half_lengths)
        closest_point = self.centers + clamped[..., None] * self.ori
        radial_vector = X[..., None, :dim] - closest_point
        distance_radial = jnp.linalg.norm(radial_vector, axis=-1) - self.radii[None, ...]
        distance_height = jnp.maximum(jnp.abs(distance_on_centers) - half_lengths, 0.0)
        dist = jnp.where(
            distance_radial > 0,
            jnp.sqrt(distance_radial**2 + distance_height**2),
            distance_radial,
        )
        return dist.min(axis=-1)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        return self.compute_distance_impl(X, **kwargs) < 0


class CuboidField(PrimitiveSignedDistanceField):
    """Cuboid (box) obstacle field.

    Parameters
    ----------
    centers : jax.Array, shape (num_cuboids, 3)
        Cuboid center positions.
    ori : SO3
        Cuboid orientations.
    r : jax.Array, shape (num_cuboids, 3)
        Half-extents.
    """

    centers: jax.Array
    ori: SO3
    r: jax.Array

    @classmethod
    def from_eulers(cls, centers: jax.Array, eulers: jax.Array, half_extents: jax.Array):
        """Create cuboid field from euler angles."""
        so3 = vmap(SO3.from_rpy_radians, in_axes=(0, 0, 0))(*eulers.T)
        return cls(centers=centers, ori=so3, r=half_extents)

    def __repr__(self):
        return f"CuboidField(centers={self.centers}, r={self.r})"

    def compute_distance_impl(self, X: jax.Array, eps=1e-12) -> jax.Array:
        dim = self.centers.shape[-1]
        vec_from_centers = self.ori.inverse().apply(X[..., None, :dim] - self.centers)
        dist = jnp.abs(vec_from_centers) - self.r
        outside_dist = jnp.maximum(dist, eps)
        inside_dist = jnp.minimum(jnp.max(dist, axis=-1), 0.0)
        return (jnp.linalg.norm(outside_dist, axis=-1) + inside_dist).min(axis=-1)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        return self.compute_distance_impl(X, **kwargs) < 0


class LimitsField(PrimitiveSignedDistanceField):
    """Workspace limits field - penalizes leaving the workspace.

    Parameters
    ----------
    min : jax.Array, shape (dim,)
        Lower bounds.
    max : jax.Array, shape (dim,)
        Upper bounds.
    """

    min: jax.Array
    max: jax.Array

    def __repr__(self):
        return f"LimitsField(min={self.min}, max={self.max})"

    def compute_distance_impl(self, X: jax.Array) -> jax.Array:
        dim = self.min.shape[-1]
        distance_to_min = jnp.linalg.norm(
            (X[..., None, :dim] - self.min + 1e-10)[..., None], axis=-1
        )
        distance_to_max = jnp.linalg.norm(
            (X[..., None, :dim] - self.max + 1e-10)[..., None], axis=-1
        )
        dist2bounds = jnp.minimum(distance_to_min, distance_to_max).min(axis=-1).min(axis=-1)
        X_is_inside = (X[..., :dim] >= self.min) & (X[..., :dim] <= self.max)
        X_is_inside = X_is_inside.all(axis=-1)
        return jnp.where(X_is_inside, dist2bounds, -dist2bounds)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        dim = self.min.shape[-1]
        X_is_inside = (X[..., :dim] >= self.min) & (X[..., :dim] <= self.max)
        return ~X_is_inside.all(axis=-1)
