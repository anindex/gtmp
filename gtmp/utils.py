"""Geometry and PRNG utilities for GTMP.

Uses jaxlie for SE(3) representations instead of brax.
"""
from typing import Optional

import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3


def default_prng_key(rng: Optional[jax.Array] = None) -> jax.Array:
    """Return a default PRNG key."""
    return jax.random.PRNGKey(0) if rng is None else rng


def quat_to_rotation_matrix(quat: jax.Array) -> jax.Array:
    """Convert a wxyz quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    quat : jax.Array, shape (..., 4)
        Quaternion in (w, x, y, z) convention.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Rotation matrix.
    """
    return SO3(quat).as_matrix()


def transform_to_se3(pos: jax.Array, rot: jax.Array) -> jax.Array:
    """Construct a 4x4 SE(3) matrix from position and wxyz quaternion.

    Parameters
    ----------
    pos : jax.Array, shape (3,)
        Translation vector.
    rot : jax.Array, shape (4,)
        Quaternion in (w, x, y, z) convention.

    Returns
    -------
    jax.Array, shape (4, 4)
        Homogeneous transformation matrix.
    """
    so3 = SO3(rot)
    se3 = SE3.from_rotation_and_translation(rotation=so3, translation=pos)
    return se3.as_matrix()


def motion_to_vec(vel: jax.Array, ang: jax.Array) -> jax.Array:
    """Concatenate linear and angular velocity into a 6D twist vector.

    Parameters
    ----------
    vel : jax.Array, shape (3,)
        Linear velocity.
    ang : jax.Array, shape (3,)
        Angular velocity.

    Returns
    -------
    jax.Array, shape (6,)
        Twist vector [vel, ang].
    """
    return jnp.concatenate([vel, ang])


def SE3_distance(
    T1: jax.Array,
    T2: jax.Array,
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    eps: float = 1e-4,
) -> jax.Array:
    """Compute weighted SE(3) distance between two homogeneous transforms.

    Parameters
    ----------
    T1 : jax.Array, shape (..., 4, 4)
        First transform.
    T2 : jax.Array, shape (..., 4, 4)
        Second transform.
    w_pos : float
        Weight for translational distance.
    w_rot : float
        Weight for rotational distance.
    eps : float
        Numerical epsilon for arccos clamping.

    Returns
    -------
    jax.Array, shape (...)
        Weighted distance.
    """
    R1, R2 = T1[..., :3, :3], T2[..., :3, :3]
    p1, p2 = T1[..., :3, 3], T2[..., :3, 3]

    # Rotation distance via trace
    R12 = jnp.einsum("...ij,...jk->...ik", R1.swapaxes(-2, -1), R2)
    trace = R12[..., 0, 0] + R12[..., 1, 1] + R12[..., 2, 2]
    cos_angle = jnp.clip((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    rot_dist = jnp.arccos(cos_angle)

    # Translation distance
    pos_dist = jnp.linalg.norm(p1 - p2, axis=-1)

    return w_pos * pos_dist + w_rot * rot_dist
