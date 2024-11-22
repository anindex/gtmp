from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from brax.base import Motion
from brax.base import Transform
from brax.math import quat_to_3x3


def default_prng_key(rng = None):
    """Return a default PRNG key."""
    return jax.random.PRNGKey(0) if rng is None else rng


def transform_to_se3(transform: Transform) -> jnp.array:
    """Converts a brax Transform to a 4x4 SE(3) matrix."""
    rot_mat = quat_to_3x3(transform.rot)
    return jnp.concatenate([jnp.concatenate([rot_mat, transform.pos[:, None]], axis=1), jnp.array([[0., 0., 0., 1.]])])


def motion_to_vec(motion: Motion) -> jnp.array:
    return jnp.concatenate([motion.vel, motion.ang])
