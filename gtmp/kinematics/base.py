"""Abstract base class for robot kinematics."""
from abc import ABC, abstractmethod
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from gtmp.utils import default_prng_key


class RobotBase(eqx.Module):
    """Base class for all robots with forward kinematics."""

    q_dim: int
    q_limits: jax.Array

    def sample(
        self, rng: jax.Array = None, num_samples: int = 10
    ) -> jax.Array:
        """Sample random joint configurations within limits."""
        rng = default_prng_key(rng)
        rng, sub_rng = random.split(rng)
        return random.uniform(
            sub_rng,
            shape=(num_samples, self.q_dim),
            minval=self.q_limits[:, 0],
            maxval=self.q_limits[:, 1],
        )

    def forward_kinematics(
        self, q: jax.Array, qd: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute forward kinematics."""
        return self.forward_kinematics_impl(q, qd, **kwargs)

    @abstractmethod
    def forward_kinematics_impl(
        self, q: jax.Array, qd: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError
