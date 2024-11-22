from abc import ABC, abstractmethod

from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable

import jax
import jax.numpy as jnp
from jax import random

from flax import struct

from gtmp.utils import default_prng_key


@struct.dataclass
class RobotBase(ABC):

    q_dim: int = None
    q_limits: jax.Array = None

    @classmethod
    def create(
            cls,
            q_limits: jax.Array,
            **kwargs: Any
    ):
        assert q_limits is not None, "q_limits cannot be None"
        q_dim = len(q_limits)

        return cls(
            q_dim=q_dim,
            q_limits=q_limits,
            **kwargs
        )

    def sample(self, rng: random.PRNGKey = None, num_samples: int = 10) -> jnp.array:
        rng = default_prng_key(rng)
        rng, rng2 = random.split(rng)
        samples = random.uniform(rng, shape=(num_samples, self.q_dim), minval=self.q_limits[:, 0], maxval=self.q_limits[:, 1])
        return samples

    def forward_kinematics(self, q: jnp.array, qd: jnp.array, **kwargs: Any) -> Tuple[jnp.array, jnp.array]:
        return self.forward_kinematics_impl(q, qd, **kwargs)

    @abstractmethod
    def forward_kinematics_impl(self, q: jnp.array, qd: jnp.array, **kwargs: Any) -> Tuple[jnp.array, jnp.array]:
        raise NotImplementedError

    @abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError
