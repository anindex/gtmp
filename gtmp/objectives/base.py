from abc import ABC, abstractmethod

import jax.numpy as jnp

from flax import struct


@struct.dataclass
class Field(ABC):
    """Base class for all fields."""

    @abstractmethod
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the field at the given points.

        Args:
            X: A batch of points to evaluate the field at.

        Returns:
            The field evaluated at the given points.
        """
        pass
