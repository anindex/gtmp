"""Abstract base class for scalar fields used as cost functions."""
from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp


class Field(eqx.Module):
    """Base class for all fields (cost landscapes, SDFs, etc.)."""

    @abstractmethod
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the field at the given points.

        Parameters
        ----------
        X : jax.Array
            A batch of points to evaluate the field at.

        Returns
        -------
        jax.Array
            The field evaluated at the given points.
        """
        ...
