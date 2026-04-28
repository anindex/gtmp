"""Composite cost functions for motion planning with FK-based collision."""
from abc import abstractmethod
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap

from gtmp.objectives.base import Field
from gtmp.kinematics.robot import Robot


class Cost(eqx.Module):
    """Abstract cost function base class."""

    dim: int = eqx.field(static=True, default=2)
    state_dim: int = eqx.field(static=True, default=4)
    traj_len: int = eqx.field(static=True, default=64)

    def __call__(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return self.eval(X, H=H, **kwargs)

    def compute_cost(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return self.eval(X, H=H, **kwargs)

    @abstractmethod
    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        ...


class CostInfinite(Cost):
    """Binary collision cost: 0 if free, inf if in collision."""

    cost_list: tuple = ()
    robot: Robot = eqx.field(static=True, default=None)
    buffer_dim: int = eqx.field(static=True, default=0)

    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        coll = self.get_collisions(X, H=H, **kwargs)
        return jnp.where(coll, jnp.inf, 0.0)

    def get_collisions(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        X_dims = X.shape[:-1]
        q, qd = X, jnp.zeros_like(X)

        q = q.reshape(-1, self.dim)
        qd = qd.reshape(-1, self.dim)
        if self.buffer_dim > 0:
            q = jnp.concatenate([q, jnp.zeros((q.shape[0], self.buffer_dim))], axis=-1)
            qd = jnp.concatenate([qd, jnp.zeros((qd.shape[0], self.buffer_dim))], axis=-1)
        H, Hd = vmap(self.robot.forward_kinematics)(q, qd)
        H = H.reshape(X_dims + H.shape[1:])
        Hd = Hd.reshape(X_dims + Hd.shape[1:])

        collisions = []
        for cost in self.cost_list:
            collisions.append(cost.get_collisions(X, H=H, **kwargs))
        collisions = jnp.stack(collisions, axis=0).any(axis=0)
        return collisions


class CostComposite(Cost):
    """Composite cost: sum of multiple cost functions with FK."""

    cost_list: tuple = ()
    robot: Robot = None
    first_order: bool = eqx.field(static=True, default=True)
    buffer_dim: int = eqx.field(static=True, default=0)
    current_trajs: jax.Array = None

    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        X_dims = X.shape[:-1]
        if self.first_order:
            q, qd = jnp.array_split(X, 2, axis=-1)
        else:
            q, qd = X, jnp.zeros_like(X)

        q = q.reshape(-1, self.dim)
        qd = qd.reshape(-1, self.dim)
        if self.buffer_dim > 0:
            q = jnp.concatenate([q, jnp.zeros((q.shape[0], self.buffer_dim))], axis=-1)
            qd = jnp.concatenate([qd, jnp.zeros((qd.shape[0], self.buffer_dim))], axis=-1)
        H, Hd = vmap(self.robot.forward_kinematics)(q, qd)
        H = H.reshape(X_dims + H.shape[1:])
        Hd = Hd.reshape(X_dims + Hd.shape[1:])

        costs = []
        for cost in self.cost_list:
            costs.append(cost.eval(X, H=H, current_trajs=self.current_trajs, **kwargs))
        costs = jnp.stack(costs, axis=0).sum(axis=0)
        return costs

    def get_collisions(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        X_dims = X.shape[:-1]
        if self.first_order:
            q, qd = jnp.array_split(X, 2, axis=-1)
        else:
            q, qd = X, jnp.zeros_like(X)

        q = q.reshape(-1, self.dim)
        qd = qd.reshape(-1, self.dim)
        if self.buffer_dim > 0:
            q = jnp.concatenate([q, jnp.zeros((q.shape[0], self.buffer_dim))], axis=-1)
            qd = jnp.concatenate([qd, jnp.zeros((qd.shape[0], self.buffer_dim))], axis=-1)
        H, Hd = vmap(self.robot.forward_kinematics)(q, qd)
        H = H.reshape(X_dims + H.shape[1:])
        Hd = Hd.reshape(X_dims + Hd.shape[1:])

        collisions = []
        for cost in self.cost_list:
            collisions.append(cost.get_collisions(X, H=H, **kwargs))
        collisions = jnp.stack(collisions, axis=0).any(axis=0)
        return collisions


class CostCollision(Cost):
    """Collision cost using a distance field."""

    K: float = eqx.field(static=True, default=1.0)
    field: Field = None

    @classmethod
    def create(cls, dim: int, traj_len: int = 1, field: Field = None, sigma: float = 1.0):
        return cls(dim=dim, traj_len=traj_len, K=1 / sigma**2, field=field)

    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return jnp.exp(self.field(H)) * self.K

    def get_collisions(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return self.field.get_collisions(H, **kwargs)
