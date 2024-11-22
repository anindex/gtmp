
from abc import ABC, abstractmethod

from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, lax
from flax import struct

from gtmp.objectives.base import Field
from gtmp.kinematics.robot import Robot


@struct.dataclass
class Cost(ABC):

    dim: int = struct.field(default=2, pytree_node=False)
    state_dim: int = struct.field(default=4, pytree_node=False)
    traj_len: int = struct.field(default=64, pytree_node=False)

    def __call__(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return self.eval(X, H=H, **kwargs)

    def compute_cost(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return self.eval(X, H=H, **kwargs)

    @abstractmethod
    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        pass


@struct.dataclass
class CostInfinite(Cost):

    cost_list: Tuple[Cost] = None
    robot: Robot = struct.field(default=None, pytree_node=False)
    buffer_dim: int = struct.field(default=0, pytree_node=False)

    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        coll = self.get_collisions(X, H=H, **kwargs)
        return jnp.where(coll, jnp.inf, 0)

    def get_collisions(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:

        X_dims = X.shape[:-1]
        q, qd = X, jnp.zeros_like(X)

        q = q.reshape(-1, self.dim)
        qd = qd.reshape(-1, self.dim)
        if self.buffer_dim > 0:
            q = jnp.concatenate([q, jnp.zeros((q.shape[0], self.buffer_dim))], axis=-1)
            qd = jnp.concatenate([qd, jnp.zeros((qd.shape[0], self.buffer_dim))], axis=-1)
        H, Hd = vmap(self.robot.forward_kinematics)(q, qd)  # H: (batch_size, traj_len, num_links, 4, 4)
        H = H.reshape(X_dims + H.shape[1:])
        Hd = Hd.reshape(X_dims + Hd.shape[1:])

        collisions = []
        for cost in self.cost_list:
            collisions.append(cost.get_collisions(X, H=H, **kwargs))
        collisions = jnp.stack(collisions, axis=0).any(axis=0)
        return collisions

    def get_linear_system(self, X: jax.Array, H: jax.Array = None, **kwargs) -> Tuple[jax.Array, jax.Array, jax.Array]:
        pass



@struct.dataclass
class CostComposite(Cost):

    cost_list: Tuple[Cost] = None
    robot: Robot = None
    first_order: bool = struct.field(default=True, pytree_node=False)
    buffer_dim: int = struct.field(default=0, pytree_node=False)
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
        H, Hd = vmap(self.robot.forward_kinematics)(q, qd)  # H: (batch_size, traj_len, num_links, 4, 4)
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
        H, Hd = vmap(self.robot.forward_kinematics)(q, qd)  # H: (batch_size, traj_len, num_links, 4, 4)
        H = H.reshape(X_dims + H.shape[1:])
        Hd = Hd.reshape(X_dims + Hd.shape[1:])

        collisions = []
        for cost in self.cost_list:
            collisions.append(cost.get_collisions(X, H=H, **kwargs))
        collisions = jnp.stack(collisions, axis=0).any(axis=0)
        return collisions


@struct.dataclass
class CostCollision(Cost):

    K: float = struct.field(default=1.0, pytree_node=False)
    field: Field = None

    @classmethod
    def create(
        cls,
        dim: int,
        traj_len: int = 1,
        field: Field = None,
        sigma: float = 1.0,
    ):
        return cls(
            dim=dim,
            traj_len=traj_len,
            K=1/sigma**2,
            field=field
        )

    def eval(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:  # NOTE: assuming this cost always use task space representation
        # NOTE: rbf type is used for now
        return jnp.exp(self.field(H)) * self.K

    def get_collisions(self, X: jax.Array, H: jax.Array = None, **kwargs) -> jax.Array:
        return self.field.get_collisions(H, **kwargs)
