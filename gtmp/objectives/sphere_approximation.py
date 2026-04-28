"""Sphere-approximation collision checking for articulated robots."""
from typing import Any, Dict, Tuple

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import yaml

from gtmp.files import get_data_config_path
from gtmp.objectives.base import Field
from gtmp.objectives.embodiment import EmbodimentDistanceFieldBase, CollisionObjectBase
from gtmp.kinematics.robot import Robot


@jit
def compute_minimum_self_dist(spheres: jax.Array, id1: int, id2: int) -> jax.Array:
    """Compute minimum distance between two sets of link spheres."""
    s1, s2 = spheres[id1], spheres[id2]
    s1_coord, s1_r = s1[:, :3], s1[:, 3]
    s2_coord, s2_r = s2[:, :3], s2[:, 3]
    link_dists = jnp.linalg.norm(s1_coord[:, None, :] - s2_coord[None, :, :], axis=-1)
    link_dists = link_dists - s1_r[:, None] - s2_r[None, :]
    return link_dists.min(axis=(-1, -2))


@jit
def compute_minimum_self_dist_over_links(spheres: jax.Array) -> jax.Array:
    """Compute minimum self-collision distance across all non-adjacent link pairs."""
    num_links = spheres.shape[0]
    links_a, links_b = jnp.triu_indices(num_links, 1)
    # Remove consecutive links from checking
    idx = np.flip(np.arange(2, 1 + num_links))
    idx[0] = 0
    idx = np.cumsum(idx)
    links_a = jnp.delete(links_a, idx)
    links_b = jnp.delete(links_b, idx)
    link_dists = vmap(compute_minimum_self_dist, in_axes=(None, 0, 0))(
        spheres, links_a, links_b
    ).min(axis=0)
    return link_dists


@jit
def transform_spheres(spheres: jax.Array, H: jax.Array) -> jax.Array:
    """Transform sphere centers from local to world frame using FK transforms."""
    pos, rot = H[:, :3, 3], H[:, :3, :3]
    transformed = jnp.einsum("bij,bkj->bik", spheres[:, :, :3], rot) + pos[:, None, :]
    return jnp.concatenate([transformed, spheres[:, :, 3:]], axis=-1)


def _load_link_spheres(robot_name: str, link_dict: Dict[str, int]):
    """Load sphere config and build link sphere arrays."""
    coll_yml = (get_data_config_path() / robot_name / "sphere_config.yaml").as_posix()
    with open(coll_yml) as file:
        coll_params = yaml.safe_load(file)

    link_spheres = []
    link_idxs = []
    for link_name, link_idx in link_dict.items():
        link_spheres.append(jnp.array(coll_params[link_name]))
        link_idxs.append(link_idx)
    return jnp.array(link_spheres), jnp.array(link_idxs)


class CollisionSphereSelfDistanceField(EmbodimentDistanceFieldBase):
    """Self-collision checking using sphere approximations."""

    link_spheres: jax.Array = None

    @classmethod
    def create(
        cls,
        robot_name: str = "panda",
        link_dict: Dict[str, int] = None,
        collision_margins: float = 0.001,
        cutoff_margin: float = 0.0,
        **kwargs: Any,
    ) -> "CollisionSphereSelfDistanceField":
        link_spheres, link_idxs = _load_link_spheres(robot_name, link_dict)
        return cls(
            link_spheres=link_spheres,
            link_idxs_for_collision_checking=link_idxs,
            collision_margins=collision_margins,
            cutoff_margin=cutoff_margin,
            **kwargs,
        )

    def compute_embodiment_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        H = X[..., self.link_idxs_for_collision_checking, :, :]
        H_dims = H.shape[:-3]
        H = H.reshape((-1,) + H.shape[-3:])
        transformed = vmap(transform_spheres, in_axes=(None, 0))(self.link_spheres, H)
        min_dist = vmap(compute_minimum_self_dist_over_links)(transformed)
        return -min_dist.reshape(H_dims)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        return signed_distances > -margin


class CollisionSphereObjectDistanceField(CollisionObjectBase):
    """Object collision checking using sphere approximations."""

    link_spheres: jax.Array = None
    sdf_list: tuple = ()

    @classmethod
    def create(
        cls,
        sdf_list: tuple = None,
        robot_name: str = "panda",
        link_dict: Dict[str, int] = None,
        collision_margins: float = 0.0,
        cutoff_margin: float = 0.001,
        **kwargs: Any,
    ) -> "CollisionSphereObjectDistanceField":
        link_spheres, link_idxs = _load_link_spheres(robot_name, link_dict)
        return cls(
            sdf_list=sdf_list,
            link_spheres=link_spheres,
            link_idxs_for_collision_checking=link_idxs,
            collision_margins=collision_margins,
            cutoff_margin=cutoff_margin,
            **kwargs,
        )

    def object_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        H = X[..., self.link_idxs_for_collision_checking, :, :]
        H_dims = H.shape[:-3]
        H = H.reshape((-1,) + H.shape[-3:])
        transformed = vmap(transform_spheres, in_axes=(None, 0))(self.link_spheres, H)
        coord, radii = transformed[..., :3], transformed[..., 3]
        sdfs = []
        for i in range(len(self.sdf_list)):
            sdfs.append((self.sdf_list[i](coord) + radii).max(axis=(-1, -2)))
        sdfs = jnp.stack(sdfs).max(axis=0)
        return sdfs.reshape(H_dims)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        return signed_distances > -margin


class CollisionSphere(CollisionObjectBase):
    """Combined self + object collision checking using spheres."""

    link_spheres: jax.Array = None
    sdf_list: tuple = ()

    @classmethod
    def create(
        cls,
        sdf_list: tuple = None,
        robot_name: str = "panda",
        link_dict: Dict[str, int] = None,
        collision_margins: float = 0.0,
        cutoff_margin: float = 0.001,
        **kwargs: Any,
    ) -> "CollisionSphere":
        link_spheres, link_idxs = _load_link_spheres(robot_name, link_dict)
        return cls(
            sdf_list=sdf_list,
            link_spheres=link_spheres,
            link_idxs_for_collision_checking=link_idxs,
            collision_margins=collision_margins,
            cutoff_margin=cutoff_margin,
            **kwargs,
        )

    def object_signed_distances(self, X: jax.Array, **kwargs) -> jax.Array:
        H = X[..., self.link_idxs_for_collision_checking, :, :]
        H_dims = H.shape[:-3]
        H = H.reshape((-1,) + H.shape[-3:])
        transformed = vmap(transform_spheres, in_axes=(None, 0))(self.link_spheres, H)

        # Object SDFs
        coord, radii = transformed[..., :3], transformed[..., 3]
        sdfs = []
        for i in range(len(self.sdf_list)):
            sdfs.append((self.sdf_list[i](coord) + radii).max(axis=(-1, -2)))
        if len(sdfs) > 0:
            sdfs = jnp.stack(sdfs).max(axis=0)
        else:
            sdfs = -jnp.inf

        # Self-collision
        min_dist = -vmap(compute_minimum_self_dist_over_links)(transformed)
        sdfs = jnp.maximum(sdfs, min_dist)

        return sdfs.reshape(H_dims)

    def get_collisions(self, X: jax.Array, **kwargs) -> jax.Array:
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        return signed_distances > -margin
