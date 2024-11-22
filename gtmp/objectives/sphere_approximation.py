from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import yaml
from flax import struct

from gtmp.files import get_data_config_path
from gtmp.objectives.base import Field
from gtmp.objectives.embodiment import EmbodimentDistanceFieldBase, CollisionObjectBase
from gtmp.kinematics.robot import Robot


@jit
def compute_minimum_self_dist(spheres: jax.Array, id1: int, id2: int) -> jax.Array:
    s1, s2 = spheres[id1], spheres[id2]
    s1_coord, s1_r = s1[:, :3], s1[:, 3]
    s2_coord, s2_r = s2[:, :3], s2[:, 3]
    link_dists = jnp.linalg.norm(s1_coord[:, None, :] - s2_coord[None, :, :], axis=-1)
    link_dists = link_dists - s1_r[:, None] - s2_r[None, :]
    link_dists = link_dists.min(axis=(-1, -2))
    return link_dists


@jit
def compute_minimum_self_dist_over_links(spheres: jax.Array) -> jax.Array:
    # pairwise vmap
    num_links = spheres.shape[0]
    links_a, links_b = jnp.triu_indices(num_links, 1)
    # remove consecutive links checking
    idx = np.flip(np.arange(2, 1 + num_links))
    idx[0] = 0
    idx = np.cumsum(idx)
    links_a = jnp.delete(links_a, idx)
    links_b = jnp.delete(links_b, idx)
    link_dists = vmap(compute_minimum_self_dist, in_axes=(None, 0, 0))(spheres, links_a, links_b).min(axis=0)
    return link_dists


@jit
def transform_spheres(spheres: jax.Array, H: jax.Array) -> jax.Array:
    pos, rot = H[:, :3, 3], H[:, :3, :3]
    transformed_spheres = jnp.einsum('bij,bkj->bik', spheres[:, :, :3], rot) + pos[:, None, :]
    transformed_spheres = jnp.concatenate([transformed_spheres, spheres[:, :, 3:]], axis=-1)
    return transformed_spheres


@struct.dataclass
class CollisionSphereSelfDistanceField(EmbodimentDistanceFieldBase):

    link_spheres: jax.Array = None

    @classmethod
    def create(
        cls,
        robot_name: str = 'panda',
        link_dict: Dict[str, int] = None,
        collision_margins: float = 0.001,
        cutoff_margin: float = 0.,
        **kwargs: Any
    ):

        coll_yml = (get_data_config_path() / robot_name / 'sphere_config.yaml').as_posix()
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        link_spheres = []
        link_idxs_for_collision_checking = []
        for link_name, link_idx in link_dict.items():
            link_spheres.append(jnp.array(coll_params[link_name]))
            link_idxs_for_collision_checking.append(link_idx)
        link_idxs_for_collision_checking = jnp.array(link_idxs_for_collision_checking)

        return cls(link_spheres=jnp.array(link_spheres), 
                   link_idxs_for_collision_checking=link_idxs_for_collision_checking, 
                   collision_margins=collision_margins,
                   cutoff_margin=cutoff_margin, **kwargs)

    def compute_embodiment_signed_distances(self, X: jnp.array, **kwargs) -> jnp.array:
        H = X[..., self.link_idxs_for_collision_checking, :, :]
        H_dims = H.shape[:-3]
        H = H.reshape((-1,) + H.shape[-3:])
        transformed_link_spheres = vmap(transform_spheres, in_axes=(None, 0))(self.link_spheres, H)  # (batch_size, num_links, num_spheres, 4)
        min_sphere_dist = vmap(compute_minimum_self_dist_over_links)(transformed_link_spheres)
        min_sphere_dist = min_sphere_dist.reshape(H_dims)
        return -min_sphere_dist

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        # position tensor
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        collisions = signed_distances > -margin
        # reduce over links
        return collisions


@struct.dataclass
class CollisionSphereObjectDistanceField(CollisionObjectBase):

    link_spheres: jax.Array = None
    sdf_list: Tuple[Field] = None

    @classmethod
    def create(
        cls,
        sdf_list: Tuple[Field] = None,
        robot_name: str = 'panda',
        link_dict: Dict[str, int] = None,
        collision_margins: float = 0.0,
        cutoff_margin: float = 0.001,
        **kwargs: Any
    ):

        coll_yml = (get_data_config_path() / robot_name / 'sphere_config.yaml').as_posix()
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        link_spheres = []
        link_idxs_for_collision_checking = []
        for link_name, link_idx in link_dict.items():
            link_spheres.append(jnp.array(coll_params[link_name]))
            link_idxs_for_collision_checking.append(link_idx)
        link_idxs_for_collision_checking = jnp.array(link_idxs_for_collision_checking)

        return cls(sdf_list=sdf_list,
                   link_spheres=jnp.array(link_spheres), 
                   link_idxs_for_collision_checking=link_idxs_for_collision_checking, 
                   collision_margins=collision_margins,
                   cutoff_margin=cutoff_margin, **kwargs)

    def object_signed_distances(self, X: jnp.array, **kwargs) -> jnp.array:
        H = X[..., self.link_idxs_for_collision_checking, :, :]
        H_dims = H.shape[:-3]
        H = H.reshape((-1,) + H.shape[-3:])
        transformed_link_spheres = vmap(transform_spheres, in_axes=(None, 0))(self.link_spheres, H)  # (batch_size, num_links, num_spheres, 4)
        coord, radii = transformed_link_spheres[..., :3], transformed_link_spheres[..., 3]
        sdfs = []
        for i in range(len(self.sdf_list)):
            sdfs.append((self.sdf_list[i](coord) + radii).max(axis=(-1, -2)))  # max over spheres and links
        sdfs = jnp.stack(sdfs).max(axis=0)
        return sdfs.reshape(H_dims)

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        # position tensor
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        collisions = signed_distances > -margin
        # reduce over links
        return collisions


@struct.dataclass
class CollisionSphere(CollisionObjectBase):

    link_spheres: jax.Array = None
    sdf_list: Tuple[Field] = None

    @classmethod
    def create(
        cls,
        sdf_list: Tuple[Field] = None,
        robot_name: str = 'panda',
        link_dict: Dict[str, int] = None,
        collision_margins: float = 0.0,
        cutoff_margin: float = 0.001,
        **kwargs: Any
    ):

        coll_yml = (get_data_config_path() / robot_name / 'sphere_config.yaml').as_posix()
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        link_spheres = []
        link_idxs_for_collision_checking = []
        for link_name, link_idx in link_dict.items():
            link_spheres.append(jnp.array(coll_params[link_name]))
            link_idxs_for_collision_checking.append(link_idx)
        link_idxs_for_collision_checking = jnp.array(link_idxs_for_collision_checking)

        return cls(sdf_list=sdf_list,
                   link_spheres=jnp.array(link_spheres), 
                   link_idxs_for_collision_checking=link_idxs_for_collision_checking, 
                   collision_margins=collision_margins,
                   cutoff_margin=cutoff_margin, **kwargs)

    def object_signed_distances(self, X: jnp.array, **kwargs) -> jnp.array:
        H = X[..., self.link_idxs_for_collision_checking, :, :]
        H_dims = H.shape[:-3]
        H = H.reshape((-1,) + H.shape[-3:])
        transformed_link_spheres = vmap(transform_spheres, in_axes=(None, 0))(self.link_spheres, H)  # (batch_size, num_links, num_spheres, 4)
        
        # object sdf
        coord, radii = transformed_link_spheres[..., :3], transformed_link_spheres[..., 3]
        sdfs = []
        for i in range(len(self.sdf_list)):
            sdfs.append((self.sdf_list[i](coord) + radii).max(axis=(-1, -2)))  # max over spheres and links
        if len(sdfs) > 0:
            sdfs = jnp.stack(sdfs).max(axis=0)
        else:
            sdfs = -jnp.inf

        # self sdf
        min_sphere_dist = -vmap(compute_minimum_self_dist_over_links)(transformed_link_spheres)
        sdfs = jnp.maximum(sdfs, min_sphere_dist)

        return sdfs.reshape(H_dims)

    def get_collisions(self, X: jnp.array, **kwargs) -> jnp.array:
        # position tensor
        margin = self.collision_margins + self.cutoff_margin
        signed_distances = self.compute_embodiment_signed_distances(X, **kwargs)
        collisions = signed_distances > -margin
        # reduce over links
        return collisions


if __name__ == '__main__':
    from kinax.model import FRANKA_PANDA
    from gtmp.objectives.primitives import SphereField
    link_dict = {'panda_link1': 1, 'panda_link2': 2, 'panda_link3': 3, 'panda_link4': 4, 'panda_link5': 5, 'panda_link6': 6, 'panda_link7': 7, 'panda_hand': 8}
    link_idxs_for_object_collision_checking = jnp.array([2, 3, 5, 7, 9])
    link_margins_for_object_collision_checking = jnp.array([
            # 0.1,
            # 0.1,
            0.125,
            0.125,
            # 0.075,
            0.13,
            # 0.1,
            0.1,
            0.08,
            # 0.025,
    ])
    
    robot = Robot.create(
        model_path=FRANKA_PANDA
    )
    # q = jnp.array([0.012, -0.57, 0., -2.81 , 0., 3.037, 0.741])
    # q = jnp.tile(q[None, None, None, :], (1, 4, 16, 1))
    # qd = jnp.zeros_like(q)
    # x, xd = vmap(vmap(vmap(robot.forward_kinematics)))(q, qd)
    field = CollisionSphereSelfDistanceField.create(robot_name='panda', link_dict=link_dict)
    # print(field(x).shape)

    sphere_field = SphereField(
        centers=jnp.array([
            [0.3, 0.3, 0.5],
            [-0.3, 0.3, 0.5],
            [0.3, -0.3, 0.5],
            [-0.3, -0.3, 0.5],
        ]),
        radii=jnp.array([0.1, 0.1, 0.1, 0.1])
    )
    obj_field = CollisionSphere.create(sdf_list=(sphere_field,), robot_name='panda', link_dict=link_dict)

    # plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # q = jnp.array([0.012, -0.57, 0., 2.81 , 0., 1.037, 0.741, 0, 0])
    q = jnp.array([0.012, -0.57, 0., -2.81 , 0., 3.037, 0.741, 0, 0])
    qd = jnp.zeros_like(q)
    x, xd = robot.forward_kinematics(q, qd)
    # print(field.compute_embodiment_signed_distances(x))
    print(obj_field.compute_embodiment_signed_distances(x))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    robot.render(ax, q)
    # plot spheres
    H = x[..., field.link_idxs_for_collision_checking, :, :]
    transformed_link_spheres = transform_spheres(field.link_spheres, H)
    for link_sphere in transformed_link_spheres:
        ax.scatter(link_sphere[:, 0], link_sphere[:, 1], link_sphere[:, 2], s=1000)
    ax.scatter(sphere_field.centers[:, 0], sphere_field.centers[:, 1], sphere_field.centers[:, 2], s=1000, c='k')
    ax.set_aspect('equal')
    plt.show()
