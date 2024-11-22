import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable
import jax
import jax.numpy as jnp
from jax import vmap

from flax import struct

import kinax
from kinax.model import URDFSystem
from kinax.kinematics import forward
from kinax.skeleton import get_skeleton_from_system

from gtmp.kinematics.base import RobotBase
from gtmp.utils import transform_to_se3, motion_to_vec


def plot_coordinate_frame(ax, pos, rot, arrow_length=0.1, arrow_alpha=1.0, arrow_linewidth=1.0):

    x_basis = jnp.array([1, 0, 0])
    y_basis = jnp.array([0, 1, 0])
    z_basis = jnp.array([0, 0, 1])

    x_axis_target = rot @ x_basis
    y_axis_target = rot @ y_basis
    z_axis_target = rot @ z_basis

    # x-axis
    ax.quiver(pos[0], pos[1], pos[2],
              x_axis_target[0], x_axis_target[1], x_axis_target[2],
              length=arrow_length, normalize=True, color='red', alpha=arrow_alpha, linewidth=arrow_linewidth)
    # y-axis
    ax.quiver(pos[0], pos[1], pos[2],
              y_axis_target[0], y_axis_target[1], y_axis_target[2],
              length=arrow_length, normalize=True, color='green', alpha=arrow_alpha, linewidth=arrow_linewidth)
    # z-axis
    ax.quiver(pos[0], pos[1], pos[2],
              z_axis_target[0], z_axis_target[1], z_axis_target[2],
              length=arrow_length, normalize=True, color='blue', alpha=arrow_alpha, linewidth=arrow_linewidth)


@struct.dataclass
class Robot(RobotBase):

    system: URDFSystem = None

    @classmethod
    def create(
            cls,
            model_path: str,
            **kwargs: Any
    ):
        system = kinax.load_model(model_path)
        q_limits = jnp.stack(system.dof.limit).T[system.joint_ids]
        q_dim = q_limits.shape[0]

        return cls(
            system=system,
            q_dim=q_dim,
            q_limits=q_limits,
            **kwargs
        )

    def forward_kinematics_impl(self, q: jax.Array, qd: jax.Array, **kwargs: Any) -> Tuple[jax.Array, jax.Array]:
        x, xd = forward(self.system, q, qd)
        x, xd = vmap(transform_to_se3)(x), vmap(motion_to_vec)(xd)  # (links, 4, 4), (links, 6)
        return x, xd

    def forward_kinematics_kinax(self, q: jax.Array, qd: jax.Array, **kwargs: Any) -> Tuple:
        return forward(self.system, q, qd)

    def render(self, ax, q=None, color='blue', arrow_length=0.1, arrow_alpha=1.0, arrow_linewidth=1.0, **kwargs):
        # draw skeleton
        qd = jnp.zeros_like(q)
        skeleton = get_skeleton_from_system(self.system, q, qd)
        skeleton.draw_skeleton(ax=ax, c=color)

        # forward kinematics
        x, xd = self.forward_kinematics(q, qd)

        # draw EE frame
        frame_EE = x[-1]
        plot_coordinate_frame(
            ax, frame_EE[:3, -1], frame_EE[:3, :3],
            arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth
        )

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:  # trajs: (batch, num_timesteps, q_dim)
            for traj, color in zip(trajs, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.)
            if start_state is not None:
                self.render(ax, start_state, color='green')
            if goal_state is not None:
                self.render(ax, goal_state, color='purple')


if __name__ == '__main__':
    from kinax.model import FRANKA_PANDA_NO_GRIPPER, FRANKA_PANDA, UR10, UR10_SUCTION, TIAGO_DUAL_HOLO, SHADOW_HAND, ALLEGRO_HAND, PLANAR_2_LINK
    from collections import OrderedDict

    link_names_for_object_collision_checking = [
            # 'panda_link0',
            # 'panda_link1',
            'panda_link2',
            'panda_link3',
            # 'panda_link4',
            'panda_link5',
            # 'panda_link6',
            'panda_link7',
            'panda_hand',
            # self.link_name_ee,
    ]
    link_margins_for_object_collision_checking = [
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
    ]

    link_names_pairs_for_self_collision_checking = OrderedDict({
            'panda_hand': ['panda_link0', 'panda_link1', 'panda_link2'],
            'panda_link6': ['panda_link0', 'panda_link1', 'panda_link2'],
            'panda_link5': ['panda_link0', 'panda_link1', 'panda_link2'],
            'panda_link4': ['panda_link1']
        })
    
    robot = Robot.create(
        model_path=FRANKA_PANDA_NO_GRIPPER,
        link_names_for_object_collision_checking=link_names_for_object_collision_checking,
        link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
        link_names_pairs_for_self_collision_checking=link_names_pairs_for_self_collision_checking
    )
    q = jnp.array([0.012, -0.57, 0., -2.81 , 0., 3.037, 0.741])
    qd = jnp.zeros_like(q)
    x, xd = robot.forward_kinematics(q, qd)
    print(x.shape)
    print(xd.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    robot.render(ax, q)
    ax.set_aspect('equal')
    plt.show()
