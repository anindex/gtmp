"""URDF-based robot kinematics using kinax FK engine."""
from typing import Any, Tuple

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import vmap

import equinox as eqx

import kinax
from kinax.model import URDFSystem
from kinax.kinematics import forward
from kinax.skeleton import get_skeleton_from_system

from gtmp.kinematics.base import RobotBase
from gtmp.utils import transform_to_se3, motion_to_vec


def plot_coordinate_frame(
    ax,
    pos: jax.Array,
    rot: jax.Array,
    arrow_length: float = 0.1,
    arrow_alpha: float = 1.0,
    arrow_linewidth: float = 1.0,
):
    """Draw a 3D coordinate frame at the given position and orientation."""
    x_basis = jnp.array([1, 0, 0])
    y_basis = jnp.array([0, 1, 0])
    z_basis = jnp.array([0, 0, 1])

    x_axis_target = rot @ x_basis
    y_axis_target = rot @ y_basis
    z_axis_target = rot @ z_basis

    for axis, color in [(x_axis_target, "red"), (y_axis_target, "green"), (z_axis_target, "blue")]:
        ax.quiver(
            pos[0], pos[1], pos[2],
            axis[0], axis[1], axis[2],
            length=arrow_length,
            normalize=True,
            color=color,
            alpha=arrow_alpha,
            linewidth=arrow_linewidth,
        )


class Robot(RobotBase):
    """Robot with URDF-based forward kinematics via kinax."""

    system: URDFSystem = eqx.field(static=True)

    @classmethod
    def create(cls, model_path: str, **kwargs: Any) -> "Robot":
        """Load a robot from a URDF file.

        Parameters
        ----------
        model_path : str
            Path to the URDF file.

        Returns
        -------
        Robot
            Robot instance with loaded kinematics.
        """
        system = kinax.load_model(model_path)
        q_limits = jnp.stack(system.dof.limit).T[system.joint_ids]
        q_dim = q_limits.shape[0]
        return cls(system=system, q_dim=q_dim, q_limits=q_limits, **kwargs)

    def forward_kinematics_impl(
        self, q: jax.Array, qd: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute forward kinematics returning SE(3) matrices and twists."""
        x, xd = forward(self.system, q, qd)
        # Convert brax Transform/Motion to SE(3) matrices and twist vectors
        x_se3 = vmap(lambda xi: transform_to_se3(xi.pos, xi.rot))(x)
        xd_vec = vmap(lambda xdi: motion_to_vec(xdi.vel, xdi.ang))(xd)
        return x_se3, xd_vec

    def forward_kinematics_kinax(
        self, q: jax.Array, qd: jax.Array, **kwargs: Any
    ) -> Tuple:
        """Return raw kinax Transform/Motion objects."""
        return forward(self.system, q, qd)

    def render(
        self,
        ax,
        q=None,
        color="blue",
        arrow_length=0.1,
        arrow_alpha=1.0,
        arrow_linewidth=1.0,
        **kwargs,
    ):
        """Render robot skeleton and end-effector frame."""
        qd = jnp.zeros_like(q)
        skeleton = get_skeleton_from_system(self.system, q, qd)
        skeleton.draw_skeleton(ax=ax, c=color)

        x, _ = self.forward_kinematics(q, qd)
        frame_EE = x[-1]
        plot_coordinate_frame(
            ax,
            frame_EE[:3, -1],
            frame_EE[:3, :3],
            arrow_length=arrow_length,
            arrow_alpha=arrow_alpha,
            arrow_linewidth=arrow_linewidth,
        )

    def render_trajectories(
        self,
        ax,
        trajs=None,
        start_state=None,
        goal_state=None,
        colors=("gray",),
        **kwargs,
    ):
        """Render robot trajectories in 3D."""
        if trajs is not None:
            for traj, color in zip(trajs, colors):
                for t in range(traj.shape[0]):
                    self.render(
                        ax,
                        traj[t],
                        color,
                        arrow_length=0.1,
                        arrow_alpha=0.5,
                        arrow_linewidth=1.0,
                        **kwargs,
                    )
            if start_state is not None:
                self.render(ax, start_state, color="green")
            if goal_state is not None:
                self.render(ax, goal_state, color="purple")
