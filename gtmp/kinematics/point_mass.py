"""Point mass robot kinematics for 2D/3D planning."""
from typing import Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import jax
import jax.numpy as jnp

from gtmp.kinematics.base import RobotBase


def plot_sphere(ax, center, pos, radius, cmap):
    """Plot a 3D sphere at the given position."""
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(
        x + center[0] + pos[0],
        y + center[1] + pos[1],
        z + center[2] + pos[2],
        cmap=cmap,
        alpha=1,
    )


class RobotPointMass(RobotBase):
    """Point mass robot - identity forward kinematics."""

    def forward_kinematics_impl(
        self, q: jax.Array, qd: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array]:
        return q, qd

    def render(self, ax, q=None, color="blue", cmap="Blues", margin=0.05, **kwargs):
        """Render point mass as circle (2D) or sphere (3D)."""
        if q is None:
            return
        if q.ndim == 1:
            if self.q_dim == 2:
                ax.add_patch(plt.Circle(q, margin, color=color, zorder=10))
            elif self.q_dim == 3:
                plot_sphere(ax, q, np.zeros_like(q), margin, cmap)
            else:
                raise NotImplementedError(f"Unsupported q_dim={self.q_dim}")
        elif q.ndim == 2:
            if q.shape[-1] == 2:
                patches = [plt.Circle(q_, margin, color=color) for q_ in q]
                ax.add_collection(mcoll.PatchCollection(patches, zorder=10))
            elif q.shape[-1] == 3:
                for q_ in q:
                    plot_sphere(ax, q_, np.zeros_like(q_), margin, cmap)
            else:
                raise NotImplementedError(f"Unsupported dim={q.shape[-1]}")
        else:
            raise NotImplementedError(f"Unsupported q.ndim={q.ndim}")

    def render_trajectories(
        self,
        ax,
        trajs=None,
        start_state=None,
        goal_state=None,
        colors=("blue",),
        linestyle="solid",
        **kwargs,
    ):
        """Render point mass trajectories."""
        if trajs is not None:
            trajs_pos = np.asarray(trajs)
            dim = trajs_pos.shape[-1]
            if dim == 3:
                segments = np.array(
                    list(zip(trajs_pos[..., 0], trajs_pos[..., 1], trajs_pos[..., 2]))
                ).swapaxes(1, 2)
                ax.add_collection(Line3DCollection(segments, colors=colors, linestyle=linestyle))
                points = trajs_pos.reshape(-1, 3)
                c = [c for seg, c in zip(segments, colors) for _ in range(seg.shape[0])]
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=c, s=4)
            else:
                segments = np.array(
                    list(zip(trajs_pos[..., 0], trajs_pos[..., 1]))
                ).swapaxes(1, 2)
                ax.add_collection(mcoll.LineCollection(segments, colors=colors, linestyle=linestyle))
                points = trajs_pos.reshape(-1, 2)
                c = [c for seg, c in zip(segments, colors) for _ in range(seg.shape[0])]
                ax.scatter(points[:, 0], points[:, 1], color=c, s=4)

        if start_state is not None:
            if len(start_state) == 3:
                ax.plot(start_state[0], start_state[1], start_state[2], "go", markersize=7)
            else:
                ax.plot(start_state[0], start_state[1], "go", markersize=7)

        if goal_state is not None:
            if len(goal_state) == 3:
                ax.plot(
                    goal_state[0], goal_state[1], goal_state[2],
                    marker="o", color="purple", markersize=7,
                )
            else:
                ax.plot(goal_state[0], goal_state[1], marker="o", color="purple", markersize=7)
