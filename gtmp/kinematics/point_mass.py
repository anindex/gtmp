from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Any, Dict, Optional, Sequence, Tuple, List, Callable
import numpy as np
import jax

from flax import struct
from gtmp.kinematics.base import RobotBase


def plot_sphere(ax, center, pos, radius, cmap):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = radius * (np.cos(u) * np.sin(v))
    y = radius * (np.sin(u) * np.sin(v))
    z = radius * np.cos(v)
    ax.plot_surface(
        x + center[0] + pos[0], y + center[1] + pos[1], z + center[2] + pos[2],
        cmap=cmap,
        alpha=1
    )


@struct.dataclass
class RobotPointMass(RobotBase):

    def forward_kinematics_impl(self, q: jax.Array, qd: jax.Array, **kwargs: Any) -> Tuple[jax.Array, jax.Array]:
        return q, qd
    
    def render(self, ax, q=None, color='blue', cmap='Blues', margin_multiplier=1., **kwargs):
        if q is not None:
            margin = self.link_margins_for_object_collision_checking[0] * margin_multiplier
            if q.ndim == 1:
                if self.q_dim == 2:
                    circle1 = plt.Circle(q, margin, color=color, zorder=10)
                    ax.add_patch(circle1)
                elif self.q_dim == 3:
                    plot_sphere(ax, q, np.zeros_like(q), margin, cmap)
                else:
                    raise NotImplementedError
            elif q.ndim == 2:
                if q.shape[-1] == 2:
                    # ax.scatter(q[:, 0], q[:, 1], color=color, s=10 ** 2, zorder=10)
                    circ = []
                    for q_ in q:
                        circ.append(plt.Circle(q_, margin, color=color))
                        coll = mcoll.PatchCollection(circ, zorder=10)
                        ax.add_collection(coll)
                elif q.shape[-1] == 3:
                    # ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=color, s=10 ** 2, zorder=10)
                    for q_ in q:
                        plot_sphere(ax, q_, np.zeros_like(q_), margin, cmap)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_pos[..., 0], trajs_pos[..., 1], trajs_pos[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_pos, (-1, 3))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_pos[..., 0], trajs_pos[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_pos, (-1, 2))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)
        if start_state is not None:
            if len(start_state) == 3:
                ax.plot(start_state[0], start_state[1], start_state[2], 'go', markersize=7)
            else:
                ax.plot(start_state[0], start_state[1], 'go', markersize=7)
        if goal_state is not None:
            if len(goal_state) == 3:
                ax.plot(goal_state[0], goal_state[1], goal_state[2], marker='o', color='purple', markersize=7)
            else:
                ax.plot(goal_state[0], goal_state[1], marker='o', color='purple', markersize=7)
