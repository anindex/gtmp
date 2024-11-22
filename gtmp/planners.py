import jax
import jax.numpy as jnp
from jax import vmap, random, jit, lax
from flax import struct
from typing import Tuple, Optional, Union, Any, List, Dict
from functools import partial

from gtmp.splines import LayerAkima1DInterpolator


@partial(jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 2)[:-1]
    return vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points)


@jit
def get_probe_points(origin: jax.Array, 
                     points: jax.Array,
                     probes: jax.Array) -> jax.Array:
    alpha = probes[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    diff = points[jnp.newaxis, :, :] - origin[:, jnp.newaxis, :]  # [batch, num_points, dim]
    probe_points = diff[..., jnp.newaxis, :] * alpha  + origin[:, jnp.newaxis, jnp.newaxis, :]  # [batch, num_points, num_probe, dim]
    return probe_points


@partial(jit, static_argnums=(2, 3))
def sample_dream_points(rng: jax.Array, bounds: jax.Array, num_dreams: Tuple[int] = (100,), dtype: jnp.dtype = jnp.float32) -> jax.Array:
    rng, sub_rng = random.split(rng)
    dim = bounds.shape[0]
    return random.uniform(sub_rng, num_dreams + (dim,), dtype, minval=bounds[:, 0], maxval=bounds[:, 1])


@partial(jit, static_argnums=6)
def value_iteration(Cs: jax.Array, Ch: jax.Array, Cl: jax.Array, Cg: jax.Array,
                    gamma: float = 0.9, eps: float = 1e-2, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    num_layer, num_dreams = Ch.shape[0] + 1, Ch.shape[1]
    Vs = 0.
    Vh = jnp.zeros((num_layer, num_dreams), dtype=dtype)

    def optimal_bellman(V_tup: Tuple[float, float, jax.Array, jax.Array]) -> Tuple[float, float, jax.Array, jax.Array]:
        pVs, Vs, pVh, Vh = V_tup
        pVs, pVh = Vs, Vh
        Vh = Vh.at[-1].set(jnp.min(Cl + gamma * Cg, axis=-1))
        Vh = Vh.at[:num_layer - 1].set(jnp.min(Ch + gamma * Vh[1:num_layer, None, :], axis=-1))
        Vs = jnp.min(Cs + gamma * Vh[0])
        return pVs, Vs, pVh, Vh

    V_tup = optimal_bellman((Vs, Vs, Vh, Vh))
    tol = eps * (1 - gamma) / gamma
    # NOTE: this works since jnp.inf - jnp.inf = nan > tol, nan > tol is False (diversed case)
    _, Vs, _, Vh = lax.while_loop(lambda V: jnp.abs(V[0] - V[1]) > tol, optimal_bellman, V_tup)
    return Vs, Vh


@jit
def get_optimal_path(Cs: jax.Array, Ch: jax.Array, Cl: jax.Array, Cg: jax.Array, Vh: jax.Array, gamma: float = 0.9) -> jax.Array:
    num_layer = Vh.shape[0]
    current = jnp.argmin(Cs + gamma * Vh[0])
    path = [current]
    for i in range(1, num_layer):
        current = jnp.argmin(Ch[i - 1, current] + gamma * Vh[i])
        path.append(current)
    goal_id = jnp.argmin(Cl[current] + gamma * Cg)
    return jnp.array(path), goal_id


@struct.dataclass
class GTMPState():
    dim: int = 2
    q: jax.Array = None
    bounds: jax.Array = struct.field(default=None, pytree_node=False)  # (dim, 2)
    goals: jax.Array = None  # (num_goals, dim)
    time_profile: jax.Array = None  # (num_layers + 2)
    splines: LayerAkima1DInterpolator = struct.field(default=None, pytree_node=False)

    num_dreams: int = struct.field(default=50, pytree_node=False)
    num_layers: int = struct.field(default=5, pytree_node=False)
    num_probes: int = struct.field(default=10, pytree_node=False)
    probes: jax.Array = struct.field(default=None, pytree_node=False)
    occ_map: Any = None
    cell_size: float = struct.field(default=1., pytree_node=False)
    scale_objective: float = struct.field(default=1., pytree_node=False)
    scale_occ: float = struct.field(default=1., pytree_node=False)
    scale_dist: float = struct.field(default=1., pytree_node=False)
    gamma: float = struct.field(default=0.99, pytree_node=False)
    sampling_free_space: bool = struct.field(default=False, pytree_node=False)
    visualize_value: bool = struct.field(default=False, pytree_node=False)
    get_velocity: bool = struct.field(default=False, pytree_node=False)
    dtype: jnp.dtype = struct.field(default=jnp.float32, pytree_node=False)

    @classmethod
    def create(
        cls,
        dim: int = 2,
        q: jax.Array = None,
        bounds: jax.Array = None,
        goals: jax.Array = None,
        time_profile: jax.Array = None,
        splines: LayerAkima1DInterpolator = None,
        occ_map: Any = None,
        scale_objective: Optional[float] = 1.,
        scale_occ: Optional[float] = 1.,
        scale_dist: Optional[float] = 1.,
        gamma: float = 0.99,
        cell_size: float = 1.0,
        num_dreams: int = 50,
        num_layers: int = 5,
        num_probes: int = 10,
        sampling_free_space: bool = False,
        visualize_value: bool = False,
        get_velocity: bool = False,
        dtype: Optional[jnp.dtype] = jnp.float32,
        **kwargs: Any,
    ) -> "GTMPState":
        probes = jnp.linspace(0, 1, num_probes + 2)[:-1]
        if time_profile is None:
            time_profile = jnp.linspace(0, num_layers + 1, num_layers + 2, dtype=dtype)
        # if occ_map is None:
        #     occ_map = OccupancyMap.create(bounds, cell_size, infinite_cost=True)
        return cls(
            q=q,
            dim=q.shape[-1] if q is not None else dim,
            occ_map=occ_map,
            bounds=bounds,
            goals=goals,
            time_profile=time_profile,
            splines=splines,
            num_dreams=num_dreams,
            num_layers=num_layers,
            num_probes=num_probes,
            probes=probes,
            scale_objective=scale_objective,
            scale_occ=scale_occ,
            scale_dist=scale_dist,
            cell_size=cell_size,
            gamma=gamma,
            sampling_free_space=sampling_free_space,
            visualize_value=visualize_value,
            get_velocity=get_velocity,
            dtype=dtype,
        )

    def update_num_layer(self, num_layers: int) -> "GTMPState":
        time_profile = jnp.linspace(0, num_layers + 1, num_layers + 2, dtype=self.dtype)
        return self.replace(num_layers=num_layers, time_profile=time_profile)


@struct.dataclass
class GTMPOutput():

    path: jax.Array = None
    path_vel: jax.Array = None
    goal_idx: int = None
    collision: bool = False
    dream_points: jax.Array = None
    splines: LayerAkima1DInterpolator = None
    V: jax.Array = None



def gtmp_plan(key: jax.Array, state: GTMPState) -> GTMPOutput:

    # sample subgoals
    q = state.q
    dream_points = sample_dream_points(key, state.bounds, (state.num_layers, state.num_dreams), dtype=state.dtype)

    points_s_1 = get_probe_points(q[None, ...], dream_points[0], state.probes)
    if state.num_layers > 1:
        points_layers = vmap(get_probe_points, in_axes=(0, 0, None))(dream_points[:-1], dream_points[1:], state.probes)
    points_final_g = get_probe_points(dream_points[-1], state.goals, state.probes)
    # compute distances
    dist_s_1 = jnp.linalg.norm(dream_points[0] - q[None, ...], axis=-1)
    sources, targets = dream_points[:-1], dream_points[1:]
    dist_layers = jnp.linalg.norm(sources[:, :, None, :] - targets[:, None, :, :], axis=-1)
    dist_final_g = jnp.linalg.norm(dream_points[-1][:, None, :] - state.goals[None, :, :], axis=-1)

    coll_s_1 = state.occ_map(points_s_1).mean(axis=-1)
    if state.num_layers > 1:
        coll_layers = state.occ_map(points_layers).mean(axis=-1)
    coll_last_g = state.occ_map(points_final_g).mean(axis=-1)

    del points_s_1
    if state.num_layers > 1:
        del points_layers
    del points_final_g

    scale_occ = state.scale_occ
    scale_dist = state.scale_dist
    Cs = scale_dist * dist_s_1 + scale_occ * coll_s_1
    if state.num_layers > 1:
        Ch = scale_dist * dist_layers + scale_occ * coll_layers
    Cl = scale_dist * dist_final_g + scale_occ * coll_last_g
    Cg = -jnp.ones(state.goals.shape[0], dtype=state.dtype)

    del dist_s_1
    del dist_final_g
    del coll_s_1
    del coll_last_g
    if state.num_layers > 1:
        del dist_layers
        del coll_layers

    # solve MDP
    if state.num_layers > 1:
        Vs, Vh = value_iteration(Cs, Ch, Cl, Cg, state.gamma, dtype=state.dtype)
    else:
        Cs = jnp.squeeze(Cs)
        Vh = jnp.min(Cl + state.gamma * Cg, axis=-1)
        Vs = jnp.min(Cs + state.gamma * Vh)

    # get optimal path
    if state.num_layers > 1:
        def get_path(_):
            mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, state.gamma)
            mid_path = dream_points[jnp.arange(state.num_layers), mid_idx, :]
            goal = state.goals[goal_idx]
            path = jnp.concatenate((q[None, ...], mid_path, goal[None, ...]), axis=0)
            return path, goal_idx
    else:
        def get_path(_):
            mid_idx = jnp.argmin(Cs + state.gamma * Vh)
            goal_idx = jnp.argmin(Cl[mid_idx] + state.gamma * Cg)
            mid_point = dream_points[0, mid_idx]
            path = jnp.concatenate((q[None, ...], mid_point[None, ...], state.goals[goal_idx, None, ...]), axis=0)
            return path, goal_idx
    collision = jnp.isinf(Vs)
    path, goal_idx = lax.cond(collision, lambda _: (jnp.zeros((state.num_layers + 2, q.shape[-1]), state.dtype), 0), get_path, None)

    # return distance to the subgoal
    output = GTMPOutput(
        path=path,
        goal_idx=goal_idx,
        collision=collision,
    )
    if state.visualize_value:
        output = output.replace(dream_points=dream_points, V=Vh)
    return output


def gtmp_akima_plan(key: jax.Array, state: GTMPState) -> GTMPOutput:

    # sample subgoals
    q = state.q
    dream_points = sample_dream_points(key, state.bounds, (state.num_layers, state.num_dreams), dtype=state.dtype)

    # assuming one goal for now
    splines = LayerAkima1DInterpolator(state.time_profile, q, dream_points, state.goals)
    points_s_1, points_layers, points_final_g = splines.get_spline_grid_interpolation(num_points=state.num_probes)
    # compute distances
    dist_s_1 = jnp.linalg.norm(jnp.diff(points_s_1, axis=-2), axis=-1).sum(axis=-1)
    if state.num_layers > 1:
        dist_layers = jnp.linalg.norm(jnp.diff(points_layers, axis=-2), axis=-1).sum(axis=-1)
    dist_final_g = jnp.linalg.norm(jnp.diff(points_final_g, axis=-2), axis=-1).sum(axis=-1)

    coll_s_1 = state.occ_map(points_s_1).mean(axis=-1)
    if state.num_layers > 1:
        coll_layers = state.occ_map(points_layers).mean(axis=-1)
    coll_last_g = state.occ_map(points_final_g).mean(axis=-1)

    del points_s_1
    if state.num_layers > 1:
        del points_layers
    del points_final_g

    scale_occ = state.scale_occ
    scale_dist = state.scale_dist
    Cs = scale_dist * dist_s_1 + scale_occ * coll_s_1
    if state.num_layers > 1:
        Ch = scale_dist * dist_layers + scale_occ * coll_layers
    Cl = scale_dist * dist_final_g + scale_occ * coll_last_g

    # free memory
    del dist_s_1
    del dist_final_g
    del coll_s_1
    del coll_last_g
    if state.num_layers > 1:
        del dist_layers
        del coll_layers

    if Cl.ndim == 1:
        Cl = Cl[:, None]
    Cg = -jnp.ones(state.goals.shape[0], dtype=state.dtype)

    # solve MDP
    if state.num_layers > 1:
        Vs, Vh = value_iteration(Cs, Ch, Cl, Cg, state.gamma, dtype=state.dtype)
    else:
        Cs = jnp.squeeze(Cs)
        Vh = jnp.min(Cl + state.gamma * Cg, axis=-1)
        Vs = jnp.min(Cs + state.gamma * Vh)

    # get optimal path
    if state.num_layers > 1:
        def get_path(_):
            mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, state.gamma)
            path_ids = jnp.append(mid_idx, goal_idx)
            path_ids = jnp.append(0, path_ids)
            path = splines.get_spline_interpolation(path_ids, num_points=state.num_probes)
            return path, path_ids, goal_idx
    else:
        def get_path(_):
            mid_idx = jnp.argmin(Cs + state.gamma * Vh)
            goal_idx = jnp.argmin(Cl[mid_idx] + state.gamma * Cg)
            path_ids = jnp.array([0, mid_idx, goal_idx])
            path = splines.get_spline_interpolation(path_ids, num_points=state.num_probes)
            return path, path_ids, goal_idx
    collision = jnp.isinf(Vs)
    path, path_ids, goal_idx = lax.cond(collision, lambda _: (jnp.zeros((state.num_probes * (state.num_layers + 1), q.shape[-1]), state.dtype), jnp.zeros(state.num_layers + 2, dtype=jnp.int32), 0), get_path, None)

    path_vel = None
    if state.get_velocity:
        spline_vel = splines.derivative()
        path_vel = spline_vel.get_spline_interpolation(path_ids, num_points=state.num_probes)

    # return distance to the subgoal
    output = GTMPOutput(
        path=path,
        path_vel=path_vel,
        goal_idx=goal_idx,
        collision=collision,
        splines=splines,
    )
    if state.visualize_value:
        output = output.replace(dream_points=dream_points, V=Vh)
    return output


@partial(jit, static_argnums=2)
def sample_free_points(key: jax.Array, state: GTMPState, num: int = 10000) -> jax.Array:
    bounds = state.bounds
    key, sub_rng = random.split(key)
    dim = bounds.shape[0]
    points = random.uniform(sub_rng, (num, dim), state.dtype, minval=bounds[:, 0], maxval=bounds[:, 1])
    occ = state.occ_map(points).astype(bool)
    dream_points = points[~occ][:state.num_layers*state.num_dreams].reshape(state.num_layers, state.num_dreams, dim)
    return dream_points
