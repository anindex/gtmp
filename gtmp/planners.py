import jax
import jax.numpy as jnp
from jax import vmap, random, jit, lax
from flax import struct
from typing import Tuple, Optional, Union, Any, List, Dict
from functools import partial

from gtmp.splines import LayerAkima1DInterpolator
from gtmp.dubins_splines import DubinsParams,dubins_path_planning,PathCandidate

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


@partial(jit, static_argnums=4)
def value_iteration_finite(Cs: jax.Array, Ch: jax.Array, Cl: jax.Array, Cg: jax.Array, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    num_layer, num_dreams = Ch.shape[0] + 1, Ch.shape[1]
    Vs = 0.
    Vh = jnp.zeros((num_layer, num_dreams), dtype=dtype)

    def optimal_bellman(i:int, V_tup: Tuple[float, float, jax.Array, jax.Array]) -> Tuple[float, float, jax.Array, jax.Array]:
        pVs, Vs, pVh, Vh = V_tup
        pVs, pVh = Vs, Vh
        Vh = Vh.at[-1].set(jnp.min(Cl + Cg, axis=-1))
        Vh = Vh.at[:num_layer - 1].set(jnp.min(Ch + Vh[1:num_layer, None, :], axis=-1))
        Vs = jnp.min(Cs + Vh[0])
        return pVs, Vs, pVh, Vh

    V_tup = optimal_bellman(0, (Vs, Vs, Vh, Vh))
    T = num_layer + 1
    _, Vs, _, Vh = lax.fori_loop(1, T, optimal_bellman, V_tup)
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
    # dim static
    dim: int = struct.field(default=2, pytree_node=False) 
    
    q: jax.Array = None
    bounds: jax.Array = struct.field(default=None, pytree_node=False)
    goals: jax.Array = None  # (num_goals, dim)
    # time_profile static
    time_profile: jax.Array = struct.field(default=None, pytree_node=False)
    
    splines: LayerAkima1DInterpolator = struct.field(default=None, pytree_node=False)

    num_dreams: int = struct.field(default=50, pytree_node=False)
    num_layers: int = struct.field(default=5, pytree_node=False)
    num_probes: int = struct.field(default=10, pytree_node=False)
    probes: jax.Array = struct.field(default=None, pytree_node=False)
    
    # occ_map static
    occ_map: Any = struct.field(default=None, pytree_node=False) 
    
    cell_size: float = struct.field(default=1., pytree_node=False)
    scale_objective: float = struct.field(default=1., pytree_node=False)
    scale_occ: float = struct.field(default=1., pytree_node=False)
    scale_dist: float = struct.field(default=1., pytree_node=False)
    gamma: float = struct.field(default=0.99, pytree_node=False)
    vi_finite: bool = struct.field(default=True, pytree_node=False)
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
        vi_finite: bool = True,
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
            vi_finite=vi_finite,
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
    gamma = 1.0 if state.vi_finite else state.gamma
    if state.num_layers > 1:
        if state.vi_finite:
            Vs, Vh = value_iteration_finite(Cs, Ch, Cl, Cg, dtype=state.dtype)
        else:
            Vs, Vh = value_iteration(Cs, Ch, Cl, Cg, gamma, dtype=state.dtype)
    else:
        Cs = jnp.squeeze(Cs)
        Vh = jnp.min(Cl + gamma * Cg, axis=-1)
        Vs = jnp.min(Cs + gamma * Vh)

    # get optimal path
    if state.num_layers > 1:
        def get_path(_):
            mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, gamma)
            mid_path = dream_points[jnp.arange(state.num_layers), mid_idx, :]
            goal = state.goals[goal_idx]
            path = jnp.concatenate((q[None, ...], mid_path, goal[None, ...]), axis=0)
            return path, goal_idx
    else:
        def get_path(_):
            mid_idx = jnp.argmin(Cs + gamma * Vh)
            goal_idx = jnp.argmin(Cl[mid_idx] + gamma * Cg)
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

def gtmp_dubins_plan(key: jax.Array, state: GTMPState, dubins_params: DubinsParams) -> GTMPOutput:
    q = state.q  # Shape: (3,) -> [x, y, theta]
    
    # 1. Sample 3D subgoal points [x, y, theta]
    # The bounds here must be (3, 2)
    dream_points = sample_dream_points(key, state.bounds, (state.num_layers, state.num_dreams), dtype=state.dtype)

    # 2. Define batch planning helper function with automatic collision computation (key for memory optimization)
    def compute_segment_cost(start_node, end_node):
        # Generate Dubins path (Path shape: [MAX_POINTS, 3])
        path, info = dubins_path_planning(start_node, end_node, dubins_params)
        # Strip angle dimension for collision detection [MAX_POINTS, 2]
        coll = state.occ_map(path[..., :2]).mean()
        return path, info.cost, coll

    # Wrap batch processing functions
    # batch_func: process (N,) target points
    batch_func = vmap(compute_segment_cost, in_axes=(None, 0))
    # pair_func: process (N, N) paired points
    pair_func = vmap(batch_func, in_axes=(0, None))

    # 3. Compute paths, distances and collisions for each segment
    # --- Start -> First layer ---
    # points_s_1: (N, P, 3), dist_s_1: (N,), coll_s_1: (N,)
    points_s_1, dist_s_1, coll_s_1 = batch_func(q, dream_points[0])

    # --- Inter-layer segments ---
    if state.num_layers > 1:
        # Use vmap to process L-1 inter-layer connections
        # Returns dimensions with one additional layer [L-1, N, N, ...]
        _, dist_layers, coll_layers = vmap(pair_func, in_axes=(0, 0))(dream_points[:-1], dream_points[1:])

    # --- Last layer -> Goal points ---
    # points_final_g: (N, G, P, 3), dist_final_g: (N, G), coll_last_g: (N, G)
    points_final_g, dist_final_g, coll_last_g = pair_func(dream_points[-1], state.goals)

    # Note: All del statements removed - del is ineffective in JIT-compiled JAX functions
    # JAX uses functional programming and automatically manages memory. del statements interfere with compiler optimization.
    
    scale_occ = state.scale_occ
    scale_dist = state.scale_dist
    Cs = scale_dist * dist_s_1 + scale_occ * coll_s_1
    if state.num_layers > 1:
        Ch = scale_dist * dist_layers + scale_occ * coll_layers
    Cl = scale_dist * dist_final_g + scale_occ * coll_last_g
    Cg = -jnp.ones(state.goals.shape[0], dtype=state.dtype)

    # solve MDP
    gamma = 1.0 if state.vi_finite else state.gamma
    if state.num_layers > 1:
        if state.vi_finite:
            Vs, Vh = value_iteration_finite(Cs, Ch, Cl, Cg, dtype=state.dtype)
        else:
            Vs, Vh = value_iteration(Cs, Ch, Cl, Cg, gamma, dtype=state.dtype)
    else:
        Cs = jnp.squeeze(Cs)
        Vh = jnp.min(Cl + gamma * Cg, axis=-1)
        Vs = jnp.min(Cs + gamma * Vh)

    # get optimal path
    def get_path(_):
        # 1. Find optimal node indices
        if state.num_layers > 1:
            mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, gamma)
        else:
            mid_idx = jnp.argmin(Cs + gamma * Vh)
            goal_idx = jnp.argmin(Cl[mid_idx] + gamma * Cg)

        # 2. Collect key pose nodes [Start, Mid1, Mid2..., Goal]
        mid_poses = dream_points[jnp.arange(state.num_layers), mid_idx, :]
        goal_pose = state.goals[goal_idx]
        all_nodes = jnp.concatenate([q[None, :], mid_poses, goal_pose[None, :]], axis=0)

        # 3. Regenerate and concatenate optimal Dubins curves
        # Use vmap to process (L+1) segment planning
        def plan_segment(i):
            p, _ = dubins_path_planning(all_nodes[i], all_nodes[i+1], dubins_params)
            return p
        
        # Get (L+1, MAX_POINTS, 3)
        segments = vmap(plan_segment)(jnp.arange(state.num_layers + 1))
        
        # Flatten and concatenate into continuous trajectory (Total_Points, 3)
        full_path = segments.reshape(-1, 2)
        return full_path, goal_idx
    
    collision = jnp.isinf(Vs)
    total_pts = (state.num_layers + 1) * state.num_probes
    
    path, goal_idx = lax.cond(
        collision, 
        lambda _: (jnp.zeros((total_pts, q.shape[-1]-1), state.dtype), 0), 
        get_path, 
        None
    )

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
    gamma = 1.0 if state.vi_finite else state.gamma
    if state.num_layers > 1:
        if state.vi_finite:
            Vs, Vh = value_iteration_finite(Cs, Ch, Cl, Cg, dtype=state.dtype)
        else:
            Vs, Vh = value_iteration(Cs, Ch, Cl, Cg, gamma, dtype=state.dtype)
    else:
        Cs = jnp.squeeze(Cs)
        Vh = jnp.min(Cl + gamma * Cg, axis=-1)
        Vs = jnp.min(Cs + gamma * Vh)

    # get optimal path
    if state.num_layers > 1:
        def get_path(_):
            mid_idx, goal_idx = get_optimal_path(Cs, Ch, Cl, Cg, Vh, gamma)
            path_ids = jnp.append(mid_idx, goal_idx)
            path_ids = jnp.append(0, path_ids)
            path = splines.get_spline_interpolation(path_ids, num_points=state.num_probes)
            return path, path_ids, goal_idx
    else:
        def get_path(_):
            mid_idx = jnp.argmin(Cs + gamma * Vh)
            goal_idx = jnp.argmin(Cl[mid_idx] + gamma * Cg)
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
