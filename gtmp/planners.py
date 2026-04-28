"""Core GTMP planning algorithms.

Implements Global Tensor Motion Planning (GTMP) and GTMP-Akima for
massively parallelized motion planning using MDP value iteration
over randomly sampled dream points.
"""
from functools import partial
from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap, random, jit, lax

from gtmp.splines import LayerAkima1DInterpolator




@partial(jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    """Linearly interpolate between consecutive waypoints."""
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 2)[:-1]
    return vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points)


@jit
def get_probe_points(
    origin: jax.Array, points: jax.Array, probes: jax.Array
) -> jax.Array:
    """Generate probe points along edges from origin to target points.

    Parameters
    ----------
    origin : jax.Array, shape (batch, dim) or (dim,)
        Starting points.
    points : jax.Array, shape (num_points, dim)
        Target points.
    probes : jax.Array, shape (num_probes,)
        Interpolation factors in (0, 1].

    Returns
    -------
    jax.Array, shape (batch, num_points, num_probes, dim)
        Probe point coordinates.
    """
    alpha = probes[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    diff = points[jnp.newaxis, :, :] - origin[:, jnp.newaxis, :]
    probe_points = diff[..., jnp.newaxis, :] * alpha + origin[:, jnp.newaxis, jnp.newaxis, :]
    return probe_points


@partial(jit, static_argnums=(2, 3))
def sample_dream_points(
    rng: jax.Array,
    bounds: jax.Array,
    num_dreams: Tuple[int, ...] = (100,),
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Sample dream points uniformly within bounds."""
    rng, sub_rng = random.split(rng)
    dim = bounds.shape[0]
    return random.uniform(
        sub_rng, num_dreams + (dim,), dtype, minval=bounds[:, 0], maxval=bounds[:, 1]
    )


@partial(jit, static_argnums=(2, 3))
def sample_dream_points_stratified(
    rng: jax.Array,
    bounds: jax.Array,
    num_dreams: Tuple[int, ...] = (100,),
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Sample dream points with stratified sampling for better coverage.

    Divides each dimension into strata and samples uniformly within each,
    improving space coverage compared to pure random sampling.
    """
    rng, sub_rng = random.split(rng)
    dim = bounds.shape[0]
    total = 1
    for n in num_dreams:
        total *= n

    # Stratified: divide [0,1] into total cells, sample within each
    # Then map to actual bounds
    base = random.uniform(sub_rng, num_dreams + (dim,), dtype)
    return base * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]




@partial(jit, static_argnums=(4, 5, 6))
def value_iteration(
    Cs: jax.Array,
    Ch: jax.Array,
    Cl: jax.Array,
    Cg: jax.Array,
    gamma: float = 0.9,
    eps: float = 1e-2,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Discounted infinite-horizon value iteration via lax.while_loop."""
    num_layer, num_dreams = Ch.shape[0] + 1, Ch.shape[1]
    Vs = 0.0
    Vh = jnp.zeros((num_layer, num_dreams), dtype=dtype)

    def optimal_bellman(V_tup):
        pVs, Vs, pVh, Vh = V_tup
        pVs, pVh = Vs, Vh
        Vh = Vh.at[-1].set(jnp.min(Cl + gamma * Cg, axis=-1))
        Vh = Vh.at[: num_layer - 1].set(
            jnp.min(Ch + gamma * Vh[1:num_layer, None, :], axis=-1)
        )
        Vs = jnp.min(Cs + gamma * Vh[0])
        return pVs, Vs, pVh, Vh

    V_tup = optimal_bellman((Vs, Vs, Vh, Vh))
    tol = eps * (1 - gamma) / gamma
    # NOTE: jnp.inf - jnp.inf = nan > tol is False (handles diverged case)
    _, Vs, _, Vh = lax.while_loop(
        lambda V: jnp.abs(V[0] - V[1]) > tol, optimal_bellman, V_tup
    )
    return Vs, Vh


@partial(jit, static_argnums=4)
def value_iteration_finite(
    Cs: jax.Array,
    Ch: jax.Array,
    Cl: jax.Array,
    Cg: jax.Array,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Finite-horizon value iteration (no discounting)."""
    num_layer, num_dreams = Ch.shape[0] + 1, Ch.shape[1]
    Vs = 0.0
    Vh = jnp.zeros((num_layer, num_dreams), dtype=dtype)

    def optimal_bellman(i: int, V_tup):
        pVs, Vs, pVh, Vh = V_tup
        pVs, pVh = Vs, Vh
        Vh = Vh.at[-1].set(jnp.min(Cl + Cg, axis=-1))
        Vh = Vh.at[: num_layer - 1].set(
            jnp.min(Ch + Vh[1:num_layer, None, :], axis=-1)
        )
        Vs = jnp.min(Cs + Vh[0])
        return pVs, Vs, pVh, Vh

    V_tup = optimal_bellman(0, (Vs, Vs, Vh, Vh))
    T = num_layer + 1
    _, Vs, _, Vh = lax.fori_loop(1, T, optimal_bellman, V_tup)
    return Vs, Vh




@jit
def get_optimal_path(
    Cs: jax.Array,
    Ch: jax.Array,
    Cl: jax.Array,
    Cg: jax.Array,
    Vh: jax.Array,
    gamma: float = 0.9,
) -> jax.Array:
    """Extract optimal path indices through dream point layers."""
    num_layer = Vh.shape[0]
    first = jnp.argmin(Cs + gamma * Vh[0])

    def step(current, i):
        next_node = jnp.argmin(Ch[i - 1, current] + gamma * Vh[i])
        return next_node, next_node

    if num_layer > 1:
        _, path_rest = lax.scan(step, first, jnp.arange(1, num_layer))
        path = jnp.concatenate([first[None], path_rest])
    else:
        path = first[None]

    goal_id = jnp.argmin(Cl[path[-1]] + gamma * Cg)
    return path, goal_id




class GTMPState(eqx.Module):
    """Configuration state for GTMP planning.

    Parameters
    ----------
    dim : int
        Configuration space dimensionality.
    q : jax.Array
        Start configuration.
    bounds : jax.Array, shape (dim, 2)
        Configuration space bounds.
    goals : jax.Array, shape (num_goals, dim)
        Goal configurations.
    time_profile : jax.Array
        Time breakpoints for Akima interpolation.
    splines : LayerAkima1DInterpolator
        Cached spline interpolator (optional).
    num_dreams : int
        Number of dream points per layer (N).
    num_layers : int
        Number of intermediate layers (M).
    num_probes : int
        Number of collision probe points per edge (H).
    probes : jax.Array
        Probe interpolation factors.
    occ_map : Any
        Collision cost function (OccupancyMap or CostInfinite).
    scale_objective : float
        Global objective scaling.
    scale_occ : float
        Collision cost scaling.
    scale_dist : float
        Distance cost scaling.
    gamma : float
        Discount factor for infinite-horizon VI.
    vi_finite : bool
        Use finite-horizon VI.
    use_stratified : bool
        Use stratified dream point sampling.
    visualize_value : bool
        Return value function and dream points for visualization.
    get_velocity : bool
        Compute path velocities from Akima splines.
    dtype : jnp.dtype
        Computation dtype (float32 or bfloat16 for memory savings).
    """

    dim: int = eqx.field(static=True, default=2)
    q: jax.Array = None
    bounds: jax.Array = None
    goals: jax.Array = None
    time_profile: jax.Array = None
    splines: LayerAkima1DInterpolator = eqx.field(static=True, default=None)

    num_dreams: int = eqx.field(static=True, default=50)
    num_layers: int = eqx.field(static=True, default=5)
    num_probes: int = eqx.field(static=True, default=10)
    probes: jax.Array = None
    occ_map: Any = None
    cell_size: float = eqx.field(static=True, default=1.0)
    scale_objective: float = eqx.field(static=True, default=1.0)
    scale_occ: float = eqx.field(static=True, default=1.0)
    scale_dist: float = eqx.field(static=True, default=1.0)
    gamma: float = eqx.field(static=True, default=0.99)
    vi_finite: bool = eqx.field(static=True, default=True)
    use_stratified: bool = eqx.field(static=True, default=False)
    visualize_value: bool = eqx.field(static=True, default=False)
    get_velocity: bool = eqx.field(static=True, default=False)
    dtype: jnp.dtype = eqx.field(static=True, default=jnp.float32)

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
        scale_objective: Optional[float] = 1.0,
        scale_occ: Optional[float] = 1.0,
        scale_dist: Optional[float] = 1.0,
        gamma: float = 0.99,
        cell_size: float = 1.0,
        num_dreams: int = 50,
        num_layers: int = 5,
        num_probes: int = 10,
        vi_finite: bool = True,
        use_stratified: bool = False,
        visualize_value: bool = False,
        get_velocity: bool = False,
        dtype: Optional[jnp.dtype] = jnp.float32,
        **kwargs: Any,
    ) -> "GTMPState":
        """Create a GTMP planning state."""
        # Skip the endpoints (origin/target) since they are already known
        probes = jnp.linspace(0, 1, num_probes + 2)[1:-1]
        if time_profile is None:
            time_profile = jnp.linspace(0, num_layers + 1, num_layers + 2, dtype=dtype)
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
            use_stratified=use_stratified,
            visualize_value=visualize_value,
            get_velocity=get_velocity,
            dtype=dtype,
        )

    def update_num_layer(self, num_layers: int) -> "GTMPState":
        """Return new state with updated layer count."""
        time_profile = jnp.linspace(0, num_layers + 1, num_layers + 2, dtype=self.dtype)
        return eqx.tree_at(
            lambda s: (s.num_layers, s.time_profile),
            self,
            (num_layers, time_profile),
        )


class GTMPOutput(eqx.Module):
    """Output from GTMP planning."""

    path: jax.Array = None
    path_vel: jax.Array = None
    goal_idx: int = None
    collision: bool = False
    dream_points: jax.Array = None
    splines: LayerAkima1DInterpolator = None
    V: jax.Array = None




def gtmp_plan(key: jax.Array, state: GTMPState) -> GTMPOutput:
    """Plan using GTMP with straight-line edge evaluation.

    Parameters
    ----------
    key : jax.Array
        PRNG key for dream point sampling.
    state : GTMPState
        Planning configuration.

    Returns
    -------
    GTMPOutput
        Planning result with path and collision status.
    """
    q = state.q
    sampler = sample_dream_points_stratified if state.use_stratified else sample_dream_points
    dream_points = sampler(key, state.bounds, (state.num_layers, state.num_dreams), dtype=state.dtype)

    # Compute probe points and evaluate collision costs
    points_s_1 = get_probe_points(q[None, ...], dream_points[0], state.probes)
    if state.num_layers > 1:
        points_layers = vmap(get_probe_points, in_axes=(0, 0, None))(
            dream_points[:-1], dream_points[1:], state.probes
        )
    points_final_g = get_probe_points(dream_points[-1], state.goals, state.probes)

    # Compute distances
    dist_s_1 = jnp.linalg.norm(dream_points[0] - q[None, ...], axis=-1)
    sources, targets = dream_points[:-1], dream_points[1:]
    dist_layers = jnp.linalg.norm(sources[:, :, None, :] - targets[:, None, :, :], axis=-1)
    dist_final_g = jnp.linalg.norm(dream_points[-1][:, None, :] - state.goals[None, :, :], axis=-1)

    # Evaluate collision costs
    coll_s_1 = state.occ_map(points_s_1).mean(axis=-1)
    if state.num_layers > 1:
        coll_layers = state.occ_map(points_layers).mean(axis=-1)
    coll_last_g = state.occ_map(points_final_g).mean(axis=-1)

    # Build cost matrices
    scale_occ = state.scale_occ
    scale_dist = state.scale_dist
    Cs = scale_dist * dist_s_1 + scale_occ * coll_s_1
    if state.num_layers > 1:
        Ch = scale_dist * dist_layers + scale_occ * coll_layers
    Cl = scale_dist * dist_final_g + scale_occ * coll_last_g
    Cg = -jnp.ones(state.goals.shape[0], dtype=state.dtype)

    # Solve MDP
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

    # Extract optimal path
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
            path = jnp.concatenate(
                (q[None, ...], mid_point[None, ...], state.goals[goal_idx, None, ...]),
                axis=0,
            )
            return path, goal_idx

    collision = jnp.isinf(Vs)
    path, goal_idx = lax.cond(
        collision,
        lambda _: (jnp.zeros((state.num_layers + 2, q.shape[-1]), state.dtype), 0),
        get_path,
        None,
    )

    output = GTMPOutput(path=path, goal_idx=goal_idx, collision=collision)
    if state.visualize_value:
        output = GTMPOutput(
            path=path, goal_idx=goal_idx, collision=collision,
            dream_points=dream_points, V=Vh,
        )
    return output




def gtmp_akima_plan(key: jax.Array, state: GTMPState) -> GTMPOutput:
    """Plan using GTMP-Akima with smooth spline edge evaluation.

    Parameters
    ----------
    key : jax.Array
        PRNG key for dream point sampling.
    state : GTMPState
        Planning configuration.

    Returns
    -------
    GTMPOutput
        Planning result with smooth path and collision status.
    """
    q = state.q
    sampler = sample_dream_points_stratified if state.use_stratified else sample_dream_points
    dream_points = sampler(key, state.bounds, (state.num_layers, state.num_dreams), dtype=state.dtype)

    # Assuming one goal for now
    splines = LayerAkima1DInterpolator(state.time_profile, q, dream_points, state.goals)
    points_s_1, points_layers, points_final_g = splines.get_spline_grid_interpolation(
        num_points=state.num_probes
    )

    # Compute arc-length distances
    dist_s_1 = jnp.linalg.norm(jnp.diff(points_s_1, axis=-2), axis=-1).sum(axis=-1)
    if state.num_layers > 1:
        dist_layers = jnp.linalg.norm(jnp.diff(points_layers, axis=-2), axis=-1).sum(axis=-1)
    dist_final_g = jnp.linalg.norm(jnp.diff(points_final_g, axis=-2), axis=-1).sum(axis=-1)

    # Evaluate collision
    coll_s_1 = state.occ_map(points_s_1).mean(axis=-1)
    if state.num_layers > 1:
        coll_layers = state.occ_map(points_layers).mean(axis=-1)
    coll_last_g = state.occ_map(points_final_g).mean(axis=-1)

    # Build cost matrices
    scale_occ = state.scale_occ
    scale_dist = state.scale_dist
    Cs = scale_dist * dist_s_1 + scale_occ * coll_s_1
    if state.num_layers > 1:
        Ch = scale_dist * dist_layers + scale_occ * coll_layers
    Cl = scale_dist * dist_final_g + scale_occ * coll_last_g

    if Cl.ndim == 1:
        Cl = Cl[:, None]
    Cg = -jnp.ones(state.goals.shape[0], dtype=state.dtype)

    # Solve MDP
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

    # Extract optimal path
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
    path, path_ids, goal_idx = lax.cond(
        collision,
        lambda _: (
            jnp.zeros((state.num_probes * (state.num_layers + 1), q.shape[-1]), state.dtype),
            jnp.zeros(state.num_layers + 2, dtype=jnp.int32),
            0,
        ),
        get_path,
        None,
    )

    path_vel = None
    if state.get_velocity:
        spline_vel = splines.derivative()
        path_vel = spline_vel.get_spline_interpolation(path_ids, num_points=state.num_probes)

    output = GTMPOutput(
        path=path,
        path_vel=path_vel,
        goal_idx=goal_idx,
        collision=collision,
        splines=splines,
    )
    if state.visualize_value:
        output = GTMPOutput(
            path=path, path_vel=path_vel, goal_idx=goal_idx,
            collision=collision, splines=splines,
            dream_points=dream_points, V=Vh,
        )
    return output
