import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
from typing import NamedTuple


#two problem:1,theta1==0 or theta2==0; 2,ccc situations need to be checked carefully

# boundless cost
INF_COST = 1e9
TYPE_LSL = 0
TYPE_RSR = 1
TYPE_RSL = 2
TYPE_LSR = 3
TYPE_RLR = 4
TYPE_LRL = 5

class DubinsParams(NamedTuple):
    """Static parameters for the Dubins planner"""
    radius: float
    num_probes: int

class PathCandidate(NamedTuple):
    """Intermediate results containing path parameters"""
    cost: float          # path length
    beta_0: float        # param 1
    beta_1: float        # param 2
    beta_2: float        # param 3
    path_type: int       # path type ID (0-5)
    valid: bool          # path validity

@jit
def mod2pi(theta: float) -> float:
    """Normalize angle to [0, 2π) range"""
    return theta%(2.0 * jnp.pi)
@jit
def ortho(vect2d: jnp.array) -> jnp.array:
    """Computes an orthogonal vector to the one given"""
    return jnp.array((-vect2d[1], vect2d[0]))
@jit
def dist(pt_a: jnp.array, pt_b: jnp.array) -> float:
    """Euclidian distance between two (x, y) points"""
    return jnp.sqrt(jnp.sum((pt_a - pt_b) ** 2))

@jit
def find_center(point: jnp.array, side_sign: float, radius: float) -> jnp.array:
    """Finds the center of the turning circle given a point and side ('L' or 'R')"""
    angle = point[2] + side_sign*(jnp.pi / 2.0)
    return jnp.array((point[0] + jnp.cos(angle) * radius,
                      point[1] + jnp.sin(angle) * radius))
#@if whether the 'LR' should be considered
#@better to return theta(this only return the x,y position)

@jit
def circle_arc(reference: jnp.array, beta: float, center: jnp.array, x: float,radius: float) -> jnp.array:
    """Computes a point in a circular arc segment"""
    sign=jnp.sign(beta)
    sign=jnp.where(sign==0,1.0,sign)
    angle = reference[2] + ((x / radius) - jnp.pi / 2.0) * sign
    vect = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    return center + radius * vect

@jit
def solve_LSL(start: jnp.array, end: jnp.array, radius: float) -> PathCandidate:
    c_start = find_center(start, 1.0, radius)#L
    c_end = find_center(end, 1.0, radius)#L
    dist_centers = dist(c_start, c_end)
    alpha = jnp.arctan2((c_end - c_start)[1], (c_end-c_start)[0])
    beta_2 = mod2pi(end[2] - alpha)
    beta_0 = mod2pi(alpha - start[2])
    #jax.debug.print("alpha: {}", alpha)
    #jax.debug.print("angle1: {}", start[2])
    #jax.debug.print("angle2: {}", end[2])
    straight_dist = dist_centers
    total_len = radius * (beta_2 + beta_0) + straight_dist
    return PathCandidate(total_len, beta_0, straight_dist, beta_2 ,TYPE_LSL,True)

@jit
def solve_RSR(start: jnp.array, end: jnp.array, radius: float) -> PathCandidate:
    c_start = find_center(start, -1.0, radius) # R
    c_end = find_center(end, -1.0, radius)     # R
    
    dist_centers = dist(c_start, c_end)
    
    alpha = jnp.arctan2(c_end[1] - c_start[1], c_end[0] - c_start[0])
    
    beta_0 = mod2pi(start[2] - alpha)
    beta_2 = mod2pi(alpha - end[2])
    straight_dist = dist_centers
    
    total_len = radius * (beta_0 + beta_2) + straight_dist
    
    # Note: In RSR, we store positive beta values and negate them when generating points
    return PathCandidate(total_len, beta_0, straight_dist, beta_2, TYPE_RSR, True)


@jit
def solve_LSR(start: jnp.array, end: jnp.array, radius: float) -> PathCandidate:
    c_start = find_center(start, 1.0, radius) # L
    c_end = find_center(end, -1.0, radius)    # R

    dist_centers = dist(c_start, c_end)
    cond = dist_centers >= 2.0 * radius
    
    # Even if invalid, calculate a value (to prevent NaN), the valid flag will handle it
    dist_safe = jnp.maximum(dist_centers, 2.0 * radius)
    
    median_point = (c_end - c_start) / 2.0
    psia = jnp.arctan2(median_point[1], median_point[0])
    
    # [-1, 1]
    arg = jnp.clip(radius / (dist_safe / 2.0), -1.0, 1.0)
    alpha = jnp.arccos(arg)
    
    beta_0 = mod2pi(psia - alpha - start[2] + jnp.pi / 2.0)
    beta_2 = mod2pi(psia - alpha - end[2] + jnp.pi / 2.0)
    straight_dist = jnp.sqrt(jnp.maximum(dist_safe**2.0 - 4 * radius**2.0, 0.0))
    
    total_len = radius * (beta_0 + beta_2) + straight_dist
    
    # if invalid, set cost to INF
    total_len = jnp.where(cond, total_len, INF_COST)
    
    return PathCandidate(total_len, beta_0, straight_dist, beta_2, TYPE_LSR, cond)

@jit
def solve_RSL( start:jnp.array, end:jnp.array, radius: float) -> PathCandidate:
    c_start = find_center(start, -1.0, radius) # R
    c_end = find_center(end, 1.0, radius)      # L
    
    dist_centers = dist(c_start, c_end)
    cond = dist_centers >= 2.0 * radius
    dist_safe = jnp.maximum(dist_centers, 2.0 * radius)
    
    median_point = (c_end - c_start) / 2.0
    psia = jnp.arctan2(median_point[1], median_point[0])
    
    arg = jnp.clip(radius / (dist_safe / 2.0), -1.0, 1.0)
    alpha = jnp.arccos(arg)
    
    beta_0 = mod2pi(start[2] - (psia + alpha - jnp.pi / 2.0))
    beta_2 = mod2pi(end[2] - (psia + alpha - jnp.pi / 2.0))
    straight_dist = jnp.sqrt(jnp.maximum(dist_safe**2.0 - 4.0 * radius**2.0, 0.0))
    
    total_len = radius * (beta_0 + beta_2) + straight_dist
    total_len = jnp.where(cond, total_len, INF_COST)
    
    return PathCandidate(total_len, beta_0, straight_dist, beta_2, TYPE_RSL, cond)

@jit
def solve_RLR(start: jnp.array, end: jnp.array, radius: float) -> PathCandidate:
    c_start = find_center(start, -1.0, radius) # R
    c_end = find_center(end, -1.0, radius)     # R
    
    dist_centers = dist(c_start, c_end)
    cond = dist_centers < 4 * radius
    
    dist_safe = jnp.clip(dist_centers, 0.0, 4.0 * radius)
    
    median_point = (c_end - c_start) / 2.0
    psia = jnp.arctan2(median_point[1], median_point[0])
    
    arg = jnp.clip(dist_safe / (4 * radius), -1.0, 1.0)
    gamma = 2.0 * jnp.arcsin(arg)
    
    beta_0 = mod2pi(start[2] - psia + 3*jnp.pi / 2.0 - (jnp.pi - gamma) / 2.0)
    beta_2 = mod2pi(-end[2] + psia + 3*jnp.pi / 2.0 - (jnp.pi - gamma) / 2.0)
    beta_1 = mod2pi(2 * jnp.pi - gamma)
    
    total_len = radius * (beta_0 + beta_1 + beta_2)
    total_len = jnp.where(cond, total_len, INF_COST)
    
    return PathCandidate(total_len, beta_0, beta_1, beta_2, TYPE_RLR, cond)


#have a little bugs(solved)
@jit
def solve_LRL(start: jnp.array, end: jnp.array, radius: float) -> PathCandidate:
    c_start = find_center(start, 1.0, radius) # L
    c_end = find_center(end, 1.0, radius)     # L
    
    dist_centers = dist(c_start, c_end)
    cond = dist_centers < 4 * radius
    dist_safe = jnp.clip(dist_centers, 0.0, 4.0 * radius)
    
    median_point = (c_end - c_start)/2.0
    psia = jnp.arctan2(median_point[1], median_point[0])
    
    arg = jnp.clip(dist_safe / (4 * radius), -1.0, 1.0)
    gamma = 2.0 * jnp.arcsin(arg)
    
    beta_0 = mod2pi(-start[2] + psia + jnp.pi / 2.0 + (jnp.pi - gamma) / 2.0)
    beta_2 = mod2pi(end[2] - psia + jnp.pi / 2.0 + (jnp.pi - gamma) / 2.0)
    beta_1 = mod2pi(2 * jnp.pi - gamma)
    
    total_len = radius * (beta_0 + beta_1 + beta_2)
    total_len = jnp.where(cond, total_len, INF_COST)
    
    return PathCandidate(total_len, beta_0, beta_1, beta_2, TYPE_LRL, cond)


# generate point along path
def _generate_straight_type(x:float, start:jnp.array, end:jnp.array, params:DubinsParams, path_params:PathCandidate, signs:tuple) -> jnp.array:
    """Generate points for CSC types (LSL, RSR, LSR, RSL)"""
    # Unpack parameters
    radius = params.radius
    total, b0, straight_len, b2, p_type, valid = path_params
    sign_1, sign_2 = signs # only the first and last turns have signs, the middle is straight
    
    # Calculate segment lengths
    len_1 = b0 * radius
    len_3 = b2 * radius
    # len_2 = straight_len (middle straight segment)
    
    # Determine centers
    c_start = find_center(start, sign_1, radius)
    c_end = find_center(end, sign_2, radius)
    #jax.debug.print("c_start: {}", c_start)
    #jax.debug.print("c_end: {}", c_end)
    # T1: the first circular arc
    p1 = circle_arc(start, b0 * sign_1, c_start, x, radius)
    
    # T3: the third circular arc (calculated backward from the end)
    # Distance offset relative to the end: x - (total_len - len_3)
    # Here we simplify the logic: when x is in the third segment, we use circle_arc to calculate
    # However, circle_arc is based on the starting angle.
    # For JIT efficiency, we use the original logic: directly use the end point to backtrack
    p3 = circle_arc(end, b2 * sign_2, c_end, x - total, radius)

    # Straight: middle straight segment
    # We need the exact coordinates of the start of the straight segment
    # T1 end angle
    angle_t1_end = start[2] + ((b0-jnp.pi/2) * sign_1)
    t1_end_pos = c_start + radius * jnp.array([jnp.cos(angle_t1_end), jnp.sin(angle_t1_end)])
    
    # Direction vector of the straight segment
    # T3 start point (end of the straight segment)
    angle_t3_start = end[2] + ((-b2-jnp.pi/2) * sign_2)
    t3_start_pos = c_end + radius * jnp.array([jnp.cos(angle_t3_start), jnp.sin(angle_t3_start)])
    #jax.debug.print("theta1:{}",b0)
    #jax.debug.print("theta2:{}",b2)
    #jax.debug.print("angle_t1_end: {}", angle_t1_end)
    #jax.debug.print("c_1_end: {}", t1_end_pos)
    #jax.debug.print("angle_t3_start: {}", angle_t3_start)
    #jax.debug.print("c_3_start: {}", t3_start_pos)
    ratio = (x - len_1) / straight_len
    # avoid division by zero
    ratio = jnp.where(straight_len > 1e-6, ratio, 0.0)
    p2 = (1.0 - ratio) * t1_end_pos + ratio * t3_start_pos
    
    # Selection logic
    cond1 = x < len_1
    cond3 = x > (total - len_3)
    
    pos = jnp.where(cond1, p1, p2)
    pos = jnp.where(cond3, p3, pos)
    return pos


#@attention: mathmatical expression is complex here
def _generate_curve_type(x:float, start:jnp.array, end:jnp.array, params:DubinsParams, path_params:PathCandidate, signs:tuple) -> jnp.array:
    """Generate points for CCC types (RLR, LRL)"""
    radius = params.radius
    total, b0, b1, b2, p_type, valid = path_params
    sign_1, sign_2, sign_3 = signs
    
    len_1 = b0 * radius
    len_3 = b2 * radius
    
    c_start = find_center(start, sign_1, radius)
    c_end = find_center(end, sign_3, radius)
    
    # T1
    p1 = circle_arc(start, b0 * sign_1, c_start, x, radius)
    
    # T3
    p3 = circle_arc(end, b2 * sign_3, c_end, x - total, radius)
    
    # T2 (Middle Turn)
    dist_centers = dist(c_start, c_end)
    mid_point = (c_start + c_end) / 2.0
    
    # Geometric calculation of c_mid
    # calculate h
    h = jnp.sqrt(jnp.maximum(4.0 * radius**2 - (dist_centers / 2.0)**2, 0.0))
    ortho_vec = ortho((c_end - c_start) / dist_centers)
    # The sign depends on the turn type. If it is LRL (sign 1, -1, 1), the middle is R (-1)
    # If it is RLR (sign -1, 1, -1), the middle is L (1)
    c_mid = mid_point + sign_1 * ortho_vec * h
    

    # Fix: consistent angle calculation with _generate_straight_type
    ###### the junction point needs to be calculated carefully ######
    # T1 end
    angle_t1_end = start[2] + ((b0 - jnp.pi/2) * sign_1)
    t1_end_pos = c_start + radius * jnp.array([jnp.cos(angle_t1_end), jnp.sin(angle_t1_end)])
    
    dist_in_t2 = x - len_1

    angle_start_t2 = jnp.arctan2(t1_end_pos[1] - c_mid[1], t1_end_pos[0] - c_mid[0])
    
    final_angle_t2 = angle_start_t2 + (dist_in_t2 / radius) * sign_2
    p2 = c_mid + radius * jnp.array([jnp.cos(final_angle_t2), jnp.sin(final_angle_t2)])

    cond1 = x < len_1
    cond3 = x > (total - len_3)
    
    pos = jnp.where(cond1, p1, p2)
    pos = jnp.where(cond3, p3, pos)
    return pos

def get_point_at_dist(dist_val:float, start:jnp.array, end:jnp.array, params:DubinsParams, best_path:PathCandidate) -> jnp.array:
    """
    Calculate point coordinates based on distance, using switch to select geometric logic according to path type
    """
    ptype = best_path.path_type
    
    # Define handler functions for each type
    
    # CSC Types
    # LSL: (+1, +1)
    def do_lsl(x, s, e, p, pp): return _generate_straight_type(x, s, e, p, pp, (1.0, 1.0))
    # RSR: (-1, -1)
    def do_rsr(x, s, e, p, pp): return _generate_straight_type(x, s, e, p, pp, (-1.0, -1.0))
    # RSL: (-1, +1)
    def do_rsl(x, s, e, p, pp): return _generate_straight_type(x, s, e, p, pp, (-1.0, 1.0))
    # LSR: (+1, -1)
    def do_lsr(x, s, e, p, pp): return _generate_straight_type(x, s, e, p, pp, (1.0, -1.0))
    
    # CCC Types
    # RLR: (-1, +1, -1)
    def do_rlr(x, s, e, p, pp): return _generate_curve_type(x, s, e, p, pp, (-1.0, 1.0, -1.0))
    # LRL: (+1, -1, +1)
    def do_lrl(x, s, e, p, pp): return _generate_curve_type(x, s, e, p, pp, (1.0, -1.0, 1.0))
    
    # lax.switch 
    # works with JIT
    point = lax.switch(
        ptype,
        [do_lsl, do_rsr, do_rsl, do_lsr, do_rlr, do_lrl],
        dist_val, start, end, params, best_path
    )
    
    # If distance exceeds total length (padding), return end point coordinates
    point = jnp.where(dist_val > best_path.cost, end[:2], point)
    
    return point


# main (Entry Point)

@partial(jit, static_argnames=['params'])
def dubins_path_planning(start:jnp.array, end:jnp.array, params: DubinsParams):
    """Calculate Dubins path from start to end."""
    
    # calculate all 6 path candidates
    cands = [
        solve_LSL(start, end, params.radius),
        solve_RSR(start, end, params.radius),
        solve_RSL(start, end, params.radius),
        solve_LSR(start, end, params.radius),
        solve_RLR(start, end, params.radius),
        solve_LRL(start, end, params.radius)
    ]
    
    costs = jnp.stack([c.cost for c in cands])
    
    best_idx = jnp.argmin(costs)
    
    # 4. Extract best path parameters
    # Use tree_map and getitem to extract a single NamedTuple from a stack of NamedTuples
    # First convert list of structs to struct of arrays
    cands_soa = jax.tree.map(lambda *args: jnp.stack(args), *cands)
    # Then extract
    best_path = jax.tree.map(lambda arr: arr[best_idx], cands_soa)
    
    # 5. Generate sampling distance array
    
    dist_samples = jnp.linspace(0.0, best_path.cost, params.num_probes)
    
    # plan B
    #raw_dists = jnp.arange(0, MAX_PATH_POINTS) * params.point_separation
    #dist_samples = jnp.where(raw_dists <= best_path.cost, raw_dists, best_path.cost + 1.0)
    
    # VMAP
    generate_fn = partial(get_point_at_dist, start=start, end=end, params=params, best_path=best_path)
    path_points = vmap(generate_fn)(dist_samples)
    return path_points, best_path
