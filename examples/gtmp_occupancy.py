import jax
from jax import jit, vmap, random, lax
import hydra
import omegaconf
import jax.numpy as jnp
from jax import tree_util
import time
import matplotlib.pyplot as plt

from chrono import Timer
from gtmp.files import get_configs_path, get_data_path
from gtmp.planners import GTMPState, gtmp_plan, gtmp_akima_plan,gtmp_dubins_plan,GTMPOutput
from gtmp.dubins_splines import DubinsParams,dubins_path_planning,PathCandidate
from gtmp.objectives.occupancy_map import OccupancyMap
from gtmp.metrics import compute_metrics

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def print_performance_report(
    total_timer_start,
    jit_compilation_dubins_time,
    planning_time,
    cost_compute_time,
    jit_compile_time,
    mdp_solve_time,
    extraction_time,
    concatenation_time,
    num_plans,
    num_tasks_per_plan,
    num_total_tasks
):
    """Print detailed performance report and time breakdown."""
    def calculate_percentage(part_time, total_time):
        return part_time / total_time * 100
    
    def print_time_breakdown(time_data, total_time):

        print("\n")
        print(f"Complete Time Breakdown ({num_plans} plans in parallel):")
        print(f"{'='*60}")
        
        for idx, name, t, description in time_data:
            pct = calculate_percentage(t, total_time)
            print(f"{idx}. {name:25s} {t:.6f}s ({pct:.2f}%)")
            print(f"   └─ {description}")
        
        print(f"Total time:               {total_time:.6f}s (100.00%)")
    
    total_time = time.time() - total_timer_start
    
    # It is not accurate to sum up all recorded times due to overlaps and JIT effects
    reshaping_time = total_time - (
        jit_compilation_dubins_time + planning_time + cost_compute_time + 
        jit_compile_time + mdp_solve_time + extraction_time + concatenation_time
    )
    
    time_data = [
        (1, "JIT compilation (Dubins)", jit_compilation_dubins_time, "Dubins planner first run"),
        (2, "Planning execution", planning_time, f"Dubins path planning for {num_total_tasks} tasks"),
        (3, "Data reshaping", reshaping_time, "Reshape & transpose results"),
        (4, "Cost computation", cost_compute_time, "Path length + collision costs"),
        (5, "JIT compilation (MDP)", jit_compile_time, "MDP solver first run"),
        (6, "MDP solve (JIT)", mdp_solve_time, "Value iteration for all plans"),
        (7, "Path extraction", extraction_time, "Extract optimal angle sequences"),
        (8, "Path concatenation", concatenation_time, "Join segments & remove duplicates"),
    ]
    
    # print breakdown
    print_time_breakdown(time_data, total_time)
    # print summary
    print(f"\n{'='*60}")
    print(f"Performance Summary:")
    print(f"{'='*60}")
    print(f"• Plans processed:        {num_plans} (in parallel)")
    print(f"• Total Dubins paths:     {num_total_tasks}")
    



@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_dubins_occupancy")
def main(cfg: omegaconf.DictConfig):
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
# 1. load configs and data
    # Environment
    occ = 1. - jnp.load((get_data_path() / 'real_map' / str(cfg.environment.map_file)).as_posix())
    limits = jnp.array(cfg.environment.limits)
    # obtain sequential points
    Q_points = jnp.array(cfg.environment.sequential_points) 
    num_segments = Q_points.shape[0] - 1

    # generate angles for Dubins planner
    num_angles = 6
    angles = jnp.linspace(0, 2*jnp.pi, num_angles, endpoint=False)

    q_segment_starts = Q_points[:-1] 
    g_segment_goals = Q_points[1:]   
    
    # create all angle combinations for each segment (start 6 angles × end 6 angles = 36 combinations)
    def create_angle_combinations(xy_starts, xy_goals):

        num_segs = xy_starts.shape[0]
        num_angle_combos = num_angles * num_angles  # 36

        start_angles_grid, goal_angles_grid = jnp.meshgrid(angles, angles, indexing='ij')
        # flatten (36,)
        start_angles_flat = start_angles_grid.ravel()  # [0,0,0,0,0,0, π/3,π/3,π/3,π/3,π/3,π/3, ...]
        goal_angles_flat = goal_angles_grid.ravel()    # [0,π/3,2π/3,π,4π/3,5π/3, 0,π/3,2π/3,π,4π/3,5π/3, ...]
        
        # expand for each segment
        xy_starts_expanded = jnp.tile(xy_starts[:, None, :], (1, num_angle_combos, 1))
        xy_goals_expanded = jnp.tile(xy_goals[:, None, :], (1, num_angle_combos, 1))
        
        # add angle dimension
        # (36,) -> (1, 36, 1) -> (num_segments, 36, 1)
        start_angles_expanded = jnp.broadcast_to(
            start_angles_flat[None, :, None], 
            (num_segs, num_angle_combos, 1)
        )
        goal_angles_expanded = jnp.broadcast_to(
            goal_angles_flat[None, :, None],
            (num_segs, num_angle_combos, 1)
        )
        
        # (num_segments, 36, 3)
        q_combinations = jnp.concatenate([xy_starts_expanded, start_angles_expanded], axis=-1)
        g_combinations = jnp.concatenate([xy_goals_expanded, goal_angles_expanded], axis=-1)
        
        return q_combinations, g_combinations
    
    if cfg.planner.name == 'dubins':
        q_with_angles, g_with_angles = create_angle_combinations(q_segment_starts, g_segment_goals)
        # debug: print(f"Created angle combinations for Dubins planner: {q_with_angles}")
        # flatten (num_segments * 36, 3)
        q_flat_initial = q_with_angles.reshape(-1, 3)
        # debug: print(f"Flattened q_with_angles: {q_flat_initial}")
        g_flat_initial = g_with_angles.reshape(-1, 3)

        # If there are multiples of cfg.num_plans, further replicate
        q_flat = jnp.repeat(q_flat_initial, cfg.num_plans, axis=0)
        # debug: print("q_flat:", q_flat)
        goals_flat = jnp.repeat(g_flat_initial[:, None, :], cfg.num_plans, axis=0)
    else:
        # Non-Dubins planner, keep original logic
        q_flat = jnp.repeat(q_segment_starts, cfg.num_plans, axis=0)
        goals_flat = jnp.repeat(g_segment_goals[:, None, :], cfg.num_plans, axis=0)

    occ_map = OccupancyMap.from_prob(occ, limits=limits[:2], threshold=0.1, infinite_cost=True)

    # planner
    # create batched planner state using flattened inputs
    planner_state_flat = GTMPState.create(
        q=q_flat,
        goals=goals_flat,
        bounds=limits,
        transition_field=occ_map,
        occ_map=occ_map,
        get_velocity=False,
        **cfg.planner.params
    )
# -----------------------------------------------------------------------------------------
# 2. choosing different execution modes
    # 'batch', 'scan', 'vmap'
    execution_mode = getattr(cfg, 'execution_mode', 'batch')
    batch_size = getattr(cfg, 'batch_size', 10)  # only used in batch mode
    
    print(f"\n{'='*60}")
    print(f"Execution Mode: {execution_mode.upper()}")
    if execution_mode == 'batch':
        print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    # The akima and straight planners are not compatible now (need to modify the structure of planner_state)
    if cfg.planner.name == 'straight':
        plan_fn = gtmp_plan
    elif cfg.planner.name == 'akima':
        plan_fn = gtmp_akima_plan
    elif cfg.planner.name == 'dubins':
        plan_fn = gtmp_dubins_plan
    else:
        raise ValueError(f"Unknown planner: {cfg.planner.name}")

    # calculate total tasks and batch info
    num_angle_combinations = num_angles * num_angles  # 36 = 6 × 6
    num_tasks_per_plan = num_segments * num_angle_combinations
    num_total_tasks = num_tasks_per_plan * cfg.num_plans
    
    # generate random seeds
    rng_key, *keys = jax.random.split(rng_key, num_total_tasks + 1)
    keys_flat = jnp.stack(keys)[:num_total_tasks]  # Shape (num_total_tasks, 2)
    
    # reshape keys and planner_state to (num_plans, num_tasks_per_plan, ...)
    keys_per_plan = keys_flat.reshape(cfg.num_plans, num_tasks_per_plan, 2)
    planner_state_per_plan = jax.tree.map(
        lambda x: x.reshape(cfg.num_plans, num_tasks_per_plan, *x.shape[1:]) if isinstance(x, jnp.ndarray) else x,
        planner_state_flat
    )
# -----------------------------------------------------------------------------------------
# 3. planning with different execution modes
    
    # total timer start
    total_timer_start = time.time()
    
    # initialize JIT compilation time
    Jit_compilation_dubins_time = 0
    
    if execution_mode == 'vmap':
        # ============ Mode 1: Full parallelization (vmap) ============
        print("Using VMAP mode: Full parallelization (highest memory usage)")
        
        # create fully vmap-ed planning function
        if cfg.planner.name == 'dubins':
            dubins_params = DubinsParams(radius=cfg.planner.min_radius, num_probes=cfg.planner.params.num_probes)
            plan_single_fn = jit(vmap(lambda k, s: plan_fn(k, s, dubins_params), in_axes=(0, 0)))
            plan_all_fn = jit(vmap(plan_single_fn, in_axes=(0, 0)))
        else:
            plan_single_fn = jit(vmap(plan_fn, in_axes=(0, 0)))
            plan_all_fn = jit(vmap(plan_single_fn, in_axes=(0, 0)))
        
        # JIT pre-compilation
        print("Compiling vmap function...")
        start = time.time()
        _ = plan_all_fn(keys_per_plan, planner_state_per_plan).path.block_until_ready()
        Jit_compilation_dubins_time = time.time() - start
        print(f"JIT compilation time: {Jit_compilation_dubins_time:.8f} seconds")
        # Actual planning
        # with Timer() as timer:
            #paths_all = plan_all_fn(keys_per_plan, planner_state_per_plan)
            # paths_all = plan_all_fn(keys_per_plan, planner_state_per_plan)
            # paths_all.path.block_until_ready()
        
        # planning_time = timer.elapsed
        # print(f"\nPlanning execution time: {planning_time:.8f} seconds")

        start1= time.time()    
        paths_all = plan_all_fn(keys_per_plan, planner_state_per_plan)
        paths_all.path.block_until_ready()
        end1= time.time()
        planning_time = end1 - start1
        print(f"\nPlanning execution time: {planning_time:.8f}s")

    elif execution_mode == 'scan':
        print("Using LAX.SCAN mode: Sequential processing (lowest memory usage)")
        
        # create vmap function for a single plan
        if cfg.planner.name == 'dubins':
            dubins_params = DubinsParams(radius=cfg.planner.min_radius, num_probes=cfg.planner.params.num_probes)
            plan_single_fn = jit(vmap(lambda k, s: plan_fn(k, s, dubins_params), in_axes=(0, 0)))
        else:
            plan_single_fn = jit(vmap(plan_fn, in_axes=(0, 0)))
        
        def scan_fn(carry, x):
            keys_plan, state_plan = x
            result = plan_single_fn(keys_plan, state_plan)
            return carry, result
        
        print("Compiling lax.scan function...")
        start = time.time()
        _, result_warmup = lax.scan(scan_fn, None, (keys_per_plan, planner_state_per_plan))
        result_warmup.path.block_until_ready()
        Jit_compilation_dubins_time = time.time() - start
        print(f"JIT compilation time: {Jit_compilation_dubins_time:.8f} seconds")

        start1 = time.time()
        _, paths_all = lax.scan(scan_fn, None, (keys_per_plan, planner_state_per_plan))
        paths_all.path.block_until_ready()
        end1 = time.time()
        planning_time = end1 - start1
        print(f"\nPlanning execution time: {planning_time:.8f}s")
        
    elif execution_mode == 'batch':

        print(f"Using BATCH mode: Processing in batches of {batch_size}")
        
        # calculate number of batches (ceiling division to handle non-divisible cases)
        num_batches = (cfg.num_plans + batch_size - 1) // batch_size
        
        # batch: split data by batch_size (use padding to maintain consistent shape)
        keys_batches = []
        state_batches = []
        actual_batch_sizes = []  # record actual size of each batch (for later trimming)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, cfg.num_plans)
            actual_size = end_idx - start_idx
            actual_batch_sizes.append(actual_size)
            
            keys_batch = keys_per_plan[start_idx:end_idx]
            state_batch = jax.tree.map(
                lambda x: x[start_idx:end_idx] if isinstance(x, jnp.ndarray) else x,
                planner_state_per_plan
            )
            
            # If the last batch is smaller than batch_size, pad it
            if actual_size < batch_size:
                pad_size = batch_size - actual_size
                # Padding keys: replicate the last element
                keys_batch = jnp.concatenate([
                    keys_batch,
                    jnp.repeat(keys_batch[-1:], pad_size, axis=0)
                ], axis=0)
                # Padding state: replicate the last element for each ndarray
                state_batch = jax.tree.map(
                    lambda x: jnp.concatenate([
                        x,
                        jnp.repeat(x[-1:], pad_size, axis=0)
                    ], axis=0) if isinstance(x, jnp.ndarray) else x,
                    state_batch
                )
            
            keys_batches.append(keys_batch)
            state_batches.append(state_batch)
        
        # create batch processing function (vmap over batch_size, then vmap over num_tasks_per_plan)
        if cfg.planner.name == 'dubins':
            dubins_params = DubinsParams(radius=cfg.planner.min_radius, num_probes=cfg.planner.params.num_probes)
            # for a single plan's all tasks implement vmap
            plan_single_fn = jit(vmap(lambda k, s: plan_fn(k, s, dubins_params), in_axes=(0, 0)))
            # for a batch of plans implement vmap
            plan_batch_fn = jit(vmap(plan_single_fn, in_axes=(0, 0)))
        else:
            plan_single_fn = jit(vmap(plan_fn, in_axes=(0, 0)))
            plan_batch_fn = jit(vmap(plan_single_fn, in_axes=(0, 0)))
        
        print(f"Total plans: {cfg.num_plans}, divided into {num_batches} batches")
        print("Compiling batch processing function...")
        start = time.time()
        warmup_result = plan_batch_fn(keys_batches[0], state_batches[0])
        warmup_result.path.block_until_ready()
        Jit_compilation_dubins_time = time.time() - start
        print(f"JIT compilation time: {Jit_compilation_dubins_time:.8f} seconds")
        
        print("\nStarting batch processing...")
        all_results = []
        
        batch_timer_start = time.time()
        for batch_idx in range(num_batches):
            batch_start = time.time()
            
            batch_result = plan_batch_fn(keys_batches[batch_idx], state_batches[batch_idx])
            batch_result.path.block_until_ready()
            # Trim padding if necessary    
            actual_size = actual_batch_sizes[batch_idx]
            if actual_size < batch_size:
                batch_result = jax.tree.map(
                    lambda x: x[:actual_size],
                    batch_result
                )
                
            all_results.append(batch_result)
                
            print(f"Batch {batch_idx + 1}/{num_batches} completed "
                      f"({actual_size} plans, {time.time() - batch_start:.8f}s)")
        
        planning_time = time.time() - batch_timer_start
        print(f"\nPlanning execution time: {planning_time:.8f}s")
        print(f"Average time per plan: {planning_time / cfg.num_plans:.8f}s")
        
        # concatenate all batch results
        paths_all = jax.tree.map(
            lambda *args: jnp.concatenate(args, axis=0),
            *all_results
        )
    
    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode}. Must be 'vmap', 'scan', or 'batch'")
    
    # === unify results ===
    # flatten results back to original shape (num_total_tasks, ...)
    paths_flat = jax.tree.map(
        lambda x: x.reshape(num_total_tasks, *x.shape[2:]),
        paths_all
    )
        
    num_segments = Q_points.shape[0] - 1

    # Reshape: (num_segments * 36 * num_plans, ...) -> (num_segments, 36, num_plans, ...)
    output_unflattened = tree_util.tree_map(
        lambda x: x.reshape(num_segments, num_angle_combinations, cfg.num_plans, *x.shape[1:]),
        paths_flat
    )

    # transpose to make num_plans the first dimension: (num_plans, num_segments, 36, ...)
    output_nested = tree_util.tree_map(
        lambda x: jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))),
        output_unflattened
    )
    
    # === solve optimal angle sequences for each plan (with angle smoothness constraint) ===
    # batch Value Iteration
    
    print("\n" + "="*60)
    print("Solving optimal angle sequences using MDP (Value Iteration)")
    print("With angle smoothness constraint")
    print("="*60)
    
    # 1. pre-compute transition cost matrix (36, 36)
    start_angles_grid, goal_angles_grid = jnp.meshgrid(jnp.arange(num_angles), jnp.arange(num_angles), indexing='ij')
    start_angle_indices = start_angles_grid.ravel()  # (36,)
    goal_angle_indices = goal_angles_grid.ravel()    # (36,)
    
    curr_goal_angles = goal_angle_indices[:, None]  # (36, 1)
    next_start_angles = start_angle_indices[None, :]  # (1, 36)
    angle_mismatch_penalty = 1e5
    
    # transition cost matrix (36, 36): [j, k] = cost of transitioning from angle combination j to k
    transition_cost = jnp.where(
        curr_goal_angles == next_start_angles,
        0.0,
        angle_mismatch_penalty
    )
    
    # 2. calculate plans segment costs
    print(f"Computing segment costs for all {cfg.num_plans} plans in parallel...")
    cost_compute_start = time.time()
    
    collision_penalty = 1e6
    
    # collision_costs (num_plans, num_segments, 36) 
    collision_costs = jnp.where(output_nested.collision, collision_penalty, 0.0)
    
    # path length costs - optimized as pure tensor operations, avoiding nested vmap
    # output_nested.path: (num_plans, num_segments, 36, num_points, 2)
    path_diffs = jnp.diff(output_nested.path, axis=-2)  # (num_plans, num_segments, 36, num_points-1, 2)
    segment_lengths = jnp.linalg.norm(path_diffs, axis=-1)  # (num_plans, num_segments, 36, num_points-1)
    path_lengths = jnp.sum(segment_lengths, axis=-1)  # (num_plans, num_segments, 36)
    
    # (num_plans, num_segments, 36)
    segment_costs = collision_costs + path_lengths
    #segment_costs.block_until_ready()  # ensure computation is complete
    
    cost_compute_time = time.time() - cost_compute_start
    print(f"Segment costs computed in {cost_compute_time:.8f} seconds (all plans in parallel)")
    
    # 3. define and JIT compile batch Value Iteration function
    @jit
    def batch_value_iteration(segment_costs_batch, transition_cost_matrix):
        """Batch solve Value Iteration for all plans"""

        num_plans_batch, num_segs, num_angles = segment_costs_batch.shape
        
        V = jnp.zeros((num_plans_batch, num_segs, num_angles))
        policy = jnp.zeros((num_plans_batch, num_segs, num_angles), dtype=jnp.int32)
        
        V = V.at[:, -1, :].set(segment_costs_batch[:, -1, :])
        
        # Backward dynamic programming (using lax.fori_loop)
        def backward_step(i_rev, carry):
            V, policy = carry
            i = num_segs - 2 - i_rev  # actual segment index (reverse order)
            
            # transition_cost_matrix: (36, 36)
            # V[:, i+1, :]: (num_plans, 36)
            # Calculate V[:, i+1, :][..., None, :] + transition_cost_matrix[None, ...]
            #   -> (num_plans, 36_from, 36_to)
            
            costs_matrix = (
                V[:, i + 1, :][..., None, :] +  # (num_plans, 1, 36_to)
                transition_cost_matrix[None, ...]  # (1, 36_from, 36_to)
            )  # (num_plans, 36_from, 36_to)
            
            best_next = jnp.argmin(costs_matrix, axis=2)  # (num_plans, 36_from)
            min_future_cost = jnp.min(costs_matrix, axis=2)  # (num_plans, 36_from)
            
            # Update V and policy
            V = V.at[:, i, :].set(segment_costs_batch[:, i, :] + min_future_cost)
            policy = policy.at[:, i, :].set(best_next)
            
            return (V, policy)
        
        # if only one segment, skip the loop
        V, policy = lax.cond(
            num_segs > 1,
            lambda _: lax.fori_loop(0, num_segs - 1, backward_step, (V, policy)),
            lambda _: (V, policy),
            None
        )
        
        return V, policy
    
    # 4. define and JIT compile batch forward trace function
    @jit
    def batch_forward_trace(V, policy):
        """Batch forward trace the optimal angle sequences for all plans"""

        num_plans_batch, num_segs, _ = V.shape
        
        first_angles = jnp.argmin(V[:, 0, :], axis=1)  # (num_plans,)
        
        best_indices = jnp.zeros((num_plans_batch, num_segs), dtype=jnp.int32)
        best_indices = best_indices.at[:, 0].set(first_angles)
        
        # Forward trace (using lax.fori_loop)
        def forward_step(i, indices):
            prev_angles = indices[:, i - 1]  # (num_plans,)
            
            # vmap over plans to get next angles
            def get_next_angle(plan_idx):
                prev_angle = prev_angles[plan_idx]
                return policy[plan_idx, i - 1, prev_angle]
            
            next_angles = vmap(get_next_angle)(jnp.arange(num_plans_batch))
            indices = indices.at[:, i].set(next_angles)
            return indices
        
        best_indices = lax.cond(
            num_segs > 1,
            lambda _: lax.fori_loop(1, num_segs, forward_step, best_indices),
            lambda _: best_indices,
            None
        )
        
        def get_total_cost(plan_idx):
            first_angle = best_indices[plan_idx, 0]
            return V[plan_idx, 0, first_angle]
        
        total_costs = vmap(get_total_cost)(jnp.arange(num_plans_batch))
        
        return best_indices, total_costs
    
    # 5. JIT compile MDP solver with full data (all plans)
    print(f"JIT compiling MDP solver for {cfg.num_plans} plans...")
    jit_compile_start = time.time()
    
    # the first run to trigger JIT compilation
    V, policy = batch_value_iteration(segment_costs, transition_cost)
    all_best_indices, all_costs = batch_forward_trace(V, policy)
    all_costs.block_until_ready() # ensure computation is complete
    
    jit_compile_time = time.time() - jit_compile_start
    print(f"JIT compilation + first run: {jit_compile_time:.8f} seconds")
    
    # 6. run MDP optimization for all plans
    print(f"Running MDP optimization (compiled) for {cfg.num_plans} plans in parallel...")
    mdp_solve_start = time.time()
    
    V, policy = batch_value_iteration(segment_costs, transition_cost)
    all_best_indices, all_costs = batch_forward_trace(V, policy)
    all_costs.block_until_ready()
    
    mdp_solve_time = time.time() - mdp_solve_start
    
    print(f"\n{'='*60}")
    print(f"MDP Optimization Results ({cfg.num_plans} plans in parallel):")
    print(f"{'='*60}")
    print(f"JIT compilation + warmup:  {jit_compile_time:.8f} seconds")
    print(f"MDP solve time (JIT):      {mdp_solve_time:.8f} seconds")
    print(f"Average total cost:        {jnp.mean(all_costs):.2f}")
    print(f"Cost range:                [{jnp.min(all_costs):.2f}, {jnp.max(all_costs):.2f}]")
    print(f"{'='*60}\n")
    
    # 4. Extract optimal paths based on best angle sequences
    print(f"Extracting optimal paths for {cfg.num_plans} plans...")
    extraction_start = time.time()
    
    def extract_optimal_sequence(plan_idx):
        best_idx = all_best_indices[plan_idx]  # (num_segments,)
        
        def extract_segment(seg_idx):
            angle_idx = best_idx[seg_idx]
            seg_output = tree_util.tree_map(
                lambda x: x[plan_idx, seg_idx, angle_idx],
                output_nested
            )
            return seg_output
        
        segments_output = tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0),
            *[extract_segment(i) for i in range(num_segments)]
        )
        return segments_output
    
    # select optimal sequences for all plans
    output_selected = tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *[extract_optimal_sequence(i) for i in range(cfg.num_plans)]
    )
    
    extraction_time = time.time() - extraction_start
    print(f"Path extraction completed in {extraction_time:.6f} seconds")
    
    print(f"Concatenating path segments...")
    concatenation_start = time.time()
    
    # output_selected.collision shape: (num_plans, num_segments)
    final_collision = jnp.any(output_selected.collision, axis=1)  # (num_plans,)
    
    # output_selected.path shape: (num_plans, num_segments, num_points, dim)
    paths_segments = output_selected.path
    
    segment_list = [paths_segments[:, 0, ...]]  
    for i in range(1, num_segments):
        # skip the first point of each subsequent segment to avoid duplication
        segment_list.append(paths_segments[:, i, 1:, ...])
    
    full_path = jnp.concatenate(segment_list, axis=1)  # (num_plans, total_points, dim)
    full_path.block_until_ready() 
    
    concatenation_time = time.time() - concatenation_start
    print(f"Path concatenation completed in {concatenation_time:.6f} seconds")
    
    # output final paths
    final_paths = GTMPOutput(
        path=full_path,
        collision=final_collision,
    )
    
    # 5. Performance report
    print_performance_report(
        total_timer_start=total_timer_start,
        jit_compilation_dubins_time=Jit_compilation_dubins_time,
        planning_time=planning_time,
        cost_compute_time=cost_compute_time,
        jit_compile_time=jit_compile_time,
        mdp_solve_time=mdp_solve_time,
        extraction_time=extraction_time,
        concatenation_time=concatenation_time,
        num_plans=cfg.num_plans,
        num_tasks_per_plan=num_tasks_per_plan,
        num_total_tasks=num_total_tasks
    )
    
    # 6. Visualization
    visualize_paths(
        final_paths=final_paths,
        Q_points=Q_points,
        occ_map=occ_map,
        limits=limits
    )



def visualize_paths(final_paths, Q_points, occ_map, limits):

    fig, ax = plt.subplots(figsize=(10, 10))
    
    X = jnp.linspace(*limits[0], occ_map.map.shape[0])
    Y = jnp.linspace(*limits[1], occ_map.map.shape[1])
    X, Y = jnp.meshgrid(X, Y)
    ax.contourf(X, Y, occ_map.map.T, cmap='Greys', alpha=1)
    
    # divide paths into collision-free and collision paths
    free_paths = final_paths.path[jnp.invert(final_paths.collision)]
    coll_paths = final_paths.path[final_paths.collision]

    for i in range(free_paths.shape[0]):
        ax.plot(free_paths[i, :, 0], free_paths[i, :, 1], 
                'b-', linewidth=1.5, alpha=1)
    
    # collision paths
    for i in range(coll_paths.shape[0]):
        ax.plot(coll_paths[i, :, 0], coll_paths[i, :, 1], 
                'r--', linewidth=1, alpha=0.8)
    
    start_point = Q_points[0]
    ax.plot(start_point[0], start_point[1], 'sr', markersize=10, 
            label='Start (P0)', zorder=5)
    
    num_waypoints = Q_points.shape[0]
    for i in range(1, num_waypoints - 1):
        ax.plot(Q_points[i, 0], Q_points[i, 1], 
                'mo', markersize=8, 
                label=f'Via-point {i} (P{i})', zorder=5)
    
    goal_point = Q_points[-1]
    ax.plot(goal_point[0], goal_point[1], 'go', markersize=10, 
            label=f'Goal (P{num_waypoints-1})', zorder=5)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Multi-Segment Dubins Path Planning', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.show()
    
    # summary
    num_collision_free = jnp.sum(~final_paths.collision)
    print(f"Visualization Summary:")
    print(f"{'='*60}")
    print(f"Total plans: {final_paths.path.shape[0]}")
    print(f"Collision-free paths: {num_collision_free} ({num_collision_free/final_paths.path.shape[0]*100:.1f}%)")
    print(f"Collision paths: {final_paths.path.shape[0] - num_collision_free}")
    print(f"Waypoints: {Q_points.shape[0]}")



if __name__ == "__main__":
    main()