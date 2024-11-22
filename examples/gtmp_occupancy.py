import jax
from jax import jit, vmap, random
import hydra
import omegaconf
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from chrono import Timer
from gtmp.files import get_configs_path, get_data_path
from gtmp.planners import GTMPState, gtmp_plan, gtmp_akima_plan
from gtmp.objectives.occupancy_map import OccupancyMap
from gtmp.metrics import compute_metrics


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_occupancy")
def main(cfg: omegaconf.DictConfig):
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)

    # Environment
    occ = 1. - jnp.load((get_data_path() / 'real_map' / str(cfg.environment.map_file)).as_posix())
    limits = jnp.array(cfg.environment.limits)
    q = jnp.array(cfg.environment.start_state)
    goals = jnp.array(cfg.environment.goal_state)[None, :]
    occ_map = OccupancyMap.from_prob(occ, limits=limits, threshold=0.1, infinite_cost=True)

    # planner
    planner_state = GTMPState.create(
        q=q,
        goals=goals,
        bounds=limits,
        transition_field=occ_map,
        occ_map=occ_map,
        get_velocity=False,
        **cfg.planner.params
    )

    # warmup
    if cfg.planner.name == 'straight':
        gtmp = jit(vmap(gtmp_plan, in_axes=(0, None)))
    elif cfg.planner.name == 'akima':
        gtmp = jit(vmap(gtmp_akima_plan, in_axes=(0, None)))
    # plan
    keys = jax.random.split(rng_key, cfg.num_plans)
    start = time.time()
    paths = gtmp(keys, planner_state)
    print(f"JIT Time taken: {time.time() - start} seconds")
    with Timer() as timer:    
        paths = gtmp(keys, planner_state)
    metrics = compute_metrics(paths)
    print(f"Time taken: {timer.elapsed} seconds")
    print(f"Collision-free percentage: {metrics[0]}")
    print(f"Averaged path length: {metrics[1]}")
    print(f"Path Diversities: {metrics[2]}")
    print(f"Min Cosin: {metrics[3]}")
    print(f"Mean Cosin: {metrics[4]}")

    fig, ax = plt.subplots()
    X = jnp.linspace(*limits[0], occ.shape[0])
    Y = jnp.linspace(*limits[1], occ.shape[1])
    X, Y = jnp.meshgrid(X, Y)
    # ax.contourf(X, Y, occ.T, cmap='Greys')
    ax.contourf(X, Y, occ_map.map.T, cmap='Greys')
    free_path = paths.path[~paths.collision]
    # path_vel = paths.path_vel[~paths.collision]   # getting path velocities here from Akima splines (which are constant vel per segments)
    coll_path = paths.path[paths.collision]
    for i in range(free_path.shape[0]):
        ax.plot(free_path[i, :, 0], free_path[i, :, 1], 'bo--', linewidth=1, markersize=1, alpha=0.7)
    for i in range(coll_path.shape[0]):
        ax.plot(coll_path[i, :, 0], coll_path[i, :, 1], 'ro--', linewidth=1, alpha=0.3)
    ax.plot(q[0], q[1], 'ro', markersize=5)
    ax.plot(goals[0, 0], goals[0, 1], 'go', markersize=5)
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    main()
