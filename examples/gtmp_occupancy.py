"""GTMP planning demo on 2D occupancy maps."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jax import jit, vmap
import hydra
import omegaconf
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from gtmp.files import get_configs_path, get_data_path
from gtmp.planners import GTMPState, gtmp_plan, gtmp_akima_plan
from gtmp.objectives.occupancy_map import OccupancyMap
from gtmp.metrics import compute_metrics

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_occupancy")
def main(cfg: omegaconf.DictConfig):
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)

    # Environment
    occ = 1.0 - jnp.load((get_data_path() / "real_map" / str(cfg.environment.map_file)).as_posix())
    limits = jnp.array(cfg.environment.limits)
    q = jnp.array(cfg.environment.start_state)
    goals = jnp.array(cfg.environment.goal_state)[None, :]
    occ_map = OccupancyMap.from_prob(occ, limits=limits, threshold=0.1, infinite_cost=True)

    # Configure planner
    planner_state = GTMPState.create(
        q=q,
        goals=goals,
        bounds=limits,
        occ_map=occ_map,
        get_velocity=False,
        **cfg.planner.params,
    )

    # Compile and plan
    if cfg.planner.name == "straight":
        gtmp = jit(vmap(gtmp_plan, in_axes=(0, None)))
    elif cfg.planner.name == "akima":
        gtmp = jit(vmap(gtmp_akima_plan, in_axes=(0, None)))

    keys = jax.random.split(rng_key, cfg.num_plans)

    # Warmup (includes JIT compilation)
    t0 = time.perf_counter()
    paths = gtmp(keys, planner_state)
    jax.block_until_ready(paths.path)
    print(f"JIT compile + first run: {time.perf_counter() - t0:.3f}s")

    # Timed run
    t0 = time.perf_counter()
    paths = gtmp(keys, planner_state)
    jax.block_until_ready(paths.path)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(paths)
    print(f"Planning time: {elapsed:.4f}s")
    print(f"Collision-free: {metrics[0]:.1%}")
    print(f"Avg path length: {metrics[1]:.3f}")
    print(f"Path diversity: {metrics[2]:.4f}")
    print(f"Min cosine sim: {metrics[3]:.4f}")
    print(f"Mean cosine sim: {metrics[4]:.4f}")

    # Visualize
    fig, ax = plt.subplots()
    X = jnp.linspace(*limits[0], occ.shape[0])
    Y = jnp.linspace(*limits[1], occ.shape[1])
    X, Y = jnp.meshgrid(X, Y)
    ax.contourf(X, Y, occ_map.map.T, cmap="Greys")
    free_path = paths.path[~paths.collision]
    coll_path = paths.path[paths.collision]
    for i in range(free_path.shape[0]):
        ax.plot(free_path[i, :, 0], free_path[i, :, 1], "bo--", linewidth=1, markersize=1, alpha=0.7)
    for i in range(coll_path.shape[0]):
        ax.plot(coll_path[i, :, 0], coll_path[i, :, 1], "ro--", linewidth=1, alpha=0.3)
    ax.plot(q[0], q[1], "ro", markersize=5)
    ax.plot(goals[0, 0], goals[0, 1], "go", markersize=5)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    main()
