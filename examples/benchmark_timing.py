"""GTMP timing benchmark - measures per-plan latency across configurations.

Reports wall-clock time for the planning step only (excluding JIT compilation).
All measurements use jax.block_until_ready() for accurate timing.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import jit, vmap
import time
import warnings

warnings.filterwarnings("ignore")

from gtmp.planners import GTMPState, gtmp_plan, gtmp_akima_plan
from gtmp.objectives.occupancy_map import OccupancyMap
from gtmp.files import get_data_path


def benchmark_config(occ_map, q, goals, limits, num_plans, num_dreams, num_layers, num_probes, planner_fn, label):
    """Benchmark a single configuration, returning median per-plan time."""
    state = GTMPState.create(
        q=q, goals=goals, bounds=limits, occ_map=occ_map,
        num_dreams=num_dreams, num_layers=num_layers, num_probes=num_probes,
    )
    keys = jax.random.split(jax.random.PRNGKey(42), num_plans)
    planner = jit(vmap(planner_fn, in_axes=(0, None)))

    # Warmup (JIT compilation)
    result = planner(keys, state)
    jax.block_until_ready(result.path)

    # Timed runs - take median of 20 repetitions
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        result = planner(keys, state)
        jax.block_until_ready(result.path)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    free = int((~result.collision).sum())
    per_plan = median / num_plans

    print(f"  {label:40s}  total={median*1000:8.2f}ms  per_plan={per_plan*1e6:8.1f}µs  free={free}/{num_plans}")
    return median, per_plan


if __name__ == "__main__":
    device = jax.devices()[0]
    print(f"Device: {device.platform} ({device.device_kind})")
    print()

    # Load environment
    occ = 1.0 - jnp.load((get_data_path() / "real_map" / "intel.npy").as_posix())
    limits = jnp.array([[0.0, 20.0], [0.0, 20.0]])
    q = jnp.array([5.5, 4.9])
    goals = jnp.array([[15.5, 16.2]])
    occ_map = OccupancyMap.from_prob(occ, limits=limits, threshold=0.1, infinite_cost=True)

    # ===== Part 1: Vary num_plans (batch size) =====
    print("=" * 90)
    print("Batch size sweep (N=100, M=2, H=100, straight)")
    print("=" * 90)
    for K in [1, 10, 50, 100, 500]:
        benchmark_config(occ_map, q, goals, limits, K, 100, 2, 100, gtmp_plan, f"K={K:4d}")

    # ===== Part 2: Vary num_dreams =====
    print()
    print("=" * 90)
    print("Dream points sweep (K=100, M=2, H=50, straight)")
    print("=" * 90)
    for N in [10, 25, 50, 100, 200]:
        benchmark_config(occ_map, q, goals, limits, 100, N, 2, 50, gtmp_plan, f"N={N:4d}")

    # ===== Part 3: Vary num_probes =====
    print()
    print("=" * 90)
    print("Probe sweep (K=100, N=50, M=2, straight)")
    print("=" * 90)
    for H in [5, 10, 25, 50, 100]:
        benchmark_config(occ_map, q, goals, limits, 100, 50, 2, H, gtmp_plan, f"H={H:4d}")

    # ===== Part 4: Vary num_layers =====
    print()
    print("=" * 90)
    print("Layer sweep (K=100, N=50, H=50, straight)")
    print("=" * 90)
    for M in [1, 2, 3, 5]:
        benchmark_config(occ_map, q, goals, limits, 100, 50, M, 50, gtmp_plan, f"M={M:4d}")

    # ===== Part 5: Straight vs Akima =====
    print()
    print("=" * 90)
    print("Planner comparison (K=100, N=100, M=2, H=100)")
    print("=" * 90)
    benchmark_config(occ_map, q, goals, limits, 100, 100, 2, 100, gtmp_plan, "Straight")
    benchmark_config(occ_map, q, goals, limits, 100, 100, 2, 100, gtmp_akima_plan, "Akima")

    # ===== Part 6: Minimal config (fastest possible) =====
    print()
    print("=" * 90)
    print("Minimal config (fastest possible)")
    print("=" * 90)
    for K in [1, 10, 100]:
        benchmark_config(occ_map, q, goals, limits, K, 10, 1, 5, gtmp_plan, f"K={K:4d}, N=10, M=1, H=5")

    # ===== Part 7: Empty map (no obstacles) =====
    print()
    print("=" * 90)
    print("Empty map (no obstacles, K=100, straight)")
    print("=" * 90)
    empty_occ = jnp.zeros((100, 100))
    empty_map = OccupancyMap.from_prob(empty_occ, limits=limits, threshold=0.5, infinite_cost=True)
    benchmark_config(empty_map, q, goals, limits, 100, 50, 2, 50, gtmp_plan, "N=50, M=2, H=50 (empty)")
    benchmark_config(empty_map, q, goals, limits, 100, 100, 2, 100, gtmp_plan, "N=100, M=2, H=100 (empty)")

    print()
    print(f"Benchmark complete on {device.platform} ({device.device_kind}).")
    print("See BENCHMARK.md for analysis.")
