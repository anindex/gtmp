# GTMP Benchmark

Timing benchmarks for GTMP planning on CPU and GPU.  
All measurements are **median of 20 repetitions** after JIT warmup, using `jax.block_until_ready()` for accurate timing.

## Hardware

| | Spec |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5090 (32 GB) |
| **CPU** | AMD (single-threaded, no GPU fallback) |
| **JAX** | 0.10.0 + CUDA 12 |
| **Environment** | 2D occupancy map (`intel.npy`), 20×20 workspace |

## GPU Results (RTX 5090)

### Batch Scaling - Production Config (N=100, M=2, H=100)

| Batch (K) | Total | Per-plan | Free rate |
|-----------|-------|----------|-----------|
| 100 | 2.8 ms | 27.7 µs | 87% |
| 200 | 2.9 ms | 14.4 µs | 87% |
| 500 | 3.5 ms | **7.0 µs** | 89% |
| 1,000 | 4.6 ms | **4.6 µs** | 89% |
| 2,000 | 12.3 ms | **6.1 µs** | 89% |

> At K=1000, GTMP plans **1,000 diverse, collision-checked paths in 4.6 ms** on GPU - **4.6 µs per path** with 89% collision-free rate.

### Parameter Sensitivity (K=100)

**Dream points (N)** - controls path coverage; cost is O(N²)

| N | Total | Per-plan | Free rate |
|---|-------|----------|-----------|
| 10 | 2.6 ms | 26 µs | 6% |
| 50 | 2.5 ms | 25 µs | 60% |
| 100 | 2.7 ms | 27 µs | 90% |
| 200 | 3.3 ms | 33 µs | 100% |

**Probes (H)** - collision checks per edge; linear cost

| H | Total | Per-plan | Free rate |
|---|-------|----------|-----------|
| 5 | 2.6 ms | 26 µs | 100% |
| 25 | 2.6 ms | 26 µs | 75% |
| 100 | 2.5 ms | 25 µs | 56% |

**Layers (M)** - waypoint layers; cost is O(M·N²)

| M | Total | Per-plan | Free rate |
|---|-------|----------|-----------|
| 1 | 2.5 ms | 25 µs | 24% |
| 2 | 2.5 ms | 25 µs | 60% |
| 5 | 2.7 ms | 27 µs | 95% |

**Planner type** (N=100, M=2, H=100)

| Planner | Total | Per-plan |
|---------|-------|----------|
| Straight | 2.7 ms | 27 µs |
| Akima | 7.8 ms | 78 µs |

> On GPU, the total wall-clock time is dominated by kernel launch overhead (~2.5 ms floor). Varying N, H, and M within practical ranges has minimal impact because the computation fully saturates the GPU's parallelism.

## CPU Results (single core)

### Batch Scaling - Production Config (N=100, M=2, H=100)

| Batch (K) | Total | Per-plan | Free rate |
|-----------|-------|----------|-----------|
| 1 | 1.1 ms | 1,100 µs | 100% |
| 10 | 9.1 ms | 912 µs | 100% |
| 100 | 80.2 ms | **802 µs** | 87% |
| 500 | 407.3 ms | 815 µs | 89% |

### Parameter Sensitivity (K=100)

**Dream points (N)** - dominant scaling factor on CPU

| N | Total | Per-plan | Free rate |
|---|-------|----------|-----------|
| 10 | 0.6 ms | 6 µs | 6% |
| 50 | 12.2 ms | 122 µs | 60% |
| 100 | 43.4 ms | 434 µs | 90% |
| 200 | 225.0 ms | 2,250 µs | 100% |

**Planner type** (N=100, M=2, H=100)

| Planner | Total | Per-plan |
|---------|-------|----------|
| Straight | 82.0 ms | 820 µs |
| Akima | 184.3 ms | 1,843 µs |

## GPU vs CPU Speedup

| Config (K=100) | CPU | GPU | Speedup |
|----------------|-----|-----|---------|
| N=100, M=2, H=100 | 80.2 ms | 2.8 ms | **29×** |
| N=200, M=2, H=50 | 225.0 ms | 3.3 ms | **68×** |
| N=50, M=5, H=50 | 53.0 ms | 2.7 ms | **20×** |

At larger batch sizes, the speedup is even more dramatic:

| Config | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| K=500, N=100 | 407.3 ms | 3.5 ms | **116×** |
| K=1000, N=100 | ~815 ms* | 4.6 ms | **~177×** |

*CPU K=1000 extrapolated from linear scaling.

## Scaling Characteristics

On **GPU**, total wall-clock time is nearly flat across N, H, and M variations at K=100 (~2.5-3.3 ms). This ~2.5 ms floor is the kernel launch and synchronization overhead. The GPU fully parallelizes both the batch dimension (K plans via `jax.vmap`) and the internal computation (N² cost matrices, N×H probe evaluations). Per-plan amortized cost drops inversely with K until the GPU saturates.

On **CPU**, time scales as `K × (N²·M + N·M·H)`. N is the dominant factor due to the quadratic cost matrix. There is no parallelism benefit from increasing K - each plan is computed sequentially.

## Reproducing

```bash
# Install (see README.md)
pip install -e .

# Run benchmark
python examples/benchmark_timing.py
```
