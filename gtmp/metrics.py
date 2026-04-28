"""Path quality metrics for GTMP evaluation.

Includes path diversity (via OTT-JAX Sinkhorn), cosine similarity,
and composite metric computation.
"""
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def entropy_path(paths: jax.Array) -> jax.Array:
    """Compute entropy of pairwise Fréchet distance matrix.

    Parameters
    ----------
    paths : jax.Array, shape (num_paths, num_points, dim)
        Collection of paths.

    Returns
    -------
    jax.Array
        Entropy value.
    """
    frechet_matrix = jnp.linalg.norm(
        paths[:, None, :, :] - paths[None, :, :, :], axis=-1
    ).sum(axis=-1)
    frechet_matrix = frechet_matrix / frechet_matrix.sum()
    return -jnp.sum(frechet_matrix * jnp.log(frechet_matrix + 1e-12))


def solve_ott(
    x: jax.Array, y: jax.Array, eps: float = 5e-2, threshold: float = 1e-3
) -> jax.Array:
    """Solve optimal transport between two point clouds using Sinkhorn.

    Parameters
    ----------
    x, y : jax.Array
        Point clouds.
    eps : float
        Entropic regularization.
    threshold : float
        Convergence threshold.

    Returns
    -------
    tuple of (f, g, primal_cost, n_iters)
    """
    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(
        threshold=threshold,
        max_iterations=200,
        norm_error=2,
        lse_mode=True,
    )
    out = solver(prob)
    return out.f, out.g, out.primal_cost, out.n_iters


def path_diversity(paths: jax.Array) -> jax.Array:
    """Compute average pairwise OT distance between paths.

    Parameters
    ----------
    paths : jax.Array, shape (num_paths, num_points, dim)

    Returns
    -------
    jax.Array
        Mean pairwise Sinkhorn distance.
    """
    num_paths = paths.shape[0]
    path_a, path_b = jnp.triu_indices(num_paths, 1)

    def path_dist(paths, id1, id2):
        _, _, reg_ot, _ = solve_ott(paths[id1], paths[id2])
        return reg_ot

    dists = vmap(path_dist, in_axes=(None, 0, 0))(paths, path_a, path_b)
    return dists.mean()


def path_diversity_np(paths: List[np.ndarray]) -> float:
    """Compute path diversity using numpy paths (non-JIT).

    Parameters
    ----------
    paths : list of np.ndarray

    Returns
    -------
    float
        Mean pairwise Sinkhorn distance.
    """
    num_paths = len(paths)
    dists = []

    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            _, _, reg_ot, _ = solve_ott(paths[i], paths[j])
            dists.append(reg_ot)
    if not dists:
        return 0.0
    return float(np.mean(np.asarray(dists)))


def _compute_cosine_similarity(paths: jax.Array) -> jax.Array:
    """Compute per-segment cosine similarity for a batch of paths.

    Parameters
    ----------
    paths : jax.Array, shape (..., num_points, dim)

    Returns
    -------
    jax.Array, shape (..., num_segments-1)
        Cosine similarity between consecutive segments.
    """
    path_vecs = jnp.diff(paths, axis=-2)
    v1, v2 = path_vecs[..., :-1, :], path_vecs[..., 1:, :]
    v1_norm = jnp.linalg.norm(v1, axis=-1)
    v2_norm = jnp.linalg.norm(v2, axis=-1)
    # Guard against zero-length segments
    safe_v1_norm = jnp.maximum(v1_norm, 1e-12)
    safe_v2_norm = jnp.maximum(v2_norm, 1e-12)
    nv1 = v1 / safe_v1_norm[..., None]
    nv2 = v2 / safe_v2_norm[..., None]
    return jnp.einsum("...i,...i->...", nv1, nv2)


def min_cosin_sim(paths: jax.Array) -> jax.Array:
    """Compute mean of minimum cosine similarities across paths."""
    return _compute_cosine_similarity(paths).min(axis=-1).mean()


def mean_cosin_sim(paths: jax.Array) -> jax.Array:
    """Compute mean of average cosine similarities across paths."""
    return _compute_cosine_similarity(paths).mean(axis=-1).mean()


def min_cosin_sim_np(paths: List[np.ndarray]) -> float:
    """Compute min cosine similarity for list of numpy paths."""
    cosines = []
    for p in paths:
        vecs = np.diff(p, axis=-2)
        v1, v2 = vecs[..., :-1, :], vecs[..., 1:, :]
        n1 = np.maximum(np.linalg.norm(v1, axis=-1), 1e-12)
        n2 = np.maximum(np.linalg.norm(v2, axis=-1), 1e-12)
        nv1 = v1 / n1[..., None]
        nv2 = v2 / n2[..., None]
        cosines.append(np.einsum("...i,...i->...", nv1, nv2).min())
    return float(np.mean(cosines))


def mean_cosin_sim_np(paths: List[np.ndarray]) -> float:
    """Compute mean cosine similarity for list of numpy paths."""
    cosines = []
    for p in paths:
        vecs = np.diff(p, axis=-2)
        v1, v2 = vecs[..., :-1, :], vecs[..., 1:, :]
        n1 = np.maximum(np.linalg.norm(v1, axis=-1), 1e-12)
        n2 = np.maximum(np.linalg.norm(v2, axis=-1), 1e-12)
        nv1 = v1 / n1[..., None]
        nv2 = v2 / n2[..., None]
        cosines.append(np.einsum("...i,...i->...", nv1, nv2).mean())
    return float(np.mean(cosines))


def compute_metrics(data) -> List[float]:
    """Compute comprehensive planning quality metrics.

    Parameters
    ----------
    data : GTMPOutput
        Planning output with path and collision fields.

    Returns
    -------
    list of [collision_free_rate, path_cost, diversity, min_cosine, mean_cosine]
    """
    paths = data.path
    collision_free = 1 - data.collision.mean()
    free_paths = paths[~data.collision]
    if free_paths.shape[0] == 0:
        return [collision_free, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
    path_cost = jnp.linalg.norm(jnp.diff(free_paths, axis=-2), axis=-1).sum(-1).mean(-1)
    entropy = path_diversity(free_paths)
    cosin = min_cosin_sim(free_paths)
    mean_cosin = mean_cosin_sim(free_paths)
    return [collision_free, path_cost, entropy, cosin, mean_cosin]
