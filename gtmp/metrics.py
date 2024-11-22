import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, random, jit, lax
from typing import Tuple, Optional, Union, Any, List, Dict

import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


# @jit
def entropy_path(paths: jax.Array) -> jax.Array:
    frechet_matrix = jnp.linalg.norm(paths[:, None, :, :] - paths[None, :, :, :], axis=-1).sum(axis=-1)
    # normalize
    frechet_matrix = frechet_matrix / frechet_matrix.sum()
    # compute entropy
    entropy = -jnp.sum(frechet_matrix * jnp.log(frechet_matrix + 1e-12))
    return entropy


# @jit
def solve_ott(x: jax.Array, y: jax.Array, eps: float = 5e-2, threshold: float = 1e-3) -> jax.Array:
    n, m = x.shape[0], y.shape[0]
    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(
        threshold=threshold,
        max_iterations=200,
        norm_error=2,
        lse_mode=True,
    )
    out = solver(prob)
    # # center dual variables to facilitate comparison
    f, g = out.f, out.g
    # f, g = f - jnp.mean(f), g + jnp.mean(f)
    # a, b = jnp.ones(n) / n, jnp.ones(m) / m
    # reg_ot = jnp.sum(f * a) + jnp.sum(g * b)
    return f, g, out.primal_cost, out.n_iters


# @jit
def path_diversity(paths: jax.Array) -> jax.Array:
    num_paths = paths.shape[0]
    path_a, path_b = jnp.triu_indices(num_paths, 1)
    
    def path_dist(paths, id1, id2):
        _, _, reg_ot, _ = solve_ott(paths[id1], paths[id2])
        return reg_ot
    
    dists = vmap(path_dist, in_axes=(None, 0, 0))(paths, path_a, path_b)
    return dists.mean()


def path_diversity_np(paths: List[np.ndarray]) -> float:
    num_paths = len(paths)
    
    def path_dist(path1, path2):
        _, _, reg_ot, _ = solve_ott(path1, path2)
        return reg_ot
    
    dists = []
    for i in range(num_paths):
        for j in range(i, num_paths):
            dists.append(path_dist(paths[i], paths[j]))
    dists = np.asarray(dists)
    return dists.mean()


# @jit
def min_cosin_sim(paths: jax.Array) -> jax.Array:
    path_vecs =  jnp.diff(paths, axis=-2)
    v1, v2 = path_vecs[..., :-1, :], path_vecs[..., 1:, :]
    v1_norm, v2_norm = jnp.linalg.norm(v1, axis=-1), jnp.linalg.norm(v2, axis=-1)
    nv1, nv2 = v1 / v1_norm[..., None], v2 / v2_norm[..., None]
    cosin_sim = jnp.einsum('...i,...i->...', nv1, nv2).min(axis=-1)
    return cosin_sim.mean()


# @jit
def mean_cosin_sim(paths: jax.Array) -> jax.Array:
    path_vecs =  jnp.diff(paths, axis=-2)
    v1, v2 = path_vecs[..., :-1, :], path_vecs[..., 1:, :]
    v1_norm, v2_norm = jnp.linalg.norm(v1, axis=-1), jnp.linalg.norm(v2, axis=-1)
    nv1, nv2 = v1 / v1_norm[..., None], v2 / v2_norm[..., None]
    cosin_sim = jnp.einsum('...i,...i->...', nv1, nv2).mean(axis=-1)
    return cosin_sim.mean()


def min_cosin_sim_np(paths: List[np.ndarray]) -> float:
    num_paths = len(paths)
    path_vecs =  [np.diff(p, axis=-2) for p in paths]
    v1, v2 = [p[..., :-1, :] for p in path_vecs], [p[..., 1:, :] for p in path_vecs]
    nv1, nv2 = [v / np.linalg.norm(v, axis=-1)[..., None] for v in v1], [v / np.linalg.norm(v, axis=-1)[..., None] for v in v2]
    cosin_sim = [np.einsum('...i,...i->...', nv1[i], nv2[i]).min() for i in range(num_paths)]
    return np.mean(cosin_sim)


def mean_cosin_sim_np(paths: List[np.ndarray]) -> float:
    num_paths = len(paths)
    path_vecs =  [np.diff(p, axis=-2) for p in paths]
    v1, v2 = [p[..., :-1, :] for p in path_vecs], [p[..., 1:, :] for p in path_vecs]
    nv1, nv2 = [v / np.linalg.norm(v, axis=-1)[..., None] for v in v1], [v / np.linalg.norm(v, axis=-1)[..., None] for v in v2]
    cosin_sim = [np.einsum('...i,...i->...', nv1[i], nv2[i]).mean() for i in range(num_paths)]
    return np.mean(cosin_sim)


def compute_metrics(data) -> List[float]:
    # path cost
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
