"""GTMP: Global Tensor Motion Planning in JAX.

A massively parallelized motion planner using MDP value iteration
over randomly sampled dream points, with optional Akima spline smoothing.
"""

__version__ = "0.2.0"

from gtmp.planners import (
    GTMPState,
    GTMPOutput,
    gtmp_plan,
    gtmp_akima_plan,
    interpolate_path,
)
from gtmp.splines import LayerAkima1DInterpolator, LayerPPoly
from gtmp.objectives.occupancy_map import OccupancyMap
from gtmp.metrics import compute_metrics

__all__ = [
    "GTMPState",
    "GTMPOutput",
    "gtmp_plan",
    "gtmp_akima_plan",
    "interpolate_path",
    "LayerAkima1DInterpolator",
    "LayerPPoly",
    "OccupancyMap",
    "compute_metrics",
]
