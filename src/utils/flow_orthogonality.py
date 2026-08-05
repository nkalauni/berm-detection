"""
Flow-orthogonality channel: |sin(theta_struct - theta_flow)|.

High where a linear structure (e.g. a berm, from the Hessian of the 15m
residual relief) runs perpendicular to the D-infinity flow direction;
low where it runs parallel (e.g. a road along a drainage divide).

Convention note (verified empirically, not assumed -- see synthetic tests):
  - hessian_matrix(..., order='rc') returns (H_rr, H_rc, H_cc). Mapping to
    image x=col/y=row WITHOUT negating H_rc and WITHOUT the "+pi/2"
    ridge-normal-to-parallel shift is what matches ground truth on
    synthetic ridges at 0/45/90/135 degrees. The raw eigenvector-angle
    formula already gives the ridge-PARALLEL direction for a positive
    (concave-down) relief bump.
  - WhiteboxTools' d_inf_pointer output is degrees clockwise-from-north
    (verified against two synthetic slope tests). Converted to standard
    math radians (CCW-from-east) via (90 - flowdir_deg) before combining
    with the Hessian angle, which is already in that convention.
"""

import numpy as np
from skimage.feature import hessian_matrix


def structure_angle(residual: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Ridge-parallel orientation of `residual`, in radians, standard math
    convention (0 = east/+x, CCW positive), mod pi (undirected line).
    """
    H_rr, H_rc, H_cc = hessian_matrix(
        residual, sigma=sigma, order="rc", use_gaussian_derivatives=True
    )
    Hxx, Hyy, Hxy = H_cc, H_rr, H_rc
    return 0.5 * np.arctan2(2 * Hxy, Hxx - Hyy)


def flow_orthogonality(
    residual: np.ndarray,
    flowdir_deg_cw_from_north: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Args:
        residual: (H, W) residual-relief raster (e.g. 15m diff-from-mean-elev).
        flowdir_deg_cw_from_north: (H, W) D-infinity flow direction in degrees,
            clockwise from north (WhiteboxTools d_inf_pointer convention).
        sigma: Gaussian derivative scale for the Hessian, ~half the crest width.

    Returns:
        (H, W) float32 in [0, 1]. ~1 = structure perpendicular to flow (berm),
        ~0 = structure parallel to flow (e.g. road along a divide).
    """
    theta_struct = structure_angle(residual, sigma=sigma)
    theta_flow = np.deg2rad(90.0 - flowdir_deg_cw_from_north)
    omega = np.abs(np.sin(theta_struct - theta_flow))
    return omega.astype(np.float32)
