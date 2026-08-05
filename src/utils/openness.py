"""
Positive topographic openness (Yokoyama, Kikuchi & Ohuchi, 2002).

WhiteboxTools' open-source build doesn't include an openness tool (it's a
paid-extension feature there), so this reimplements the published formula
directly:

    phi_d(x)   = max_{r<=L} arctan((Z(x + r*d) - Z(x)) / (r * step_d))
    Openness+  = mean_d (90 deg - phi_d)

over the 8 principal compass directions d, using GPU tensor shifts for
speed on large rasters.
"""

import numpy as np
import torch

# 8 principal directions as (row_step, col_step) in cell units, and the
# per-step ground distance multiplier (1 for cardinal, sqrt(2) for diagonal)
_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]


def positive_openness(
    dem: np.ndarray,
    valid: np.ndarray,
    radius_cells: int = 30,
    cell_size: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    Args:
        dem:    (H, W) float32 elevation, arbitrary fill value where invalid.
        valid:  (H, W) bool, True where dem is real data.
        radius_cells: L in the formula, search radius in cells.
        cell_size:    ground resolution (m/cell).
        device: torch device string.

    Returns:
        (H, W) float32 positive openness in degrees, NaN where invalid.
    """
    H, W = dem.shape
    z = torch.from_numpy(np.where(valid, dem, np.nan).astype(np.float32)).to(device)

    angle_sum = torch.zeros((H, W), dtype=torch.float32, device=device)

    for dy, dx in _DIRECTIONS:
        step_dist = cell_size * np.hypot(dy, dx)
        best_for_dir = torch.full((H, W), -90.0, dtype=torch.float32, device=device)
        for r in range(1, radius_cells + 1):
            shifted = torch.roll(z, shifts=(-r * dy, -r * dx), dims=(0, 1))
            # invalidate wrapped-around edges
            if dy > 0:
                shifted[-r * dy:, :] = float("nan")
            elif dy < 0:
                shifted[: -r * dy, :] = float("nan")
            if dx > 0:
                shifted[:, -r * dx:] = float("nan")
            elif dx < 0:
                shifted[:, : -r * dx] = float("nan")

            diff = shifted - z
            angle_deg = torch.rad2deg(torch.atan2(diff, torch.tensor(r * step_dist, device=device)))
            angle_deg = torch.nan_to_num(angle_deg, nan=-90.0)
            best_for_dir = torch.maximum(best_for_dir, angle_deg)
        angle_sum += best_for_dir

    openness = 90.0 - (angle_sum / len(_DIRECTIONS))
    result = openness.cpu().numpy()
    result[~valid] = np.nan
    return result


def positive_openness_chunked(
    dem: np.ndarray,
    valid: np.ndarray,
    radius_cells: int = 30,
    cell_size: float = 1.0,
    device: str = "cuda",
    chunk_rows: int = 4000,
) -> np.ndarray:
    """
    Same as positive_openness, but processes the raster in horizontal strips
    with a `radius_cells`-row halo on each side, so chunk boundaries don't
    truncate the search radius. Needed for rasters too large to hold in GPU
    memory alongside the ~6 working tensors positive_openness allocates.
    """
    H, W = dem.shape
    result = np.full((H, W), np.nan, dtype=np.float32)

    for start in range(0, H, chunk_rows):
        end = min(start + chunk_rows, H)
        halo_lo = max(0, start - radius_cells)
        halo_hi = min(H, end + radius_cells)

        dem_chunk = dem[halo_lo:halo_hi]
        valid_chunk = valid[halo_lo:halo_hi]
        chunk_result = positive_openness(
            dem_chunk, valid_chunk, radius_cells=radius_cells,
            cell_size=cell_size, device=device,
        )

        # crop off the halo before writing into the final array
        keep_lo = start - halo_lo
        keep_hi = keep_lo + (end - start)
        result[start:end] = chunk_result[keep_lo:keep_hi]

    return result
