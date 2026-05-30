from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCloudGrid:
    height: np.ndarray
    obstacle: np.ndarray
    resolution: float
    origin_xy: tuple[float, float]
    points_per_cell: np.ndarray


def sample_point_cloud_from_grid(
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin_xy: tuple[float, float],
    samples_per_cell: int = 2,
    noise_std: float = 0.025,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = height.shape
    xs = origin_xy[0] + np.arange(cols) * resolution
    ys = origin_xy[1] + np.arange(rows) * resolution
    points = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            count = samples_per_cell + int(obstacle[row, col]) * samples_per_cell
            jitter_xy = rng.uniform(-0.45 * resolution, 0.45 * resolution, size=(count, 2))
            z = height[row, col] + rng.normal(0.0, noise_std, size=count)
            cell_points = np.column_stack([x + jitter_xy[:, 0], y + jitter_xy[:, 1], z])
            points.append(cell_points)
    return np.vstack(points).astype(np.float64)


def point_cloud_to_elevation_grid(
    points: np.ndarray,
    resolution: float,
    min_points_per_cell: int = 1,
    obstacle_height_percentile: float = 97.0,
    obstacle_relief_threshold: float = 1.0,
) -> PointCloudGrid:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    origin_xy = (float(min_xy[0]), float(min_xy[1]))
    cols = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
    rows = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1

    height_sum = np.zeros((rows, cols), dtype=np.float64)
    height_min = np.full((rows, cols), np.inf, dtype=np.float64)
    height_max = np.full((rows, cols), -np.inf, dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.int32)

    col_indices = np.clip(((points[:, 0] - origin_xy[0]) / resolution).astype(int), 0, cols - 1)
    row_indices = np.clip(((points[:, 1] - origin_xy[1]) / resolution).astype(int), 0, rows - 1)
    z = points[:, 2]
    np.add.at(height_sum, (row_indices, col_indices), z)
    np.add.at(counts, (row_indices, col_indices), 1)
    np.minimum.at(height_min, (row_indices, col_indices), z)
    np.maximum.at(height_max, (row_indices, col_indices), z)

    observed = counts >= min_points_per_cell
    height = np.divide(height_sum, np.maximum(counts, 1), where=np.ones_like(height_sum, dtype=bool))
    height = fill_missing_height(height, observed)
    height = smooth_height(height, observed)
    relief = np.where(observed, height_max - height_min, 0.0)
    high_threshold = np.percentile(height[observed], obstacle_height_percentile) if np.any(observed) else np.inf
    obstacle = (height >= high_threshold) | (relief >= obstacle_relief_threshold)

    return PointCloudGrid(
        height=height,
        obstacle=obstacle,
        resolution=resolution,
        origin_xy=origin_xy,
        points_per_cell=counts,
    )


def fill_missing_height(height: np.ndarray, observed: np.ndarray, iterations: int = 6) -> np.ndarray:
    result = height.copy()
    if not np.any(observed):
        return np.zeros_like(height)
    result[~observed] = float(np.mean(result[observed]))
    known = observed.copy()
    for _ in range(iterations):
        updated = result.copy()
        for row in range(result.shape[0]):
            for col in range(result.shape[1]):
                if known[row, col]:
                    continue
                r0 = max(0, row - 1)
                r1 = min(result.shape[0], row + 2)
                c0 = max(0, col - 1)
                c1 = min(result.shape[1], col + 2)
                neighbors = known[r0:r1, c0:c1]
                if np.any(neighbors):
                    updated[row, col] = float(np.mean(result[r0:r1, c0:c1][neighbors]))
                    known[row, col] = True
        result = updated
    return result


def smooth_height(height: np.ndarray, observed: np.ndarray) -> np.ndarray:
    padded = np.pad(height, 1, mode="edge")
    result = height.copy()
    weights = np.array(
        [
            [0.05, 0.10, 0.05],
            [0.10, 0.40, 0.10],
            [0.05, 0.10, 0.05],
        ],
        dtype=np.float64,
    )
    for row in range(height.shape[0]):
        for col in range(height.shape[1]):
            patch = padded[row : row + 3, col : col + 3]
            filtered = float(np.sum(patch * weights))
            result[row, col] = 0.75 * height[row, col] + 0.25 * filtered if observed[row, col] else filtered
    return result
