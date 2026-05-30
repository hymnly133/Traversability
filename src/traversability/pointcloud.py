from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PointCloudGrid:
    height: np.ndarray
    obstacle: np.ndarray
    resolution: float
    origin_xy: tuple[float, float]
    points_per_cell: np.ndarray


def load_point_cloud(path: Path | str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix in {".csv", ".xyz", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        points = np.loadtxt(path, delimiter=delimiter, comments="#")
    elif suffix == ".ply":
        points = load_ascii_ply(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")
    return validate_points(points)


def save_point_cloud(path: Path | str, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = validate_points(points)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, points)
    elif suffix == ".csv":
        np.savetxt(path, points, delimiter=",", fmt="%.6f")
    elif suffix in {".xyz", ".txt"}:
        np.savetxt(path, points, fmt="%.6f")
    elif suffix == ".ply":
        save_ascii_ply(path, points)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")


def validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point cloud must be an Nx3 array or have at least x,y,z columns")
    points = points[:, :3]
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("point cloud contains no finite xyz points")
    return points


def load_ascii_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        first = file.readline().strip()
        if first != "ply":
            raise ValueError("Only ASCII PLY files are supported")
        vertex_count = None
        is_ascii = False
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                is_ascii = True
            elif stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if not is_ascii or vertex_count is None:
            raise ValueError("Only ASCII PLY with element vertex is supported")
        rows = []
        for _ in range(vertex_count):
            parts = file.readline().split()
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return validate_points(np.asarray(rows, dtype=np.float64))


def save_ascii_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for x, y, z in points:
            file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


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
    points = validate_points(points)
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
