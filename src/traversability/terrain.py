from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerrainLayer:
    name: str
    height: np.ndarray
    obstacle: np.ndarray
    resolution: float
    origin_xy: tuple[float, float]
    slope: np.ndarray
    roughness: np.ndarray
    step: np.ndarray
    risk: np.ndarray


@dataclass(frozen=True)
class TerrainPyramid:
    layers: list[TerrainLayer]

    @property
    def finest(self) -> TerrainLayer:
        return self.layers[0]

    @property
    def coarsest(self) -> TerrainLayer:
        return self.layers[-1]


def generate_large_rough_terrain(
    size: int = 224,
    resolution: float = 0.25,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)

    height = (
        0.72 * np.sin(2.9 * np.pi * xx + 0.4)
        + 0.52 * np.cos(2.5 * np.pi * yy - 0.2)
        + 0.36 * np.sin(4.7 * np.pi * (xx + yy))
        + 0.22 * np.cos(7.5 * np.pi * (xx - 0.45 * yy))
        + 0.18 * np.sin(13.0 * xx + 6.0 * np.cos(yy))
        + 0.22 * rng.normal(size=(size, size))
    )

    ridge = 2.1 * np.exp(-((xx + 0.12) ** 2 / 0.010 + (yy - 0.08) ** 2 / 0.40))
    cross_ridge = 1.25 * np.exp(-(((yy - 0.48 * xx) - 0.12) ** 2 / 0.014 + (xx + 0.22) ** 2 / 0.70))
    trench = -1.35 * np.exp(-((xx - 0.33) ** 2 / 0.014 + (yy + 0.20) ** 2 / 0.13))
    side_trench = -0.72 * np.exp(-(((yy + 0.55 * xx) + 0.08) ** 2 / 0.010 + (xx - 0.15) ** 2 / 0.62))
    ramp = 1.05 * np.clip((xx + 0.58) / 0.52, 0.0, 1.0) * np.exp(-((yy + 0.55) ** 2) / 0.18)
    height = height + ridge + cross_ridge + trench + side_trench + ramp

    for _ in range(42):
        cx, cy = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
        radius_x = rng.uniform(0.018, 0.070)
        radius_y = rng.uniform(0.018, 0.090)
        sign = rng.choice([-1.0, 1.0], p=[0.35, 0.65])
        mound = np.exp(-(((xx - cx) / radius_x) ** 2 + ((yy - cy) / radius_y) ** 2))
        height += sign * rng.uniform(0.18, 0.65) * mound

    obstacle = np.zeros((size, size), dtype=bool)
    obstacle |= ((xx + 0.40) ** 2 / 0.020 + (yy - 0.34) ** 2 / 0.050) < 1.0
    obstacle |= ((xx - 0.48) ** 2 / 0.030 + (yy + 0.42) ** 2 / 0.035) < 1.0
    obstacle |= (np.abs(xx - 0.05) < 0.040) & (yy > -0.70) & (yy < 0.32)
    obstacle |= (np.abs(yy + 0.30 * xx - 0.18) < 0.035) & (xx > -0.74) & (xx < 0.70)

    debris = rng.random((size, size)) > 0.990
    obstacle |= debris
    corridor = np.abs(yy - (0.58 * xx - 0.08)) < 0.16
    corridor |= np.abs(yy + 0.20) < 0.10
    corridor |= np.abs(yy - (0.95 * xx - 0.02)) < 0.18
    obstacle &= ~corridor
    height = np.where(corridor, 0.32 * np.sin(4.2 * xx) + 0.18 * np.cos(3.6 * yy), height)
    height[obstacle] += 3.1

    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    return height.astype(np.float64), obstacle, origin


def make_pyramid(
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin_xy: tuple[float, float],
    levels: int = 4,
) -> TerrainPyramid:
    layers: list[TerrainLayer] = []
    current_height = height
    current_obstacle = obstacle
    current_resolution = resolution
    for level in range(levels):
        layers.append(
            analyze_layer(
                name=f"level_{level}",
                height=current_height,
                obstacle=current_obstacle,
                resolution=current_resolution,
                origin_xy=origin_xy,
            )
        )
        if min(current_height.shape) < 24:
            break
        current_height = block_reduce(current_height, factor=2, reducer="mean")
        current_obstacle = block_reduce(current_obstacle.astype(float), factor=2, reducer="max") > 0
        current_resolution *= 2.0
    return TerrainPyramid(layers=layers)


def analyze_layer(
    name: str,
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin_xy: tuple[float, float],
) -> TerrainLayer:
    gy, gx = np.gradient(height, resolution, edge_order=1)
    slope = np.hypot(gx, gy)
    roughness = rolling_std(height, radius=2)
    step = local_relief(height, radius=1)

    slope_cost = np.clip(slope / 1.2, 0.0, 1.0)
    roughness_cost = np.clip(roughness / 0.35, 0.0, 1.0)
    step_cost = np.clip(step / 0.65, 0.0, 1.0)
    risk = 0.45 * slope_cost + 0.35 * roughness_cost + 0.20 * step_cost
    risk = np.where(obstacle, np.inf, risk)

    return TerrainLayer(
        name=name,
        height=height,
        obstacle=obstacle,
        resolution=resolution,
        origin_xy=origin_xy,
        slope=slope,
        roughness=roughness,
        step=step,
        risk=risk,
    )


def crop_local_window(
    name: str,
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin_xy: tuple[float, float],
    center_xy: tuple[float, float],
    radius: float,
) -> TerrainLayer:
    center_col = int(round((center_xy[0] - origin_xy[0]) / resolution))
    center_row = int(round((center_xy[1] - origin_xy[1]) / resolution))
    radius_cells = max(1, int(np.ceil(radius / resolution)))
    row0 = max(0, center_row - radius_cells)
    row1 = min(height.shape[0], center_row + radius_cells + 1)
    col0 = max(0, center_col - radius_cells)
    col1 = min(height.shape[1], center_col + radius_cells + 1)
    local_origin = (origin_xy[0] + col0 * resolution, origin_xy[1] + row0 * resolution)
    return analyze_layer(
        name=name,
        height=height[row0:row1, col0:col1].copy(),
        obstacle=obstacle[row0:row1, col0:col1].copy(),
        resolution=resolution,
        origin_xy=local_origin,
    )


def block_reduce(values: np.ndarray, factor: int, reducer: str) -> np.ndarray:
    rows = values.shape[0] // factor * factor
    cols = values.shape[1] // factor * factor
    trimmed = values[:rows, :cols]
    blocks = trimmed.reshape(rows // factor, factor, cols // factor, factor)
    if reducer == "mean":
        return blocks.mean(axis=(1, 3))
    if reducer == "max":
        return blocks.max(axis=(1, 3))
    raise ValueError(f"Unsupported reducer: {reducer}")


def rolling_std(values: np.ndarray, radius: int) -> np.ndarray:
    mean = rolling_mean(values, radius)
    mean_sq = rolling_mean(values * values, radius)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def rolling_mean(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    rows, cols = values.shape
    result = np.zeros_like(values, dtype=np.float64)
    window_area = float((2 * radius + 1) ** 2)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result += padded[dy : dy + rows, dx : dx + cols]
    return result / window_area


def local_relief(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    rows, cols = values.shape
    local_min = np.full_like(values, np.inf, dtype=np.float64)
    local_max = np.full_like(values, -np.inf, dtype=np.float64)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            patch = padded[dy : dy + rows, dx : dx + cols]
            local_min = np.minimum(local_min, patch)
            local_max = np.maximum(local_max, patch)
    return local_max - local_min


def world_to_grid(layer: TerrainLayer, xy: tuple[float, float]) -> tuple[int, int]:
    x, y = xy
    col = int(round((x - layer.origin_xy[0]) / layer.resolution))
    row = int(round((y - layer.origin_xy[1]) / layer.resolution))
    row = int(np.clip(row, 0, layer.height.shape[0] - 1))
    col = int(np.clip(col, 0, layer.height.shape[1] - 1))
    return row, col


def grid_to_world(layer: TerrainLayer, cell: tuple[int, int]) -> tuple[float, float]:
    row, col = cell
    x = layer.origin_xy[0] + col * layer.resolution
    y = layer.origin_xy[1] + row * layer.resolution
    return x, y
