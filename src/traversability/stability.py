from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from traversability.terrain import TerrainLayer, world_to_grid


@dataclass(frozen=True)
class RobotFootprint:
    length: float = 0.58
    width: float = 0.42
    clearance: float = 0.18
    sample_rows: int = 5
    sample_cols: int = 5
    max_roll: float = math.radians(34.0)
    max_pitch: float = math.radians(38.0)
    max_support_std: float = 0.22
    max_step: float = 0.58


@dataclass(frozen=True)
class ConfigurationStability:
    roll: float
    pitch: float
    support_std: float
    max_step: float
    min_clearance: float
    stability: float
    feasible: bool


def estimate_configuration_stability(
    layer: TerrainLayer,
    xy: tuple[float, float],
    yaw: float,
    footprint: RobotFootprint | None = None,
) -> ConfigurationStability:
    footprint = footprint or RobotFootprint()
    points = footprint_points(xy, yaw, footprint)
    heights = []
    collision = False
    for point in points:
        row, col = world_to_grid(layer, point)
        if layer.obstacle[row, col] or not np.isfinite(layer.risk[row, col]):
            collision = True
        heights.append(layer.height[row, col])

    heights_array = np.asarray(heights, dtype=np.float64)
    local_points = local_footprint_points(footprint)
    plane = fit_plane(local_points, heights_array)
    predicted = plane[0] * local_points[:, 0] + plane[1] * local_points[:, 1] + plane[2]
    residual = heights_array - predicted

    pitch = math.atan(plane[0])
    roll = math.atan(plane[1])
    support_std = float(np.std(residual))
    max_step = float(np.max(heights_array) - np.min(heights_array))
    chassis_height = float(np.percentile(heights_array, 85) + footprint.clearance)
    min_clearance = float(np.min(chassis_height - heights_array))

    roll_score = 1.0 - min(abs(roll) / footprint.max_roll, 1.0)
    pitch_score = 1.0 - min(abs(pitch) / footprint.max_pitch, 1.0)
    support_score = 1.0 - min(support_std / footprint.max_support_std, 1.0)
    step_score = 1.0 - min(max_step / footprint.max_step, 1.0)
    clearance_score = 1.0 if min_clearance >= 0.0 else 0.0

    stability = (
        0.24 * roll_score
        + 0.26 * pitch_score
        + 0.24 * support_score
        + 0.18 * step_score
        + 0.08 * clearance_score
    )
    feasible = (
        not collision
        and abs(roll) <= footprint.max_roll
        and abs(pitch) <= footprint.max_pitch
        and support_std <= footprint.max_support_std
        and max_step <= footprint.max_step
        and min_clearance >= -0.03
    )

    return ConfigurationStability(
        roll=roll,
        pitch=pitch,
        support_std=support_std,
        max_step=max_step,
        min_clearance=min_clearance,
        stability=float(np.clip(stability, 0.0, 1.0)),
        feasible=feasible,
    )


def footprint_points(
    xy: tuple[float, float],
    yaw: float,
    footprint: RobotFootprint,
) -> list[tuple[float, float]]:
    local = local_footprint_points(footprint)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    points = []
    for px, py in local[:, :2]:
        wx = xy[0] + px * cos_yaw - py * sin_yaw
        wy = xy[1] + px * sin_yaw + py * cos_yaw
        points.append((wx, wy))
    return points


def local_footprint_points(footprint: RobotFootprint) -> np.ndarray:
    xs = np.linspace(-footprint.length / 2.0, footprint.length / 2.0, footprint.sample_cols)
    ys = np.linspace(-footprint.width / 2.0, footprint.width / 2.0, footprint.sample_rows)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def fit_plane(local_xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    design = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.ones(local_xy.shape[0])])
    plane, *_ = np.linalg.lstsq(design, z, rcond=None)
    return plane
