from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap
from traversability.stability import estimate_configuration_stability_on_map


@dataclass(frozen=True)
class OptimizationConfig:
    iterations: int = 2
    sample_offsets: tuple[float, ...] = (-0.25, 0.0, 0.25)
    length_weight: float = 1.0
    risk_weight: float = 2.2
    curvature_weight: float = 0.7
    stability_weight: float = 1.4
    max_risk: float = 0.92
    min_stability: float = 0.16


def optimize_path(
    terrain_map: ImplicitTerrainMap,
    path: list[tuple[float, float]],
    config: OptimizationConfig | None = None,
) -> list[tuple[float, float]]:
    config = config or OptimizationConfig()
    if len(path) <= 2:
        return path
    optimized = list(path)
    for _ in range(config.iterations):
        changed = False
        for index in range(1, len(optimized) - 1, 2):
            previous = optimized[index - 1]
            current = optimized[index]
            next_point = optimized[index + 1]
            best = current
            best_cost = local_cost(terrain_map, previous, current, next_point, config)
            for candidate in lateral_candidates(previous, current, next_point, config.sample_offsets):
                if not point_is_safe(terrain_map, candidate, previous, next_point, config):
                    continue
                candidate_cost = local_cost(terrain_map, previous, candidate, next_point, config)
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
            if best != current:
                optimized[index] = best
                changed = True
        if not changed:
            break
    return optimized


def lateral_candidates(
    previous: tuple[float, float],
    current: tuple[float, float],
    next_point: tuple[float, float],
    offsets: tuple[float, ...],
) -> list[tuple[float, float]]:
    tangent = np.array([next_point[0] - previous[0], next_point[1] - previous[1]], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        return [current]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64) / norm
    base = np.array(current, dtype=np.float64)
    return [tuple((base + offset * normal).tolist()) for offset in offsets]


def point_is_safe(
    terrain_map: ImplicitTerrainMap,
    point: tuple[float, float],
    previous: tuple[float, float],
    next_point: tuple[float, float],
    config: OptimizationConfig,
) -> bool:
    query = terrain_map.query(point)
    if query.obstacle or query.risk > config.max_risk:
        return False
    yaw = math.atan2(next_point[1] - previous[1], next_point[0] - previous[0])
    stability = estimate_configuration_stability_on_map(terrain_map, point, yaw)
    return (
        stability.feasible
        and stability.stability >= config.min_stability
        and segment_is_safe(terrain_map, previous, point, config)
        and segment_is_safe(terrain_map, point, next_point, config)
    )


def segment_is_safe(
    terrain_map: ImplicitTerrainMap,
    start: tuple[float, float],
    end: tuple[float, float],
    config: OptimizationConfig,
) -> bool:
    distance = math.dist(start, end)
    steps = max(2, int(distance / (terrain_map.resolution * 0.75)))
    yaw = math.atan2(end[1] - start[1], end[0] - start[0])
    for index in range(steps + 1):
        t = index / steps
        point = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        query = terrain_map.query(point)
        if query.obstacle or query.risk > config.max_risk:
            return False
        if index == 0 or index == steps or index == steps // 2:
            stability = estimate_configuration_stability_on_map(terrain_map, point, yaw)
            if not stability.feasible or stability.stability < config.min_stability:
                return False
    return True


def local_cost(
    terrain_map: ImplicitTerrainMap,
    previous: tuple[float, float],
    current: tuple[float, float],
    next_point: tuple[float, float],
    config: OptimizationConfig,
) -> float:
    query = terrain_map.query(current)
    yaw = math.atan2(next_point[1] - previous[1], next_point[0] - previous[0])
    stability = estimate_configuration_stability_on_map(terrain_map, current, yaw)
    length = math.dist(previous, current) + math.dist(current, next_point)
    curvature = curvature_cost(previous, current, next_point)
    instability = 1.0 - stability.stability
    infeasible_penalty = 5.0 if not stability.feasible else 0.0
    obstacle_penalty = 10.0 if query.obstacle else 0.0
    return (
        config.length_weight * length
        + config.risk_weight * min(query.risk, 1.5)
        + config.curvature_weight * curvature
        + config.stability_weight * instability
        + infeasible_penalty
        + obstacle_penalty
    )


def curvature_cost(
    previous: tuple[float, float],
    current: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    heading_a = math.atan2(current[1] - previous[1], current[0] - previous[0])
    heading_b = math.atan2(next_point[1] - current[1], next_point[0] - current[0])
    return abs(math.atan2(math.sin(heading_b - heading_a), math.cos(heading_b - heading_a)))
