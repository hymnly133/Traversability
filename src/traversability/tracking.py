from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap
from traversability.planner import resample_polyline
from traversability.stability import estimate_configuration_stability_on_map


@dataclass(frozen=True)
class TrackerConfig:
    dt: float = 0.08
    speed: float = 0.75
    lookahead: float = 1.45
    wheelbase: float = 0.7
    max_angular_rate: float = 1.4
    goal_tolerance: float = 0.55
    max_steps: int = 1600


@dataclass(frozen=True)
class TrackingState:
    t: float
    x: float
    y: float
    yaw: float
    target_index: int
    cross_track_error: float
    risk: float
    stability: float
    feasible: bool


@dataclass(frozen=True)
class TrackingResult:
    states: list[TrackingState]
    success: bool
    duration_s: float
    mean_error_m: float
    max_error_m: float
    final_error_m: float
    mean_risk: float
    max_risk: float
    min_stability: float
    feasible_rate: float


def simulate_tracking(
    terrain_map: ImplicitTerrainMap,
    path: list[tuple[float, float]],
    config: TrackerConfig | None = None,
) -> TrackingResult:
    config = config or TrackerConfig()
    if len(path) < 2:
        return empty_tracking_result()

    reference = resample_polyline(path, spacing=terrain_map.resolution)
    x, y = reference[0]
    yaw = math.atan2(reference[1][1] - reference[0][1], reference[1][0] - reference[0][0])
    states: list[TrackingState] = []

    for step in range(config.max_steps):
        nearest_index = nearest_path_index(reference, (x, y))
        target_index = lookahead_index(reference, nearest_index, config.lookahead)
        target = reference[target_index]
        alpha = normalize_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
        angular_rate = np.clip(2.0 * config.speed * math.sin(alpha) / config.wheelbase, -config.max_angular_rate, config.max_angular_rate)

        x += config.speed * math.cos(yaw) * config.dt
        y += config.speed * math.sin(yaw) * config.dt
        yaw = normalize_angle(yaw + float(angular_rate) * config.dt)

        query = terrain_map.query((x, y))
        config_stability = estimate_configuration_stability_on_map(terrain_map, (x, y), yaw)
        error = distance_to_polyline(reference, (x, y), nearest_index)
        states.append(
            TrackingState(
                t=step * config.dt,
                x=x,
                y=y,
                yaw=yaw,
                target_index=target_index,
                cross_track_error=error,
                risk=query.risk,
                stability=config_stability.stability,
                feasible=config_stability.feasible and not query.obstacle,
            )
        )

        if math.dist((x, y), reference[-1]) <= config.goal_tolerance:
            break

    if not states:
        return empty_tracking_result()

    errors = np.array([state.cross_track_error for state in states], dtype=np.float64)
    risks = np.array([state.risk for state in states], dtype=np.float64)
    stabilities = np.array([state.stability for state in states], dtype=np.float64)
    feasible = np.array([state.feasible for state in states], dtype=bool)
    final_error = math.dist((states[-1].x, states[-1].y), reference[-1])
    success = (
        final_error <= config.goal_tolerance
        and float(np.mean(errors)) <= 0.75
        and float(np.max(errors)) <= 1.8
        and float(np.nanmax(risks)) <= 0.95
        and float(np.nanmin(stabilities)) >= 0.08
        and float(np.mean(feasible)) >= 0.75
    )

    return TrackingResult(
        states=states,
        success=success,
        duration_s=states[-1].t,
        mean_error_m=float(np.mean(errors)),
        max_error_m=float(np.max(errors)),
        final_error_m=final_error,
        mean_risk=float(np.nanmean(risks)),
        max_risk=float(np.nanmax(risks)),
        min_stability=float(np.nanmin(stabilities)),
        feasible_rate=float(np.mean(feasible)),
    )


def nearest_path_index(path: list[tuple[float, float]], point: tuple[float, float]) -> int:
    distances = [math.dist(point, waypoint) for waypoint in path]
    return int(np.argmin(distances))


def lookahead_index(path: list[tuple[float, float]], start_index: int, lookahead: float) -> int:
    distance = 0.0
    for index in range(start_index, len(path) - 1):
        distance += math.dist(path[index], path[index + 1])
        if distance >= lookahead:
            return index + 1
    return len(path) - 1


def distance_to_polyline(path: list[tuple[float, float]], point: tuple[float, float], nearest_index: int) -> float:
    best = math.dist(point, path[nearest_index])
    start = max(0, nearest_index - 2)
    end = min(len(path) - 1, nearest_index + 3)
    px, py = point
    for index in range(start, end):
        ax, ay = path[index]
        bx, by = path[index + 1]
        vx, vy = bx - ax, by - ay
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
        projection = (ax + t * vx, ay + t * vy)
        best = min(best, math.dist(point, projection))
    return best


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def empty_tracking_result() -> TrackingResult:
    return TrackingResult(
        states=[],
        success=False,
        duration_s=0.0,
        mean_error_m=float("inf"),
        max_error_m=float("inf"),
        final_error_m=float("inf"),
        mean_risk=float("inf"),
        max_risk=float("inf"),
        min_stability=0.0,
        feasible_rate=0.0,
    )
