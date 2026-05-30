from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap
from traversability.tracked_stability import (
    body_checkpoints,
    convex_hull,
    evaluate_plane,
    fit_plane,
    point_in_convex_polygon,
    polygon_area,
    query_local_heights,
    rotate_plane_toward_support,
)


@dataclass(frozen=True)
class WheeledRobotModel:
    body_length: float = 0.74
    body_width: float = 0.48
    wheelbase: float = 0.52
    track_width: float = 0.40
    wheel_radius: float = 0.13
    wheel_width: float = 0.08
    body_clearance: float = 0.16
    com_height: float = 0.25
    contact_angle_limit: float = math.radians(55.0)
    contact_tolerance: float = 0.035
    sample_spacing: float = 0.04
    max_iterations: int = 7
    rotation_step_rad: float = math.radians(3.0)


@dataclass(frozen=True)
class WheeledStabilityResult:
    roll: float
    pitch: float
    wheel_contact_count: int
    support_area: float
    com_inside_support: bool
    body_collision: bool
    feasible: bool
    stability: float
    support_polygon: list[tuple[float, float]]


def estimate_wheeled_configuration_stability(
    terrain_map: ImplicitTerrainMap,
    xy: tuple[float, float],
    yaw: float,
    model: WheeledRobotModel | None = None,
) -> WheeledStabilityResult:
    model = model or WheeledRobotModel()
    local_points, compensation = wheel_contact_checkpoints(model)
    terrain_heights = query_local_heights(terrain_map, xy, yaw, local_points)
    contact_heights = terrain_heights + compensation
    plane = fit_plane(local_points, contact_heights)
    best_plane = plane
    best_contacts = np.empty((0, 2), dtype=np.float64)
    best_polygon: list[tuple[float, float]] = []
    best_area = 0.0
    stable = False

    for _ in range(model.max_iterations):
        predicted = evaluate_plane(best_plane, local_points)
        contacts = wheel_contact_points(local_points, contact_heights, predicted, model.contact_tolerance)
        wheel_contacts = representative_wheel_contacts(contacts, model)
        polygon = convex_hull(wheel_contacts)
        area = polygon_area(polygon)
        if area > best_area:
            best_area = area
            best_contacts = wheel_contacts
            best_polygon = polygon
            plane = best_plane
        if len(wheel_contacts) >= 3 and point_in_convex_polygon((0.0, 0.0), polygon):
            stable = True
            break
        best_plane = rotate_plane_toward_support(best_plane, wheel_contacts, model.rotation_step_rad)

    roll = math.atan(plane[1])
    pitch = math.atan(plane[0])
    body_collision = check_wheeled_body_collision(terrain_map, xy, yaw, plane, model)
    feasible = stable and not body_collision
    stability = wheeled_stability_score(best_area, best_polygon, body_collision, model)
    return WheeledStabilityResult(
        roll=roll,
        pitch=pitch,
        wheel_contact_count=count_contacted_wheels(best_contacts, model),
        support_area=best_area,
        com_inside_support=stable,
        body_collision=body_collision,
        feasible=feasible,
        stability=stability,
        support_polygon=[tuple(point) for point in best_polygon],
    )


def wheel_contact_checkpoints(model: WheeledRobotModel) -> tuple[np.ndarray, np.ndarray]:
    centers = wheel_centers(model)
    lateral = np.arange(-model.wheel_width / 2.0, model.wheel_width / 2.0 + 1e-9, model.sample_spacing)
    longitudinal = np.arange(-model.wheel_radius * 0.65, model.wheel_radius * 0.65 + 1e-9, model.sample_spacing)
    points = []
    compensation = []
    for cx, cy in centers:
        for dx in longitudinal:
            if abs(dx) > model.wheel_radius * math.sin(model.contact_angle_limit):
                continue
            arc_compensation = model.wheel_radius - math.sqrt(max(model.wheel_radius**2 - dx**2, 0.0))
            for dy in lateral:
                points.append((cx + dx, cy + dy))
                compensation.append(arc_compensation)
    return np.asarray(points, dtype=np.float64), np.asarray(compensation, dtype=np.float64)


def wheel_centers(model: WheeledRobotModel) -> np.ndarray:
    xs = np.array([-model.wheelbase / 2.0, model.wheelbase / 2.0])
    ys = np.array([-model.track_width / 2.0, model.track_width / 2.0])
    return np.asarray([(x, y) for x in xs for y in ys], dtype=np.float64)


def wheel_contact_points(
    local_xy: np.ndarray,
    contact_heights: np.ndarray,
    predicted: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    residual = contact_heights - predicted
    max_residual = float(np.max(residual))
    return local_xy[residual >= max_residual - tolerance]


def representative_wheel_contacts(points: np.ndarray, model: WheeledRobotModel) -> np.ndarray:
    if len(points) == 0:
        return points
    representatives = []
    centers = wheel_centers(model)
    for center in centers:
        distances = np.linalg.norm(points - center, axis=1)
        mask = distances <= max(model.wheel_radius, model.wheel_width)
        if np.any(mask):
            representatives.append(np.mean(points[mask], axis=0))
    return np.asarray(representatives, dtype=np.float64) if representatives else np.empty((0, 2), dtype=np.float64)


def count_contacted_wheels(points: np.ndarray, model: WheeledRobotModel) -> int:
    if len(points) == 0:
        return 0
    centers = wheel_centers(model)
    count = 0
    for center in centers:
        if np.min(np.linalg.norm(points - center, axis=1)) <= max(model.wheel_radius, model.wheel_width):
            count += 1
    return count


def check_wheeled_body_collision(
    terrain_map: ImplicitTerrainMap,
    xy: tuple[float, float],
    yaw: float,
    plane: np.ndarray,
    model: WheeledRobotModel,
) -> bool:
    body_model = type(
        "BodyModel",
        (),
        {
            "body_length": model.body_length,
            "body_width": model.body_width,
            "sample_spacing": model.sample_spacing * 2.0,
        },
    )()
    local = body_checkpoints(body_model)
    heights = query_local_heights(terrain_map, xy, yaw, local)
    body_plane = evaluate_plane(plane, local) + model.body_clearance
    return bool(np.any(heights > body_plane))


def wheeled_stability_score(
    support_area: float,
    polygon: list[tuple[float, float]],
    body_collision: bool,
    model: WheeledRobotModel,
) -> float:
    nominal_area = model.wheelbase * model.track_width
    area_score = min(support_area / max(nominal_area, 1e-9), 1.0)
    zmp_score = 1.0 if point_in_convex_polygon((0.0, 0.0), polygon) else 0.0
    collision_score = 0.0 if body_collision else 1.0
    return float(np.clip(0.45 * area_score + 0.40 * zmp_score + 0.15 * collision_score, 0.0, 1.0))
