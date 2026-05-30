from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap


@dataclass(frozen=True)
class TrackedRobotModel:
    body_length: float = 0.82
    body_width: float = 0.42
    main_track_length: float = 0.62
    track_width: float = 0.14
    track_separation: float = 0.34
    flipper_length: float = 0.28
    flipper_width: float = 0.12
    body_clearance: float = 0.18
    com_height: float = 0.32
    contact_tolerance: float = 0.045
    sample_spacing: float = 0.08
    max_iterations: int = 8
    rotation_step_rad: float = math.radians(4.0)


@dataclass(frozen=True)
class TrackedStabilityResult:
    roll: float
    pitch: float
    front_flipper_angle: float
    rear_flipper_angle: float
    main_contact_count: int
    support_area: float
    com_inside_support: bool
    stable_without_flippers: bool
    stable_with_flippers: bool
    body_collision: bool
    feasible: bool
    stability: float
    support_polygon: list[tuple[float, float]]


def estimate_tracked_configuration_stability(
    terrain_map: ImplicitTerrainMap,
    xy: tuple[float, float],
    yaw: float,
    model: TrackedRobotModel | None = None,
) -> TrackedStabilityResult:
    model = model or TrackedRobotModel()
    local_track_points = main_track_checkpoints(model)
    heights = query_local_heights(terrain_map, xy, yaw, local_track_points)
    plane = fit_plane(local_track_points, heights)
    pitch = math.atan(plane[0])
    roll = math.atan(plane[1])

    best_polygon: list[tuple[float, float]] = []
    best_area = 0.0
    best_plane = plane
    best_contacts = np.empty((0, 2), dtype=np.float64)
    stable_without_flippers = False

    for _ in range(model.max_iterations):
        predicted = evaluate_plane(best_plane, local_track_points)
        contacts = contact_points(local_track_points, heights, predicted, model.contact_tolerance)
        polygon = convex_hull(contacts)
        area = polygon_area(polygon)
        if area > best_area:
            best_area = area
            best_polygon = polygon
            best_contacts = contacts
            best_plane = plane
        if len(polygon) >= 3 and point_in_convex_polygon((0.0, 0.0), polygon):
            stable_without_flippers = True
            break
        plane = rotate_plane_toward_support(plane, contacts, model.rotation_step_rad)
        pitch = math.atan(plane[0])
        roll = math.atan(plane[1])

    front_angle, rear_angle, flipper_contacts = estimate_flipper_support(
        terrain_map,
        xy,
        yaw,
        best_plane,
        model,
    )
    combined_contacts = np.vstack([best_contacts, flipper_contacts]) if len(flipper_contacts) else best_contacts
    combined_polygon = convex_hull(combined_contacts)
    combined_area = polygon_area(combined_polygon)
    stable_with_flippers = len(combined_polygon) >= 3 and point_in_convex_polygon((0.0, 0.0), combined_polygon)

    body_collision = check_body_collision(terrain_map, xy, yaw, best_plane, heights, model)
    support_area = max(best_area, combined_area)
    feasible = (stable_without_flippers or stable_with_flippers) and not body_collision
    stability = stability_score(support_area, combined_polygon or best_polygon, body_collision, model)

    return TrackedStabilityResult(
        roll=roll,
        pitch=pitch,
        front_flipper_angle=front_angle,
        rear_flipper_angle=rear_angle,
        main_contact_count=len(best_contacts),
        support_area=support_area,
        com_inside_support=stable_without_flippers or stable_with_flippers,
        stable_without_flippers=stable_without_flippers,
        stable_with_flippers=stable_with_flippers,
        body_collision=body_collision,
        feasible=feasible,
        stability=stability,
        support_polygon=[tuple(point) for point in (combined_polygon or best_polygon)],
    )


def main_track_checkpoints(model: TrackedRobotModel) -> np.ndarray:
    xs = np.arange(-model.main_track_length / 2.0, model.main_track_length / 2.0 + 1e-9, model.sample_spacing)
    ys = np.array([-model.track_separation / 2.0, model.track_separation / 2.0])
    offsets = np.linspace(-model.track_width / 2.0, model.track_width / 2.0, 3)
    points = []
    for y in ys:
        for offset in offsets:
            for x in xs:
                points.append((x, y + offset))
    return np.asarray(points, dtype=np.float64)


def flipper_checkpoints(model: TrackedRobotModel, front: bool) -> np.ndarray:
    root_x = model.main_track_length / 2.0 if front else -model.main_track_length / 2.0
    direction = 1.0 if front else -1.0
    xs = root_x + direction * np.arange(model.sample_spacing, model.flipper_length + 1e-9, model.sample_spacing)
    ys = np.array([-model.track_separation / 2.0, model.track_separation / 2.0])
    offsets = np.linspace(-model.flipper_width / 2.0, model.flipper_width / 2.0, 3)
    points = []
    for y in ys:
        for offset in offsets:
            for x in xs:
                points.append((x, y + offset))
    return np.asarray(points, dtype=np.float64)


def body_checkpoints(model: TrackedRobotModel) -> np.ndarray:
    xs = np.arange(-model.body_length / 2.0, model.body_length / 2.0 + 1e-9, model.sample_spacing)
    ys = np.arange(-model.body_width / 2.0, model.body_width / 2.0 + 1e-9, model.sample_spacing)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def query_local_heights(
    terrain_map: ImplicitTerrainMap,
    xy: tuple[float, float],
    yaw: float,
    local_points: np.ndarray,
) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    heights = []
    for px, py in local_points:
        wx = xy[0] + px * cos_yaw - py * sin_yaw
        wy = xy[1] + px * sin_yaw + py * cos_yaw
        heights.append(terrain_map.query((wx, wy)).height)
    return np.asarray(heights, dtype=np.float64)


def fit_plane(local_xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    design = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.ones(local_xy.shape[0])])
    plane, *_ = np.linalg.lstsq(design, z, rcond=None)
    return plane


def evaluate_plane(plane: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    return plane[0] * local_xy[:, 0] + plane[1] * local_xy[:, 1] + plane[2]


def contact_points(
    local_xy: np.ndarray,
    heights: np.ndarray,
    predicted: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    residual = heights - predicted
    max_residual = float(np.max(residual))
    mask = residual >= max_residual - tolerance
    return local_xy[mask]


def rotate_plane_toward_support(plane: np.ndarray, contacts: np.ndarray, step: float) -> np.ndarray:
    if len(contacts) == 0:
        return plane
    center = np.mean(contacts, axis=0)
    new_plane = plane.copy()
    new_plane[0] += np.clip(-center[0], -1.0, 1.0) * math.tan(step)
    new_plane[1] += np.clip(-center[1], -1.0, 1.0) * math.tan(step)
    return new_plane


def estimate_flipper_support(
    terrain_map: ImplicitTerrainMap,
    xy: tuple[float, float],
    yaw: float,
    plane: np.ndarray,
    model: TrackedRobotModel,
) -> tuple[float, float, np.ndarray]:
    contacts = []
    angles = []
    for front in (True, False):
        local = flipper_checkpoints(model, front)
        heights = query_local_heights(terrain_map, xy, yaw, local)
        predicted = evaluate_plane(plane, local)
        residual = heights - predicted
        if front:
            run = np.maximum(local[:, 0] - model.main_track_length / 2.0, model.sample_spacing)
        else:
            run = np.maximum(-local[:, 0] - model.main_track_length / 2.0, model.sample_spacing)
        candidate_angles = np.arctan2(residual, run)
        chosen = float(candidate_angles[np.argmax(np.abs(candidate_angles))])
        angles.append(chosen)
        mask = np.abs(candidate_angles - chosen) <= math.radians(7.0)
        support = local[mask & (np.abs(candidate_angles) > math.radians(3.0))]
        if len(support):
            contacts.append(support)
    flipper_contacts = np.vstack(contacts) if contacts else np.empty((0, 2), dtype=np.float64)
    return angles[0], angles[1], flipper_contacts


def check_body_collision(
    terrain_map: ImplicitTerrainMap,
    xy: tuple[float, float],
    yaw: float,
    plane: np.ndarray,
    track_heights: np.ndarray,
    model: TrackedRobotModel,
) -> bool:
    local = body_checkpoints(model)
    heights = query_local_heights(terrain_map, xy, yaw, local)
    body_plane = evaluate_plane(plane, local) + model.body_clearance
    max_track_height = float(np.max(track_heights))
    return bool(np.any(heights > body_plane) or float(np.max(heights) - max_track_height) > model.body_clearance)


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        area += a[0] * b[1] - b[0] * a[1]
    return abs(area) * 0.5


def point_in_convex_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    signs = []
    px, py = point
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        signs.append(cross)
    return all(value >= -1e-9 for value in signs) or all(value <= 1e-9 for value in signs)


def stability_score(
    support_area: float,
    polygon: list[tuple[float, float]],
    body_collision: bool,
    model: TrackedRobotModel,
) -> float:
    nominal_area = model.main_track_length * (model.track_separation + model.track_width)
    area_score = min(support_area / max(nominal_area, 1e-9), 1.0)
    zmp_score = 1.0 if point_in_convex_polygon((0.0, 0.0), polygon) else 0.0
    collision_score = 0.0 if body_collision else 1.0
    return float(np.clip(0.50 * area_score + 0.35 * zmp_score + 0.15 * collision_score, 0.0, 1.0))
