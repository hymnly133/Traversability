from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from traversability.stability import RobotFootprint, estimate_configuration_stability
from traversability.terrain import TerrainLayer, TerrainPyramid, grid_to_world, world_to_grid


@dataclass(frozen=True)
class PlannerWeights:
    slope: float = 4.2
    roughness: float = 3.6
    step: float = 3.8
    obstacle_margin: float = 4.0
    stability: float = 5.0
    corridor_radius_cells: int = 9
    lethal_risk: float = 0.90


@dataclass(frozen=True)
class StabilitySample:
    x: float
    y: float
    yaw: float
    slope: float
    roughness: float
    step: float
    roll: float
    pitch: float
    support_std: float
    min_clearance: float
    stability: float
    feasible: bool
    risk: float


@dataclass(frozen=True)
class PlanningResult:
    path_xy: list[tuple[float, float]]
    coarse_path_xy: list[tuple[float, float]]
    samples: list[StabilitySample]
    runtime_ms: float
    expanded_nodes: int
    path_length_m: float
    mean_risk: float
    max_risk: float
    min_stability: float
    feasible_rate: float
    success: bool


def plan_multilevel(
    pyramid: TerrainPyramid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    weights: PlannerWeights | None = None,
) -> PlanningResult:
    weights = weights or PlannerWeights()
    begin = time.perf_counter()
    expanded_total = 0

    coarse_layer = pyramid.coarsest
    coarse_start = nearest_free(coarse_layer, world_to_grid(coarse_layer, start_xy), allowed_mask=None)
    coarse_goal = nearest_free(coarse_layer, world_to_grid(coarse_layer, goal_xy), allowed_mask=None)
    if coarse_start is None or coarse_goal is None:
        return empty_result(begin)
    coarse_cells, expanded = astar(coarse_layer, coarse_start, coarse_goal, weights)
    expanded_total += expanded
    if not coarse_cells:
        return empty_result(begin)

    guide_cells = coarse_cells
    for layer in reversed(pyramid.layers[:-1]):
        corridor = build_corridor(layer, guide_cells, coarse_layer, weights.corridor_radius_cells)
        cells, expanded = astar(
            layer,
            nearest_free(layer, world_to_grid(layer, start_xy), corridor) or world_to_grid(layer, start_xy),
            nearest_free(layer, world_to_grid(layer, goal_xy), corridor) or world_to_grid(layer, goal_xy),
            weights,
            allowed_mask=corridor,
        )
        expanded_total += expanded
        if not cells:
            fine_start = nearest_free(layer, world_to_grid(layer, start_xy), allowed_mask=None)
            fine_goal = nearest_free(layer, world_to_grid(layer, goal_xy), allowed_mask=None)
            if fine_start is None or fine_goal is None:
                return empty_result(begin)
            cells, expanded = astar(layer, fine_start, fine_goal, weights)
            expanded_total += expanded
        guide_cells = cells
        coarse_layer = layer

    fine_layer = pyramid.finest
    raw_path = [grid_to_world(fine_layer, cell) for cell in guide_cells]
    smooth_path = shortcut_path(fine_layer, raw_path, max_risk=0.92)
    samples = evaluate_stability(fine_layer, smooth_path)
    runtime_ms = (time.perf_counter() - begin) * 1000.0
    risks = np.array([sample.risk for sample in samples], dtype=np.float64)
    stabilities = np.array([sample.stability for sample in samples], dtype=np.float64)
    feasible_rate = float(np.mean([sample.feasible for sample in samples]))

    return PlanningResult(
        path_xy=smooth_path,
        coarse_path_xy=[grid_to_world(pyramid.coarsest, cell) for cell in coarse_cells],
        samples=samples,
        runtime_ms=runtime_ms,
        expanded_nodes=expanded_total,
        path_length_m=polyline_length(smooth_path),
        mean_risk=float(np.mean(risks)),
        max_risk=float(np.max(risks)),
        min_stability=float(np.min(stabilities)),
        feasible_rate=feasible_rate,
        success=bool(
            np.all(np.isfinite(risks))
            and np.max(risks) <= weights.lethal_risk
            and np.min(stabilities) > 0.08
            and feasible_rate >= 0.80
        ),
    )


def astar(
    layer: TerrainLayer,
    start: tuple[int, int],
    goal: tuple[int, int],
    weights: PlannerWeights,
    allowed_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], int]:
    if not is_free(layer, start, allowed_mask) or not is_free(layer, goal, allowed_mask):
        return [], 0

    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
    expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        expanded += 1
        if current == goal:
            break

        for neighbor, distance in neighbors(layer, current):
            if not is_free(layer, neighbor, allowed_mask):
                continue
            terrain_cost = cell_cost(layer, neighbor, weights)
            new_cost = cost_so_far[current] + distance * layer.resolution * terrain_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal) * layer.resolution
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return [], expanded

    path = [goal]
    while path[-1] != start:
        parent = came_from[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return path, expanded


def nearest_free(
    layer: TerrainLayer,
    cell: tuple[int, int],
    allowed_mask: np.ndarray | None,
    max_radius: int = 12,
) -> tuple[int, int] | None:
    if is_free(layer, cell, allowed_mask):
        return cell
    row, col = cell
    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for radius in range(1, max_radius + 1):
        r0 = max(0, row - radius)
        r1 = min(layer.height.shape[0] - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(layer.height.shape[1] - 1, col + radius)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if rr not in (r0, r1) and cc not in (c0, c1):
                    continue
                candidate = (rr, cc)
                if not is_free(layer, candidate, allowed_mask):
                    continue
                distance = heuristic(cell, candidate)
                if distance < best_distance:
                    best = candidate
                    best_distance = distance
        if best is not None:
            return best
    return None


def cell_cost(layer: TerrainLayer, cell: tuple[int, int], weights: PlannerWeights) -> float:
    row, col = cell
    if layer.risk[row, col] >= weights.lethal_risk:
        return 1_000.0
    slope = min(layer.slope[row, col] / 1.2, 1.0)
    roughness = min(layer.roughness[row, col] / 0.35, 1.0)
    step = min(layer.step[row, col] / 0.65, 1.0)
    margin = obstacle_margin_cost(layer, cell)
    stability_cost = footprint_instability_cost(layer, cell)
    risk_barrier = 1.0 / max(0.08, weights.lethal_risk - float(layer.risk[row, col]))
    return (
        1.0
        + weights.slope * slope
        + weights.roughness * roughness
        + weights.step * step
        + weights.obstacle_margin * margin
        + weights.stability * stability_cost
        + risk_barrier
    )


def footprint_instability_cost(layer: TerrainLayer, cell: tuple[int, int]) -> float:
    row, col = cell
    radius = max(1, int(round(0.35 / layer.resolution)))
    r0 = max(0, row - radius)
    r1 = min(layer.height.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(layer.height.shape[1], col + radius + 1)
    patch = layer.height[r0:r1, c0:c1]
    if np.any(layer.obstacle[r0:r1, c0:c1]):
        return 1.0
    relief = float(np.max(patch) - np.min(patch))
    support_std = float(np.std(patch))
    return min(1.0, 0.55 * relief / 0.58 + 0.45 * support_std / 0.22)


def obstacle_margin_cost(layer: TerrainLayer, cell: tuple[int, int], radius: int = 3) -> float:
    row, col = cell
    r0 = max(0, row - radius)
    r1 = min(layer.obstacle.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(layer.obstacle.shape[1], col + radius + 1)
    window = layer.obstacle[r0:r1, c0:c1]
    if not np.any(window):
        return 0.0
    obstacle_locations = np.argwhere(window)
    local = np.array([row - r0, col - c0])
    distances = np.linalg.norm(obstacle_locations - local, axis=1)
    return float(max(0.0, 1.0 - np.min(distances) / max(radius, 1)))


def neighbors(layer: TerrainLayer, cell: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    row, col = cell
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < layer.height.shape[0] and 0 <= nc < layer.height.shape[1]:
                result.append(((nr, nc), math.sqrt(2.0) if dr and dc else 1.0))
    return result


def is_free(layer: TerrainLayer, cell: tuple[int, int], allowed_mask: np.ndarray | None) -> bool:
    row, col = cell
    if layer.obstacle[row, col] or not np.isfinite(layer.risk[row, col]):
        return False
    if allowed_mask is not None and not allowed_mask[row, col]:
        return False
    return True


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_corridor(
    layer: TerrainLayer,
    guide_cells: list[tuple[int, int]],
    guide_layer: TerrainLayer,
    radius: int,
) -> np.ndarray:
    mask = np.zeros(layer.height.shape, dtype=bool)
    for guide_cell in guide_cells:
        xy = grid_to_world(guide_layer, guide_cell)
        row, col = world_to_grid(layer, xy)
        r0 = max(0, row - radius)
        r1 = min(layer.height.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(layer.height.shape[1], col + radius + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def shortcut_path(
    layer: TerrainLayer,
    path: list[tuple[float, float]],
    max_risk: float,
    feasibility_grid: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    result = [path[0]]
    anchor = 0
    max_lookahead = 18
    while anchor < len(path) - 1:
        next_index = min(len(path) - 1, anchor + max_lookahead)
        while next_index > anchor + 1:
            if line_is_safe(layer, path[anchor], path[next_index], max_risk, feasibility_grid):
                break
            next_index -= 1
        result.append(path[next_index])
        anchor = next_index
    return result


def line_is_safe(
    layer: TerrainLayer,
    start: tuple[float, float],
    goal: tuple[float, float],
    max_risk: float,
    feasibility_grid: np.ndarray | None = None,
) -> bool:
    distance = math.dist(start, goal)
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    steps = max(2, int(distance / (layer.resolution * 0.5)))
    for idx in range(steps + 1):
        t = idx / steps
        xy = (start[0] + (goal[0] - start[0]) * t, start[1] + (goal[1] - start[1]) * t)
        row, col = world_to_grid(layer, xy)
        if layer.obstacle[row, col] or layer.risk[row, col] >= max_risk:
            return False
        if feasibility_grid is not None and not feasibility_grid[row, col]:
            return False
        if idx % 3 == 0:
            config = estimate_configuration_stability(layer, xy, yaw)
            if not config.feasible or config.stability < 0.18:
                return False
    return True


def evaluate_stability(layer: TerrainLayer, path: list[tuple[float, float]]) -> list[StabilitySample]:
    samples: list[StabilitySample] = []
    footprint = RobotFootprint()
    resampled = resample_polyline_with_yaw(path, spacing=layer.resolution * 1.5)
    for xy, yaw in resampled:
        row, col = world_to_grid(layer, xy)
        slope = float(layer.slope[row, col])
        roughness = float(layer.roughness[row, col])
        step = float(layer.step[row, col])
        risk = float(layer.risk[row, col])
        config = estimate_configuration_stability(layer, xy, yaw, footprint)
        terrain_stability = 1.0 - (
            0.44 * min(slope / 1.2, 1.0)
            + 0.28 * min(roughness / 0.35, 1.0)
            + 0.18 * min(step / 0.65, 1.0)
            + 0.10 * (0.0 if config.feasible else 1.0)
        )
        stability = 0.55 * terrain_stability + 0.45 * config.stability
        samples.append(
            StabilitySample(
                x=xy[0],
                y=xy[1],
                yaw=yaw,
                slope=slope,
                roughness=roughness,
                step=step,
                roll=config.roll,
                pitch=config.pitch,
                support_std=config.support_std,
                min_clearance=config.min_clearance,
                stability=float(np.clip(stability, 0.0, 1.0)),
                feasible=config.feasible,
                risk=risk,
            )
        )
    return samples


def resample_polyline_with_yaw(
    path: list[tuple[float, float]], spacing: float
) -> list[tuple[tuple[float, float], float]]:
    if len(path) < 2:
        return [((path[0][0], path[0][1]), 0.0)] if path else []
    result = []
    for start, end in zip(path[:-1], path[1:]):
        distance = math.dist(start, end)
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        steps = max(1, math.ceil(distance / spacing))
        for idx in range(steps):
            t = idx / steps
            result.append(((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t), yaw))
    last = path[-1]
    previous = path[-2]
    result.append((last, math.atan2(last[1] - previous[1], last[0] - previous[0])))
    return result


def resample_polyline(path: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    if len(path) < 2:
        return path
    result = []
    for start, end in zip(path[:-1], path[1:]):
        distance = math.dist(start, end)
        steps = max(1, math.ceil(distance / spacing))
        for idx in range(steps):
            t = idx / steps
            result.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    result.append(path[-1])
    return result


def polyline_length(path: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def empty_result(begin: float) -> PlanningResult:
    return PlanningResult(
        path_xy=[],
        coarse_path_xy=[],
        samples=[],
        runtime_ms=(time.perf_counter() - begin) * 1000.0,
        expanded_nodes=0,
        path_length_m=0.0,
        mean_risk=float("inf"),
        max_risk=float("inf"),
        min_stability=0.0,
        feasible_rate=0.0,
        success=False,
    )
