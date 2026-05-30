from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap
from traversability.ndt_map import NDTImplicitMap
from traversability.planner import polyline_length, resample_polyline
from traversability.tracked_stability import estimate_tracked_configuration_stability


@dataclass(frozen=True)
class HybridLocalPlannerConfig:
    step_length: float = 0.32
    heading_bins: int = 32
    max_steer: float = math.radians(32.0)
    wheelbase: float = 0.62
    local_window_radius: float = 3.0
    goal_tolerance: float = 0.35
    max_iterations: int = 9000
    risk_weight: float = 1.4
    stability_weight: float = 1.1
    steering_weight: float = 0.12
    max_risk: float = 0.92
    min_stability: float = 0.30
    global_waypoint_limit: int = 24
    global_traversability_radius_cells: int = 1


@dataclass(frozen=True)
class NDTLocalTraversabilityGuide:
    """Local query facade for traversability shared from the global NDT layer."""

    ndt_map: NDTImplicitMap
    traversable_keys: frozenset[tuple[int, int, int]]

    @classmethod
    def from_ndt_map(cls, ndt_map: NDTImplicitMap) -> "NDTLocalTraversabilityGuide":
        metrics = ndt_map.compute_metrics()
        traversable = frozenset(key for key, metric in metrics.items() if np.isfinite(metric.traversal_cost))
        return cls(ndt_map=ndt_map, traversable_keys=traversable)

    def is_traversable(self, xy: tuple[float, float], z: float, radius_cells: int = 1) -> bool:
        key = self.key_from_xyz((xy[0], xy[1], z))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    if (key[0] + dx, key[1] + dy, key[2] + dz) in self.traversable_keys:
                        return True
        return False

    def key_from_xyz(self, xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        index = np.floor((np.asarray(xyz, dtype=np.float64) - self.ndt_map.origin) / self.ndt_map.config.voxel_size)
        return int(index[0]), int(index[1]), int(index[2])


@dataclass(frozen=True)
class HybridState:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class HybridLocalPlanningResult:
    path: list[tuple[float, float, float]]
    runtime_ms: float
    expanded_nodes: int
    path_length_m: float
    mean_risk: float
    min_stability: float
    success: bool


def plan_hybrid_local(
    terrain_map: ImplicitTerrainMap,
    start: tuple[float, float, float],
    global_path: list[tuple[float, float]],
    config: HybridLocalPlannerConfig | None = None,
    global_traversability: NDTLocalTraversabilityGuide | None = None,
) -> HybridLocalPlanningResult:
    config = config or HybridLocalPlannerConfig()
    begin = time.perf_counter()
    if len(global_path) < 2:
        return empty_result(begin)

    local_goal = select_local_goal(start[:2], global_path, config)
    dense_global = resample_polyline(global_path[: config.global_waypoint_limit + 1], terrain_map.resolution)
    start_state = HybridState(start[0], start[1], normalize_angle(start[2]))
    start_key = state_key(start_state, terrain_map.resolution, config.heading_bins)
    frontier: list[tuple[float, tuple[int, int, int]]] = [(0.0, start_key)]
    states = {start_key: start_state}
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    cost_so_far: dict[tuple[int, int, int], float] = {start_key: 0.0}
    expanded = 0
    best_key = start_key
    best_goal_distance = math.dist(start[:2], local_goal)

    while frontier and expanded < config.max_iterations:
        _, current_key = heapq.heappop(frontier)
        current = states[current_key]
        expanded += 1
        goal_distance = math.dist((current.x, current.y), local_goal)
        if goal_distance < best_goal_distance:
            best_goal_distance = goal_distance
            best_key = current_key
        if goal_distance <= config.goal_tolerance:
            return build_result(begin, terrain_map, states, came_from, current_key, expanded)

        for steer in (-config.max_steer, 0.0, config.max_steer):
            successor = propagate(current, steer, config)
            if not inside_local_window(start[:2], (successor.x, successor.y), config.local_window_radius):
                continue
            query = terrain_map.query((successor.x, successor.y))
            if query.obstacle or query.risk > config.max_risk:
                continue
            if global_traversability is not None and not global_traversability.is_traversable(
                (successor.x, successor.y),
                query.height,
                config.global_traversability_radius_cells,
            ):
                continue
            stability = estimate_tracked_configuration_stability(terrain_map, (successor.x, successor.y), successor.yaw)
            if not stability.feasible or stability.stability < config.min_stability:
                continue
            successor_key = state_key(successor, terrain_map.resolution, config.heading_bins)
            step_cost = (
                config.step_length
                + config.risk_weight * min(query.risk, 1.0)
                + config.stability_weight * (1.0 - stability.stability)
                + config.steering_weight * abs(steer) / max(config.max_steer, 1e-9)
            )
            new_cost = cost_so_far[current_key] + step_cost
            if successor_key not in cost_so_far or new_cost < cost_so_far[successor_key]:
                cost_so_far[successor_key] = new_cost
                states[successor_key] = successor
                heuristic = global_path_heuristic((successor.x, successor.y), dense_global, local_goal)
                heapq.heappush(frontier, (new_cost + heuristic, successor_key))
                came_from[successor_key] = current_key

    if best_goal_distance <= config.goal_tolerance * 1.6:
        return build_result(begin, terrain_map, states, came_from, best_key, expanded)
    return empty_result(begin, expanded)


def select_local_goal(
    start_xy: tuple[float, float],
    global_path: list[tuple[float, float]],
    config: HybridLocalPlannerConfig,
) -> tuple[float, float]:
    nearest = min(range(len(global_path)), key=lambda idx: math.dist(start_xy, global_path[idx]))
    end = min(len(global_path) - 1, nearest + config.global_waypoint_limit)
    for idx in range(nearest + 1, end + 1):
        if math.dist(start_xy, global_path[idx]) >= config.local_window_radius * 0.78:
            return global_path[idx]
    return global_path[end]


def propagate(state: HybridState, steer: float, config: HybridLocalPlannerConfig) -> HybridState:
    if abs(steer) <= 1e-9:
        yaw = state.yaw
        x = state.x + config.step_length * math.cos(yaw)
        y = state.y + config.step_length * math.sin(yaw)
        return HybridState(x, y, yaw)
    yaw_rate = math.tan(steer) / config.wheelbase
    delta_yaw = config.step_length * yaw_rate
    radius = 1.0 / yaw_rate
    next_yaw = normalize_angle(state.yaw + delta_yaw)
    x = state.x + radius * (math.sin(next_yaw) - math.sin(state.yaw))
    y = state.y - radius * (math.cos(next_yaw) - math.cos(state.yaw))
    return HybridState(x, y, next_yaw)


def state_key(state: HybridState, resolution: float, heading_bins: int) -> tuple[int, int, int]:
    ix = int(round(state.x / resolution))
    iy = int(round(state.y / resolution))
    heading = int(round(normalize_angle_positive(state.yaw) / (2.0 * math.pi) * heading_bins)) % heading_bins
    return ix, iy, heading


def global_path_heuristic(
    xy: tuple[float, float],
    dense_global: list[tuple[float, float]],
    local_goal: tuple[float, float],
) -> float:
    nearest = min(range(len(dense_global)), key=lambda idx: math.dist(xy, dense_global[idx]))
    distance_to_path = math.dist(xy, dense_global[nearest])
    remaining = polyline_length(dense_global[nearest:]) if nearest < len(dense_global) - 1 else 0.0
    tail = math.dist(dense_global[-1], local_goal) if dense_global else 0.0
    return distance_to_path + remaining + tail


def inside_local_window(
    start_xy: tuple[float, float],
    xy: tuple[float, float],
    radius: float,
) -> bool:
    return math.dist(start_xy, xy) <= radius


def reconstruct_path(
    states: dict[tuple[int, int, int], HybridState],
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None],
    key: tuple[int, int, int],
) -> list[HybridState]:
    path = [states[key]]
    while came_from[path_key := key] is not None:
        key = came_from[path_key]
        path.append(states[key])
    path.reverse()
    return path


def build_result(
    begin: float,
    terrain_map: ImplicitTerrainMap,
    states: dict[tuple[int, int, int], HybridState],
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None],
    goal_key: tuple[int, int, int],
    expanded: int,
) -> HybridLocalPlanningResult:
    state_path = reconstruct_path(states, came_from, goal_key)
    path = [(state.x, state.y, state.yaw) for state in state_path]
    risks = []
    stabilities = []
    for state in state_path:
        query = terrain_map.query((state.x, state.y))
        stability = estimate_tracked_configuration_stability(terrain_map, (state.x, state.y), state.yaw)
        risks.append(query.risk)
        stabilities.append(stability.stability)
    return HybridLocalPlanningResult(
        path=path,
        runtime_ms=(time.perf_counter() - begin) * 1000.0,
        expanded_nodes=expanded,
        path_length_m=polyline_length([(x, y) for x, y, _ in path]),
        mean_risk=float(np.mean(risks)),
        min_stability=float(np.min(stabilities)),
        success=len(path) >= 2,
    )


def empty_result(begin: float, expanded: int = 0) -> HybridLocalPlanningResult:
    return HybridLocalPlanningResult(
        path=[],
        runtime_ms=(time.perf_counter() - begin) * 1000.0,
        expanded_nodes=expanded,
        path_length_m=0.0,
        mean_risk=float("inf"),
        min_stability=0.0,
        success=False,
    )


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_angle_positive(angle: float) -> float:
    value = normalize_angle(angle)
    return value if value >= 0.0 else value + 2.0 * math.pi
