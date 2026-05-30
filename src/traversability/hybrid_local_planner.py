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
    smoothing_iterations: int = 2
    smoothing_offsets: tuple[float, ...] = (-0.12, 0.0, 0.12)


@dataclass(frozen=True)
class NDTLocalTraversabilityGuide:
    """Local query facade for traversability shared from the global NDT layer."""

    ndt_map: NDTImplicitMap
    traversable_keys: frozenset[tuple[int, int, int]]
    normals: dict[tuple[int, int, int], np.ndarray]

    @classmethod
    def from_ndt_map(cls, ndt_map: NDTImplicitMap) -> "NDTLocalTraversabilityGuide":
        metrics = ndt_map.compute_metrics()
        traversable = frozenset(key for key, metric in metrics.items() if np.isfinite(metric.traversal_cost))
        normals = {key: metric.normal for key, metric in metrics.items()}
        return cls(ndt_map=ndt_map, traversable_keys=traversable, normals=normals)

    def is_traversable(self, xy: tuple[float, float], z: float, radius_cells: int = 1) -> bool:
        return self.nearest_traversable_key(xy, z, radius_cells) is not None

    def normal_at(self, xy: tuple[float, float], z: float, radius_cells: int = 1) -> np.ndarray | None:
        key = self.nearest_traversable_key(xy, z, radius_cells)
        return self.normals.get(key) if key is not None else None

    def nearest_traversable_key(
        self,
        xy: tuple[float, float],
        z: float,
        radius_cells: int = 1,
    ) -> tuple[int, int, int] | None:
        key = self.key_from_xyz((xy[0], xy[1], z))
        best_key = None
        best_distance = float("inf")
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    candidate = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if candidate not in self.traversable_keys:
                        continue
                    distance = dx * dx + dy * dy + dz * dz
                    if distance < best_distance:
                        best_key = candidate
                        best_distance = distance
        return best_key

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
    raw_path: list[tuple[float, float, float]]
    runtime_ms: float
    expanded_nodes: int
    path_length_m: float
    raw_path_length_m: float
    path_states: int
    raw_path_states: int
    waypoint_states: int
    mean_risk: float
    min_stability: float
    curvature_cost: float
    raw_curvature_cost: float
    success: bool
    global_traversability_checks: int = 0
    global_normal_initializations: int = 0


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
    global_traversability_checks = 0
    global_normal_initializations = 0

    while frontier and expanded < config.max_iterations:
        _, current_key = heapq.heappop(frontier)
        current = states[current_key]
        expanded += 1
        goal_distance = math.dist((current.x, current.y), local_goal)
        if goal_distance < best_goal_distance:
            best_goal_distance = goal_distance
            best_key = current_key
        if goal_distance <= config.goal_tolerance:
            return build_result(
                begin,
                terrain_map,
                states,
                came_from,
                current_key,
                expanded,
                global_traversability_checks,
                global_normal_initializations,
                global_traversability,
                config.global_traversability_radius_cells,
                config,
            )

        for steer in (-config.max_steer, 0.0, config.max_steer):
            successor = propagate(current, steer, config)
            if not inside_local_window(start[:2], (successor.x, successor.y), config.local_window_radius):
                continue
            query = terrain_map.query((successor.x, successor.y))
            if query.obstacle or query.risk > config.max_risk:
                continue
            initial_normal = None
            if global_traversability is not None:
                global_traversability_checks += 1
                traversable_key = global_traversability.nearest_traversable_key(
                    (successor.x, successor.y),
                    query.height,
                    config.global_traversability_radius_cells,
                )
                if traversable_key is None:
                    continue
                initial_normal = global_traversability.normals.get(traversable_key)
                if initial_normal is not None:
                    global_normal_initializations += 1
            stability = estimate_tracked_configuration_stability(
                terrain_map,
                (successor.x, successor.y),
                successor.yaw,
                initial_normal=initial_normal,
            )
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
        return build_result(
            begin,
            terrain_map,
            states,
            came_from,
            best_key,
            expanded,
            global_traversability_checks,
            global_normal_initializations,
            global_traversability,
            config.global_traversability_radius_cells,
            config,
        )
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
    global_traversability_checks: int = 0,
    global_normal_initializations: int = 0,
    global_traversability: NDTLocalTraversabilityGuide | None = None,
    global_traversability_radius_cells: int = 1,
    config: HybridLocalPlannerConfig | None = None,
) -> HybridLocalPlanningResult:
    config = config or HybridLocalPlannerConfig()
    state_path = reconstruct_path(states, came_from, goal_key)
    raw_path = [(state.x, state.y, state.yaw) for state in state_path]
    waypoint_path = smooth_hybrid_path(terrain_map, raw_path, config, global_traversability, global_traversability_radius_cells)
    path = densify_hybrid_path(waypoint_path, config.step_length)
    risks = []
    stabilities = []
    for x, y, yaw in path:
        query = terrain_map.query((x, y))
        initial_normal = (
            global_traversability.normal_at((x, y), query.height, global_traversability_radius_cells)
            if global_traversability is not None
            else None
        )
        stability = estimate_tracked_configuration_stability(
            terrain_map,
            (x, y),
            yaw,
            initial_normal=initial_normal,
        )
        risks.append(query.risk)
        stabilities.append(stability.stability)
    return HybridLocalPlanningResult(
        path=path,
        raw_path=raw_path,
        runtime_ms=(time.perf_counter() - begin) * 1000.0,
        expanded_nodes=expanded,
        path_length_m=polyline_length([(x, y) for x, y, _ in path]),
        raw_path_length_m=polyline_length([(x, y) for x, y, _ in raw_path]),
        path_states=len(path),
        raw_path_states=len(raw_path),
        waypoint_states=len(waypoint_path),
        mean_risk=float(np.mean(risks)),
        min_stability=float(np.min(stabilities)),
        curvature_cost=path_curvature_cost(path),
        raw_curvature_cost=path_curvature_cost(raw_path),
        success=len(path) >= 2,
        global_traversability_checks=global_traversability_checks,
        global_normal_initializations=global_normal_initializations,
    )


def smooth_hybrid_path(
    terrain_map: ImplicitTerrainMap,
    path: list[tuple[float, float, float]],
    config: HybridLocalPlannerConfig,
    global_traversability: NDTLocalTraversabilityGuide | None,
    global_traversability_radius_cells: int,
) -> list[tuple[float, float, float]]:
    if len(path) <= 2 or config.smoothing_iterations <= 0:
        return path
    smoothed = shortcut_hybrid_path(terrain_map, path, config, global_traversability, global_traversability_radius_cells)
    for _ in range(config.smoothing_iterations):
        changed = False
        for index in range(1, len(smoothed) - 1):
            previous = smoothed[index - 1]
            current = smoothed[index]
            next_state = smoothed[index + 1]
            best = current
            best_cost = smoothing_local_cost(terrain_map, previous, current, next_state)
            for candidate_xy in lateral_candidates(previous[:2], current[:2], next_state[:2], config.smoothing_offsets):
                candidate_yaw = math.atan2(next_state[1] - previous[1], next_state[0] - previous[0])
                candidate = (candidate_xy[0], candidate_xy[1], candidate_yaw)
                if not candidate_is_safe(
                    terrain_map,
                    candidate,
                    previous,
                    next_state,
                    config,
                    global_traversability,
                    global_traversability_radius_cells,
                ):
                    continue
                candidate_cost = smoothing_local_cost(terrain_map, previous, candidate, next_state)
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
            if best != current:
                smoothed[index] = best
                changed = True
        if not changed:
            break
    return refresh_path_yaws(smoothed)


def shortcut_hybrid_path(
    terrain_map: ImplicitTerrainMap,
    path: list[tuple[float, float, float]],
    config: HybridLocalPlannerConfig,
    global_traversability: NDTLocalTraversabilityGuide | None,
    global_traversability_radius_cells: int,
) -> list[tuple[float, float, float]]:
    if len(path) <= 2:
        return path
    result = [path[0]]
    anchor = 0
    max_lookahead = 8
    while anchor < len(path) - 1:
        next_index = min(len(path) - 1, anchor + max_lookahead)
        while next_index > anchor + 1:
            if dense_segment_is_safe(
                terrain_map,
                path[anchor],
                path[next_index],
                config,
                global_traversability,
                global_traversability_radius_cells,
            ):
                break
            next_index -= 1
        result.append(path[next_index])
        anchor = next_index
    return refresh_path_yaws(result)


def candidate_is_safe(
    terrain_map: ImplicitTerrainMap,
    candidate: tuple[float, float, float],
    previous: tuple[float, float, float],
    next_state: tuple[float, float, float],
    config: HybridLocalPlannerConfig,
    global_traversability: NDTLocalTraversabilityGuide | None,
    global_traversability_radius_cells: int,
) -> bool:
    query = terrain_map.query(candidate[:2])
    if query.obstacle or query.risk > config.max_risk:
        return False
    initial_normal = None
    if global_traversability is not None:
        key = global_traversability.nearest_traversable_key(candidate[:2], query.height, global_traversability_radius_cells)
        if key is None:
            return False
        initial_normal = global_traversability.normals.get(key)
    stability = estimate_tracked_configuration_stability(
        terrain_map,
        candidate[:2],
        candidate[2],
        initial_normal=initial_normal,
    )
    return (
        stability.feasible
        and stability.stability >= config.min_stability
        and segment_is_safe(terrain_map, previous, candidate, config, global_traversability, global_traversability_radius_cells)
        and segment_is_safe(terrain_map, candidate, next_state, config, global_traversability, global_traversability_radius_cells)
    )


def segment_is_safe(
    terrain_map: ImplicitTerrainMap,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    config: HybridLocalPlannerConfig,
    global_traversability: NDTLocalTraversabilityGuide | None,
    global_traversability_radius_cells: int,
) -> bool:
    distance = math.dist(start[:2], end[:2])
    steps = max(2, int(distance / max(terrain_map.resolution, 1e-6)))
    yaw = math.atan2(end[1] - start[1], end[0] - start[0])
    for index in range(steps + 1):
        if index not in (0, steps, steps // 2):
            continue
        t = index / steps
        xy = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        query = terrain_map.query(xy)
        if query.obstacle or query.risk > config.max_risk:
            return False
        initial_normal = None
        if global_traversability is not None:
            key = global_traversability.nearest_traversable_key(xy, query.height, global_traversability_radius_cells)
            if key is None:
                return False
            initial_normal = global_traversability.normals.get(key)
        stability = estimate_tracked_configuration_stability(terrain_map, xy, yaw, initial_normal=initial_normal)
        if not stability.feasible or stability.stability < config.min_stability:
            return False
    return True


def dense_segment_is_safe(
    terrain_map: ImplicitTerrainMap,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    config: HybridLocalPlannerConfig,
    global_traversability: NDTLocalTraversabilityGuide | None,
    global_traversability_radius_cells: int,
) -> bool:
    distance = math.dist(start[:2], end[:2])
    steps = max(2, int(distance / max(terrain_map.resolution * 0.75, 1e-6)))
    yaw = math.atan2(end[1] - start[1], end[0] - start[0])
    for index in range(steps + 1):
        t = index / steps
        xy = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        query = terrain_map.query(xy)
        if query.obstacle or query.risk > config.max_risk:
            return False
        initial_normal = None
        if global_traversability is not None:
            key = global_traversability.nearest_traversable_key(xy, query.height, global_traversability_radius_cells)
            if key is None:
                return False
            initial_normal = global_traversability.normals.get(key)
        stability = estimate_tracked_configuration_stability(terrain_map, xy, yaw, initial_normal=initial_normal)
        if not stability.feasible or stability.stability < config.min_stability:
            return False
    return True


def smoothing_local_cost(
    terrain_map: ImplicitTerrainMap,
    previous: tuple[float, float, float],
    current: tuple[float, float, float],
    next_state: tuple[float, float, float],
) -> float:
    query = terrain_map.query(current[:2])
    length = math.dist(previous[:2], current[:2]) + math.dist(current[:2], next_state[:2])
    return length + 0.35 * heading_change(previous, current, next_state) + min(query.risk, 1.0)


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


def refresh_path_yaws(path: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    refreshed = []
    for index, state in enumerate(path):
        if index < len(path) - 1:
            neighbor = path[index + 1]
            yaw = math.atan2(neighbor[1] - state[1], neighbor[0] - state[0])
        elif refreshed:
            yaw = refreshed[-1][2]
        else:
            yaw = state[2]
        refreshed.append((state[0], state[1], normalize_angle(yaw)))
    return refreshed


def densify_hybrid_path(path: list[tuple[float, float, float]], spacing: float) -> list[tuple[float, float, float]]:
    if len(path) <= 1:
        return path
    dense = [path[0]]
    target_spacing = max(spacing, 1e-6)
    for start, end in zip(path[:-1], path[1:]):
        distance = math.dist(start[:2], end[:2])
        steps = max(1, int(math.ceil(distance / target_spacing)))
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        for step in range(1, steps + 1):
            t = step / steps
            dense.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t, normalize_angle(yaw)))
    return refresh_path_yaws(dense)


def path_curvature_cost(path: list[tuple[float, float, float]]) -> float:
    if len(path) < 3:
        return 0.0
    return float(sum(heading_change(a, b, c) for a, b, c in zip(path[:-2], path[1:-1], path[2:])))


def heading_change(
    previous: tuple[float, float, float],
    current: tuple[float, float, float],
    next_state: tuple[float, float, float],
) -> float:
    heading_a = math.atan2(current[1] - previous[1], current[0] - previous[0])
    heading_b = math.atan2(next_state[1] - current[1], next_state[0] - current[0])
    return abs(normalize_angle(heading_b - heading_a))


def empty_result(begin: float, expanded: int = 0) -> HybridLocalPlanningResult:
    return HybridLocalPlanningResult(
        path=[],
        raw_path=[],
        runtime_ms=(time.perf_counter() - begin) * 1000.0,
        expanded_nodes=expanded,
        path_length_m=0.0,
        raw_path_length_m=0.0,
        path_states=0,
        raw_path_states=0,
        waypoint_states=0,
        mean_risk=float("inf"),
        min_stability=0.0,
        curvature_cost=float("inf"),
        raw_curvature_cost=float("inf"),
        success=False,
    )


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_angle_positive(angle: float) -> float:
    value = normalize_angle(angle)
    return value if value >= 0.0 else value + 2.0 * math.pi
