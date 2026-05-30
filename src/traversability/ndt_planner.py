from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from traversability.ndt_map import NDTImplicitMap, NDTMetric


@dataclass(frozen=True)
class NDTPlannerConfig:
    length_weight: float = 1.0
    traversal_weight: float = 1.6
    neighbor_mode: int = 26


@dataclass(frozen=True)
class NDTPlanningResult:
    path_keys: list[tuple[int, int, int]]
    path_xyz: list[tuple[float, float, float]]
    runtime_ms: float
    expanded_nodes: int
    path_length_m: float
    mean_traversal_cost: float
    max_traversal_cost: float
    connected_components: int
    traversable_voxels: int
    success: bool
    reused_connected_set: bool = False


class UnionFind:
    def __init__(self, keys: set[tuple[int, int, int]]):
        self.parent = {key: key for key in keys}
        self.rank = {key: 0 for key in keys}

    def find(self, key: tuple[int, int, int]) -> tuple[int, int, int]:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def component_count(self) -> int:
        return len({self.find(key) for key in self.parent})


def plan_ndt_global(
    ndt_map: NDTImplicitMap,
    start_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    config: NDTPlannerConfig | None = None,
) -> NDTPlanningResult:
    config = config or NDTPlannerConfig()
    begin = time.perf_counter()
    metrics = ndt_map.compute_metrics()
    traversable = {key for key, metric in metrics.items() if np.isfinite(metric.traversal_cost)}
    if not traversable:
        return empty_result(begin)

    connected, reused_connected_set = get_connected_traversable_set(ndt_map, traversable, config.neighbor_mode)
    start_key = nearest_traversable_key(ndt_map, traversable, np.asarray(start_xyz, dtype=np.float64))
    goal_key = nearest_traversable_key(ndt_map, traversable, np.asarray(goal_xyz, dtype=np.float64))
    if start_key is None or goal_key is None:
        return empty_result(begin, connected.component_count(), len(traversable))

    start_root = connected.find(start_key)
    if connected.find(goal_key) != start_root:
        same_component = {key for key in traversable if connected.find(key) == start_root}
        goal_key = nearest_traversable_key(ndt_map, same_component, np.asarray(goal_xyz, dtype=np.float64))
        if goal_key is None:
            return empty_result(begin, connected.component_count(), len(traversable))

    path_keys, expanded = astar_ndt(ndt_map, metrics, traversable, start_key, goal_key, config)
    runtime_ms = (time.perf_counter() - begin) * 1000.0
    if not path_keys:
        return empty_result(begin, connected.component_count(), len(traversable), expanded)

    path_xyz_arrays = [ndt_map.voxel_center(key) for key in path_keys]
    costs = np.array([metrics[key].traversal_cost for key in path_keys], dtype=np.float64)
    return NDTPlanningResult(
        path_keys=path_keys,
        path_xyz=[tuple(point.tolist()) for point in path_xyz_arrays],
        runtime_ms=runtime_ms,
        expanded_nodes=expanded,
        path_length_m=polyline_length_3d(path_xyz_arrays),
        mean_traversal_cost=float(np.mean(costs)),
        max_traversal_cost=float(np.max(costs)),
        connected_components=connected.component_count(),
        traversable_voxels=len(traversable),
        success=True,
        reused_connected_set=reused_connected_set,
    )


def get_connected_traversable_set(
    ndt_map: NDTImplicitMap,
    traversable: set[tuple[int, int, int]],
    neighbor_mode: int,
) -> tuple[UnionFind, bool]:
    frozen_traversable = frozenset(traversable)
    cache_key = (neighbor_mode, frozen_traversable)
    cached = ndt_map.connected_cache.get(cache_key)
    if cached is not None:
        return cached, True
    connected = build_connected_traversable_set(traversable, neighbor_mode)
    ndt_map.connected_cache = {cache_key: connected}
    return connected, False


def build_connected_traversable_set(traversable: set[tuple[int, int, int]], neighbor_mode: int = 26) -> UnionFind:
    union_find = UnionFind(traversable)
    offsets = positive_neighbor_offsets(neighbor_mode)
    for key in traversable:
        for offset in offsets:
            neighbor = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            if neighbor in traversable:
                union_find.union(key, neighbor)
    return union_find


def astar_ndt(
    ndt_map: NDTImplicitMap,
    metrics: dict[tuple[int, int, int], NDTMetric],
    traversable: set[tuple[int, int, int]],
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    config: NDTPlannerConfig,
) -> tuple[list[tuple[int, int, int]], int]:
    frontier: list[tuple[float, tuple[int, int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int, int], float] = {start: 0.0}
    expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        expanded += 1
        if current == goal:
            break
        for neighbor in traversable_neighbors(current, traversable, config.neighbor_mode):
            transition = ndt_transition_cost(ndt_map, metrics, current, neighbor, config)
            new_cost = cost_so_far[current] + transition
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean_key_distance(neighbor, goal, ndt_map.config.voxel_size)
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


def ndt_transition_cost(
    ndt_map: NDTImplicitMap,
    metrics: dict[tuple[int, int, int], NDTMetric],
    current: tuple[int, int, int],
    neighbor: tuple[int, int, int],
    config: NDTPlannerConfig,
) -> float:
    distance = euclidean_key_distance(current, neighbor, ndt_map.config.voxel_size)
    return config.length_weight * distance + config.traversal_weight * metrics[neighbor].traversal_cost


def nearest_traversable_key(
    ndt_map: NDTImplicitMap,
    keys: set[tuple[int, int, int]],
    xyz: np.ndarray,
) -> tuple[int, int, int] | None:
    if not keys:
        return None
    return min(keys, key=lambda key: float(np.linalg.norm(ndt_map.voxel_center(key) - xyz)))


def traversable_neighbors(
    key: tuple[int, int, int],
    traversable: set[tuple[int, int, int]],
    neighbor_mode: int,
) -> list[tuple[int, int, int]]:
    result = []
    for dx, dy, dz in neighbor_offsets(neighbor_mode):
        neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
        if neighbor in traversable:
            result.append(neighbor)
    return result


def neighbor_offsets(neighbor_mode: int) -> list[tuple[int, int, int]]:
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if neighbor_mode == 6 and manhattan != 1:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def positive_neighbor_offsets(neighbor_mode: int) -> list[tuple[int, int, int]]:
    return [offset for offset in neighbor_offsets(neighbor_mode) if offset > (0, 0, 0)]


def euclidean_key_distance(a: tuple[int, int, int], b: tuple[int, int, int], voxel_size: float) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) * voxel_size


def polyline_length_3d(points: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:])))


def empty_result(
    begin: float,
    connected_components: int = 0,
    traversable_voxels: int = 0,
    expanded_nodes: int = 0,
) -> NDTPlanningResult:
    return NDTPlanningResult(
        path_keys=[],
        path_xyz=[],
        runtime_ms=(time.perf_counter() - begin) * 1000.0,
        expanded_nodes=expanded_nodes,
        path_length_m=0.0,
        mean_traversal_cost=float("inf"),
        max_traversal_cost=float("inf"),
        connected_components=connected_components,
        traversable_voxels=traversable_voxels,
        success=False,
    )
