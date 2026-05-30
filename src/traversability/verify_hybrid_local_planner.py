from __future__ import annotations

import math

import numpy as np

from traversability.hybrid_local_planner import HybridLocalPlannerConfig, plan_hybrid_local
from traversability.implicit_map import ImplicitTerrainMap
from traversability.terrain import analyze_layer
from traversability.tracked_stability import estimate_tracked_configuration_stability


def main() -> None:
    terrain_map = make_local_map()
    global_path = [(-1.25, -0.65), (-0.65, -0.65), (-0.42, 0.78), (0.45, 0.90), (1.25, 0.70)]
    result = plan_hybrid_local(
        terrain_map,
        start=(-1.25, -0.65, 0.0),
        global_path=global_path,
        config=HybridLocalPlannerConfig(
            step_length=0.24,
            local_window_radius=3.0,
            goal_tolerance=0.34,
            min_stability=0.28,
            max_iterations=7000,
        ),
    )
    require(result.success, "Hybrid A* local planner failed")
    require(result.expanded_nodes > 20, "planner expanded too few nodes to exercise Hybrid A*")
    require(2.0 <= result.path_length_m <= 4.6, "local path length is outside expected range")
    require(result.mean_risk <= 0.55, "local path mean risk is too high")
    require(result.min_stability >= 0.28, "local path stability is below threshold")
    require(path_reaches_guided_goal(result.path, global_path[-1]), "local path did not reach the guided local goal")
    require(not path_crosses_obstacle(result.path), "local path crossed the obstacle band")
    require(any(abs(y) > 0.55 for _, y, _ in result.path), "local path did not bend around the obstacle")

    for x, y, yaw in result.path[:: max(1, len(result.path) // 5)]:
        stability = estimate_tracked_configuration_stability(terrain_map, (x, y), yaw)
        require(stability.feasible, "sampled Hybrid A* state failed tracked stability")

    print("Hybrid local planner verification passed")


def make_local_map() -> ImplicitTerrainMap:
    resolution = 0.06
    size = 72
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    xs = origin[0] + np.arange(size) * resolution
    ys = origin[1] + np.arange(size) * resolution
    xx, yy = np.meshgrid(xs, ys)
    height = 0.035 * np.sin(2.0 * xx) + 0.025 * np.cos(2.4 * yy)
    obstacle = (np.abs(xx) < 0.18) & (yy > -0.40) & (yy < 0.62)
    height = height.copy()
    height[obstacle] += 0.8
    layer = analyze_layer("hybrid_local", height, obstacle, resolution, origin)
    return ImplicitTerrainMap(layer)


def path_reaches_guided_goal(path: list[tuple[float, float, float]], goal: tuple[float, float]) -> bool:
    x, y, _ = path[-1]
    return math.dist((x, y), goal) <= 0.55


def path_crosses_obstacle(path: list[tuple[float, float, float]]) -> bool:
    for x, y, _ in path:
        if abs(x) < 0.22 and -0.45 < y < 0.68:
            return True
    return False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Hybrid local planner verification failed: {message}")


if __name__ == "__main__":
    main()
