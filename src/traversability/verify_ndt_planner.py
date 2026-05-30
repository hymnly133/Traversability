from __future__ import annotations

import math

import numpy as np

from traversability.ndt_map import NDTConfig, NDTImplicitMap
from traversability.ndt_planner import build_connected_traversable_set, plan_ndt_global


def main() -> None:
    points = make_corridor_cloud()
    ndt_map = NDTImplicitMap(
        points,
        NDTConfig(
            voxel_size=0.25,
            fusion_radius=0.42,
            saturation_count=3,
            slope_threshold_rad=math.radians(35.0),
            complexity_threshold=0.70,
            robot_radius=0.30,
            robot_height=0.50,
        ),
    )
    metrics = ndt_map.compute_metrics()
    traversable = {key for key, metric in metrics.items() if np.isfinite(metric.traversal_cost)}
    connected = build_connected_traversable_set(traversable)
    require(len(traversable) > 100, "too few traversable voxels")
    require(connected.component_count() >= 1, "connected traversable voxel set was not built")

    result = plan_ndt_global(ndt_map, start_xyz=(-2.05, -0.9, 0.0), goal_xyz=(2.05, 0.9, 0.0))
    require(result.success, "NDT global planner failed")
    require(result.expanded_nodes > 5, "planner expanded too few nodes to exercise A*")
    require(result.path_length_m > 3.5, "path length is unexpectedly short")
    require(result.path_length_m < 7.0, "path length is unexpectedly long")
    require(result.max_traversal_cost < 0.70, "path crossed a risky voxel")
    require(len(result.path_keys) >= 8, "path has too few voxels")

    path = np.asarray(result.path_xyz, dtype=np.float64)
    require(np.min(path[:, 0]) < -1.6 and np.max(path[:, 0]) > 1.6, "path does not span start to goal")
    require(np.max(np.abs(path[:, 1])) > 0.35, "path did not use the low-risk corridor around the barrier")
    require(not path_crosses_barrier(metrics, result.path_keys), "path crossed the high-risk barrier")

    print("NDT global planner verification passed")


def make_corridor_cloud() -> np.ndarray:
    rng = np.random.default_rng(23)
    xs = np.arange(-2.4, 2.41, 0.08)
    ys = np.arange(-1.25, 1.26, 0.08)
    points = []
    for x in xs:
        for y in ys:
            z = 0.04 * math.sin(2.0 * x) + 0.025 * rng.normal()
            if abs(x) < 0.16 and abs(y) < 0.72:
                z += 0.95
            samples = 3
            jitter = rng.normal(0.0, 0.018, size=(samples, 3))
            points.append(np.array([x, y, z], dtype=np.float64) + jitter)
    return np.vstack(points)


def path_crosses_barrier(metrics, path_keys: list[tuple[int, int, int]]) -> bool:
    for key in path_keys:
        metric = metrics[key]
        x, y, _ = metric.center
        if abs(float(x)) < 0.22 and abs(float(y)) < 0.65:
            return True
    return False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NDT planner verification failed: {message}")


if __name__ == "__main__":
    main()
