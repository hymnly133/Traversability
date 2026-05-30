from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path


def main() -> None:
    output = Path("runs/paper_pipeline_verify")
    subprocess.run([sys.executable, "-m", "traversability.paper_pipeline_demo", "--output", str(output)], check=True)

    summary_path = output / "paper_pipeline_summary.csv"
    global_path_path = output / "global_path.csv"
    local_path_path = output / "local_path.csv"
    image_path = output / "paper_pipeline.png"
    require(summary_path.exists(), "missing paper pipeline summary")
    require(global_path_path.exists(), "missing global path")
    require(local_path_path.exists(), "missing local path")
    require(image_path.exists() and image_path.stat().st_size > 10_000, "missing or empty visualization")

    with summary_path.open(encoding="utf-8") as file:
        summary = next(csv.DictReader(file))
    with global_path_path.open(encoding="utf-8") as file:
        global_path = [(float(row["x"]), float(row["y"]), float(row["z"])) for row in csv.DictReader(file)]
    with local_path_path.open(encoding="utf-8") as file:
        local_path = [(float(row["x"]), float(row["y"]), float(row["yaw"])) for row in csv.DictReader(file)]

    require(summary["global_success"] == "1", "NDT global planner failed")
    require(summary["local_success"] == "1", "Hybrid local planner failed")
    require(float(summary["global_path_length_m"]) >= 3.0, "global path is unexpectedly short")
    require(float(summary["local_path_length_m"]) >= 1.4, "local path is unexpectedly short")
    require(float(summary["local_path_length_m"]) <= float(summary["local_raw_path_length_m"]) + 0.25, "local smoothing made the path much longer")
    require(float(summary["local_curvature_cost"]) <= float(summary["local_raw_curvature_cost"]) + 1e-6, "local smoothing did not reduce curvature")
    require(int(summary["local_path_states"]) >= 6, "smoothed local path has too few executable states")
    require(int(summary["local_waypoint_states"]) <= int(summary["local_raw_path_states"]), "local smoothing increased waypoint states")
    require(
        int(summary["local_waypoint_states"]) < int(summary["local_raw_path_states"])
        or float(summary["local_curvature_cost"]) < float(summary["local_raw_curvature_cost"]),
        "local smoothing did not simplify the Hybrid path",
    )
    require(float(summary["global_max_traversal_cost"]) < 0.82, "global path crossed untraversable cost")
    require(float(summary["local_mean_risk"]) < 0.65, "local path risk too high")
    require(float(summary["local_min_stability"]) >= 0.28, "local path stability too low")
    require(int(summary["shared_traversable_voxels"]) >= 150, "global traversability was not shared with local planning")
    require(int(summary["global_traversability_checks"]) > 0, "local planning did not query global traversability")
    require(int(summary["global_normal_initializations"]) > 0, "local stability did not use global normal initialization")
    require(int(summary["local_traversable_voxels"]) > 0, "local traversable set was not built")
    require(int(summary["local_traversable_voxels"]) <= int(summary["shared_traversable_voxels"]), "local traversable set was not windowed")
    require(int(summary["local_traversable_queries"]) > 0, "local traversable set was not queried")
    require(len(global_path) >= 8, "global path has too few waypoints")
    require(len(local_path) >= 6, "local path has too few states")
    require(path_bends_around_obstacle(global_path), "global path did not bend around the obstacle")
    require(path_bends_around_obstacle(local_path), "local path did not bend around the obstacle")
    require(not crosses_obstacle_band(global_path), "global path crossed obstacle band")
    require(not crosses_obstacle_band(local_path), "local path crossed obstacle band")
    require(local_follows_global(local_path, global_path), "local path does not follow the NDT global guide")

    print("paper pipeline verification passed")


def path_bends_around_obstacle(path: list[tuple[float, ...]]) -> bool:
    return any(abs(point[1]) > 0.55 for point in path)


def crosses_obstacle_band(path: list[tuple[float, ...]]) -> bool:
    return any(abs(point[0]) < 0.22 and -0.60 < point[1] < 0.74 for point in path)


def local_follows_global(local_path: list[tuple[float, ...]], global_path: list[tuple[float, ...]]) -> bool:
    local_end = local_path[-1]
    distances = [math.dist(local_end[:2], point[:2]) for point in global_path]
    return min(distances) <= 0.55


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"paper pipeline verification failed: {message}")


if __name__ == "__main__":
    main()
