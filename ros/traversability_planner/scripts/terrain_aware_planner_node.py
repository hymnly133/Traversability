#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traversability.planner import plan_multilevel
from traversability.pointcloud import load_point_cloud, point_cloud_to_elevation_grid
from traversability.terrain import make_pyramid


def main() -> None:
    config = read_config()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_point_cloud(config["pointcloud_file"])
    grid = point_cloud_to_elevation_grid(points, resolution=float(config["resolution"]))
    pyramid = make_pyramid(grid.height, grid.obstacle, grid.resolution, grid.origin_xy, levels=int(config["levels"]))
    start = (grid.origin_xy[0] + 2.5, grid.origin_xy[1] + 3.0)
    goal = (
        grid.origin_xy[0] + grid.height.shape[1] * grid.resolution - 3.0,
        grid.origin_xy[1] + grid.height.shape[0] * grid.resolution - 4.0,
    )
    result = plan_multilevel(pyramid, start, goal)
    write_path(output_dir / "ros_planned_path.csv", result.path_xy)
    write_summary(output_dir / "ros_summary.csv", result)
    log(f"success={result.success} path_length={result.path_length_m:.2f} output={output_dir.resolve()}")


def read_config() -> dict:
    try:
        import rospy  # type: ignore

        if not rospy.core.is_initialized():
            rospy.init_node("terrain_aware_planner", anonymous=False)
        return {
            "pointcloud_file": rospy.get_param("~pointcloud_file"),
            "resolution": rospy.get_param("~resolution", 0.25),
            "levels": rospy.get_param("~levels", 4),
            "output_dir": rospy.get_param("~output_dir", "runs/ros_offline"),
        }
    except Exception:
        parser = argparse.ArgumentParser(description="ROS-compatible offline terrain-aware planner node.")
        parser.add_argument("--pointcloud-file", required=True)
        parser.add_argument("--resolution", type=float, default=0.25)
        parser.add_argument("--levels", type=int, default=4)
        parser.add_argument("--output-dir", default="runs/ros_offline")
        args = parser.parse_args()
        return {
            "pointcloud_file": args.pointcloud_file,
            "resolution": args.resolution,
            "levels": args.levels,
            "output_dir": args.output_dir,
        }


def write_path(path: Path, path_xy: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for x, y in path_xy:
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])


def write_summary(path: Path, result) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["success", "runtime_ms", "expanded_nodes", "path_length_m", "max_risk", "feasible_rate"])
        writer.writerow(
            [
                int(result.success),
                f"{result.runtime_ms:.3f}",
                result.expanded_nodes,
                f"{result.path_length_m:.3f}",
                f"{result.max_risk:.4f}",
                f"{result.feasible_rate:.4f}",
            ]
        )


def log(message: str) -> None:
    try:
        import rospy  # type: ignore

        rospy.loginfo(message)
    except Exception:
        print(message)


if __name__ == "__main__":
    main()

