from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path


def main() -> None:
    output = Path("runs/paper_receding_verify")
    subprocess.run([sys.executable, "-m", "traversability.paper_receding_demo", "--output", str(output)], check=True)

    summary_path = output / "receding_summary.csv"
    trajectory_path = output / "executed_trajectory.csv"
    image_path = output / "paper_receding.png"
    require(summary_path.exists(), "missing receding summary")
    require(trajectory_path.exists(), "missing executed trajectory")
    require(image_path.exists() and image_path.stat().st_size > 10_000, "missing or empty visualization")

    with summary_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    with trajectory_path.open(encoding="utf-8") as file:
        trajectory = [(float(row["x"]), float(row["y"]), float(row["yaw"])) for row in csv.DictReader(file)]

    require(len(rows) >= 4, "too few receding cycles")
    success_rate = sum(int(row["success"]) for row in rows) / len(rows)
    require(success_rate >= 0.90, "receding planning success rate too low")
    require(len(trajectory) == len(rows) + 1 or len(trajectory) == len(rows) + 2, "trajectory length does not match cycles")
    require(float(rows[-1]["distance_to_goal_m"]) < float(rows[0]["distance_to_goal_m"]), "trajectory did not approach goal")
    require(float(rows[-1]["distance_to_goal_m"]) <= 1.6, "final distance remains too large for receding demo")
    require(
        all(float(row["local_path_length_m"]) <= float(row["local_raw_path_length_m"]) + 0.25 for row in rows),
        "local smoothing made a receding path much longer",
    )
    require(
        all(int(float(row["local_path_states"])) >= 4 for row in rows),
        "smoothed receding path has too few executable states",
    )
    require(
        all(int(float(row["local_waypoint_states"])) <= int(float(row["local_raw_path_states"])) for row in rows),
        "local smoothing increased receding waypoint states",
    )
    require(
        all(float(row["local_curvature_cost"]) <= float(row["local_raw_curvature_cost"]) + 1e-6 for row in rows),
        "local smoothing did not reduce receding path curvature",
    )
    require(
        any(int(float(row["local_waypoint_states"])) < int(float(row["local_raw_path_states"])) for row in rows),
        "local smoothing did not simplify any receding Hybrid path",
    )
    require(max(float(row["local_mean_risk"]) for row in rows) <= 0.45, "local path risk too high")
    require(min(float(row["local_min_stability"]) for row in rows) >= 0.28, "local path stability too low")
    require(min(int(float(row["shared_traversable_voxels"])) for row in rows) >= 150, "global traversability was not shared")
    require(min(int(float(row["global_traversability_checks"])) for row in rows) > 0, "local planning did not query global traversability")
    require(min(int(float(row["global_normal_initializations"])) for row in rows) > 0, "local stability did not use global normals")
    require(any(float(row["global_path_length_m"]) < float(rows[0]["global_path_length_m"]) for row in rows[2:]), "post-update global path was not recomputed")
    require(path_moves_smoothly(trajectory), "executed trajectory contains an implausible jump")

    print("paper receding pipeline verification passed")


def path_moves_smoothly(trajectory: list[tuple[float, float, float]]) -> bool:
    distances = [math.dist(a[:2], b[:2]) for a, b in zip(trajectory[:-1], trajectory[1:])]
    return bool(distances) and min(distances) > 0.20 and max(distances) < 1.10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"paper receding verification failed: {message}")


if __name__ == "__main__":
    main()
