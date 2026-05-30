from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the multilevel terrain-aware reproduction demo.")
    parser.add_argument("--output", type=Path, default=Path("runs/multilevel_verify"))
    parser.add_argument("--max-runtime-ms", type=float, default=6000.0)
    parser.add_argument("--max-risk", type=float, default=0.90)
    parser.add_argument("--min-stability", type=float, default=0.08)
    parser.add_argument("--min-feasible-rate", type=float, default=0.80)
    parser.add_argument("--min-path-length", type=float, default=40.0)
    parser.add_argument("--check-realtime", action="store_true")
    parser.add_argument("--check-ablation", action="store_true")
    parser.add_argument("--check-tracking", action="store_true")
    parser.add_argument("--check-pointcloud", action="store_true")
    parser.add_argument("--check-benchmark", action="store_true")
    parser.add_argument("--check-ros-wrapper", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "traversability.verify_implicit_map",
    ]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-m", "traversability.verify_pointcloud_io"], check=True)

    command = [
        sys.executable,
        "-m",
        "traversability.multilevel_demo",
        "--output",
        str(args.output),
    ]
    subprocess.run(command, check=True)

    summary_path = args.output / "summary.csv"
    path_path = args.output / "planned_path.csv"
    samples_path = args.output / "stability_samples.csv"
    image_path = args.output / "multilevel_plan.png"

    require(summary_path.exists(), f"missing {summary_path}")
    require(path_path.exists(), f"missing {path_path}")
    require(samples_path.exists(), f"missing {samples_path}")
    require(image_path.exists() and image_path.stat().st_size > 10_000, f"missing or empty {image_path}")

    with summary_path.open(encoding="utf-8") as file:
        summary = next(csv.DictReader(file))
    with path_path.open(encoding="utf-8") as file:
        path_rows = list(csv.DictReader(file))
    with samples_path.open(encoding="utf-8") as file:
        sample_rows = list(csv.DictReader(file))

    require(summary["success"] == "1", "planner did not report success")
    require(float(summary["runtime_ms"]) <= args.max_runtime_ms, "runtime exceeded threshold")
    require(float(summary["max_risk"]) <= args.max_risk, "max path risk exceeded threshold")
    require(float(summary["min_stability"]) >= args.min_stability, "minimum stability below threshold")
    require(float(summary["feasible_rate"]) >= args.min_feasible_rate, "configuration feasibility rate below threshold")
    require(float(summary["path_length_m"]) >= args.min_path_length, "path length is unexpectedly short")
    require(len(path_rows) >= 2, "planned path has fewer than two points")
    require(len(sample_rows) >= 20, "not enough stability samples")

    if args.check_realtime:
        realtime_output = args.output.parent / f"{args.output.name}_realtime"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "traversability.realtime_demo",
                "--output",
                str(realtime_output),
            ],
            check=True,
        )
        verify_realtime_outputs(realtime_output)

    if args.check_ablation:
        ablation_output = args.output.parent / f"{args.output.name}_ablation"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "traversability.ablation_demo",
                "--output",
                str(ablation_output),
            ],
            check=True,
        )
        verify_ablation_outputs(ablation_output)

    if args.check_tracking:
        tracking_output = args.output.parent / f"{args.output.name}_tracking"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "traversability.tracking_demo",
                "--output",
                str(tracking_output),
            ],
            check=True,
        )
        verify_tracking_outputs(tracking_output)

    if args.check_pointcloud:
        pointcloud_output = args.output.parent / f"{args.output.name}_pointcloud"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "traversability.pointcloud_demo",
                "--output",
                str(pointcloud_output),
            ],
            check=True,
        )
        verify_pointcloud_outputs(pointcloud_output)

    if args.check_benchmark:
        benchmark_output = args.output.parent / f"{args.output.name}_benchmark"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "traversability.benchmark_suite",
                "--output",
                str(benchmark_output),
            ],
            check=True,
        )
        verify_benchmark_outputs(benchmark_output)

    if args.check_ros_wrapper:
        subprocess.run([sys.executable, "-m", "traversability.verify_ros_wrapper"], check=True)

    Console().print("[green]verification passed[/green]")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def verify_realtime_outputs(output: Path) -> None:
    summary_path = output / "replanning_summary.csv"
    trajectory_path = output / "executed_trajectory.csv"
    image_path = output / "replanning.png"
    require(summary_path.exists(), f"missing {summary_path}")
    require(trajectory_path.exists(), f"missing {trajectory_path}")
    require(image_path.exists() and image_path.stat().st_size > 10_000, f"missing or empty {image_path}")
    with summary_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    require(len(rows) >= 4, "not enough replanning cycles")
    success_rate = sum(int(row["success"]) for row in rows) / len(rows)
    p95_runtime = sorted(float(row["runtime_ms"]) for row in rows)[int(0.95 * (len(rows) - 1))]
    require(success_rate >= 0.75, "rolling replanning success rate too low")
    require(p95_runtime <= 5000.0, "rolling replanning p95 runtime too high")


def verify_ablation_outputs(output: Path) -> None:
    summary_path = output / "ablation_summary.csv"
    image_path = output / "ablation.png"
    require(summary_path.exists(), f"missing {summary_path}")
    require(image_path.exists() and image_path.stat().st_size > 10_000, f"missing or empty {image_path}")
    with summary_path.open(encoding="utf-8") as file:
        rows = {row["planner"]: row for row in csv.DictReader(file)}
    require("multilevel" in rows and "single_level" in rows, "ablation rows missing planners")
    multilevel = rows["multilevel"]
    single = rows["single_level"]
    require(multilevel["success"] == "1", "multilevel ablation planner failed")
    require(single["success"] == "1", "single-level baseline failed")
    speedup = float(single["runtime_ms"]) / max(float(multilevel["runtime_ms"]), 1e-9)
    expansion_ratio = int(single["expanded_nodes"]) / max(int(multilevel["expanded_nodes"]), 1)
    require(speedup >= 1.5, "multilevel planner did not show enough runtime speedup")
    require(expansion_ratio >= 1.5, "multilevel planner did not reduce expanded nodes enough")
    require(float(multilevel["max_risk"]) <= 0.90, "multilevel ablation risk exceeded threshold")
    require(float(multilevel["feasible_rate"]) >= 0.80, "multilevel ablation feasibility below threshold")


def verify_tracking_outputs(output: Path) -> None:
    summary_path = output / "tracking_summary.csv"
    states_path = output / "tracking_states.csv"
    image_path = output / "tracking.png"
    require(summary_path.exists(), f"missing {summary_path}")
    require(states_path.exists(), f"missing {states_path}")
    require(image_path.exists() and image_path.stat().st_size > 10_000, f"missing or empty {image_path}")
    with summary_path.open(encoding="utf-8") as file:
        summary = next(csv.DictReader(file))
    with states_path.open(encoding="utf-8") as file:
        states = list(csv.DictReader(file))
    require(summary["plan_success"] == "1", "tracking demo planner failed")
    require(summary["tracking_success"] == "1", "path tracking failed")
    require(float(summary["mean_error_m"]) <= 0.20, "tracking mean error too high")
    require(float(summary["max_error_m"]) <= 0.60, "tracking max error too high")
    require(float(summary["final_error_m"]) <= 0.55, "tracking final error too high")
    require(float(summary["max_risk"]) <= 0.95, "tracking max risk exceeded threshold")
    require(float(summary["min_stability"]) >= 0.08, "tracking minimum stability below threshold")
    require(float(summary["feasible_rate"]) >= 0.75, "tracking feasible rate below threshold")
    require(len(states) >= 100, "tracking state trajectory is too short")


def verify_pointcloud_outputs(output: Path) -> None:
    summary_path = output / "pointcloud_summary.csv"
    image_path = output / "pointcloud_plan.png"
    require(summary_path.exists(), f"missing {summary_path}")
    require(image_path.exists() and image_path.stat().st_size > 10_000, f"missing or empty {image_path}")
    with summary_path.open(encoding="utf-8") as file:
        summary = next(csv.DictReader(file))
    require(summary["success"] == "1", "point-cloud planning failed")
    require(int(summary["points"]) >= 10_000, "point cloud has too few points")
    require(int(summary["observed_cells"]) >= 10_000, "too few observed elevation cells")
    require(float(summary["max_risk"]) <= 0.90, "point-cloud plan max risk exceeded threshold")
    require(float(summary["feasible_rate"]) >= 0.80, "point-cloud plan feasibility below threshold")


def verify_benchmark_outputs(output: Path) -> None:
    summary_path = output / "benchmark_summary.csv"
    scenarios_path = output / "benchmark_scenarios.csv"
    require(summary_path.exists(), f"missing {summary_path}")
    require(scenarios_path.exists(), f"missing {scenarios_path}")
    with summary_path.open(encoding="utf-8") as file:
        summary = {row["metric"]: row["value"] for row in csv.DictReader(file)}
    require(int(float(summary["scenarios"])) >= 3, "benchmark did not run enough scenarios")
    require(float(summary["multilevel_success_rate"]) >= 0.95, "benchmark multilevel success rate too low")
    require(float(summary["single_success_rate"]) >= 0.95, "benchmark single-level success rate too low")
    require(float(summary["mean_runtime_speedup"]) >= 1.5, "benchmark speedup below threshold")
    require(float(summary["mean_expanded_ratio"]) >= 1.5, "benchmark expanded-node ratio below threshold")
    require(float(summary["mean_multilevel_max_risk"]) <= 0.90, "benchmark mean max risk too high")
    require(float(summary["mean_multilevel_feasible_rate"]) >= 0.80, "benchmark feasible rate too low")


if __name__ == "__main__":
    main()
