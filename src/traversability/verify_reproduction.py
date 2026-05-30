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
    parser.add_argument("--max-runtime-ms", type=float, default=5000.0)
    parser.add_argument("--max-risk", type=float, default=0.90)
    parser.add_argument("--min-stability", type=float, default=0.08)
    parser.add_argument("--min-feasible-rate", type=float, default=0.80)
    parser.add_argument("--min-path-length", type=float, default=40.0)
    parser.add_argument("--check-realtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "traversability.verify_implicit_map",
    ]
    subprocess.run(command, check=True)

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


if __name__ == "__main__":
    main()
