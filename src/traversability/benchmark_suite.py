from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.ablation_demo import plan_single_level
from traversability.planner import plan_multilevel
from traversability.terrain import generate_large_rough_terrain, make_pyramid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-scenario terrain-aware planning benchmarks.")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--size", type=int, default=140)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--levels", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.append(run_scenario(seed, args.size, args.resolution, args.levels))
    write_outputs(args.output, rows)
    print_summary(args.output, rows)


def run_scenario(seed: int, size: int, resolution: float, levels: int) -> dict:
    height, obstacle, origin = generate_large_rough_terrain(size=size, resolution=resolution, seed=seed)
    pyramid = make_pyramid(height, obstacle, resolution, origin, levels=levels)
    start = (origin[0] + 2.5, origin[1] + 3.0)
    goal = (origin[0] + size * resolution - 3.0, origin[1] + size * resolution - 4.0)
    multilevel = plan_multilevel(pyramid, start, goal)
    single = plan_single_level(pyramid.finest, start, goal)

    speedup = single["runtime_ms"] / max(multilevel.runtime_ms, 1e-9)
    expansion_ratio = single["expanded_nodes"] / max(multilevel.expanded_nodes, 1)
    return {
        "seed": seed,
        "multilevel_success": int(multilevel.success),
        "single_success": int(single["success"]),
        "multilevel_runtime_ms": multilevel.runtime_ms,
        "single_runtime_ms": single["runtime_ms"],
        "runtime_speedup": speedup,
        "multilevel_expanded": multilevel.expanded_nodes,
        "single_expanded": single["expanded_nodes"],
        "expanded_ratio": expansion_ratio,
        "multilevel_path_length_m": multilevel.path_length_m,
        "single_path_length_m": single["path_length_m"],
        "multilevel_max_risk": multilevel.max_risk,
        "single_max_risk": single["max_risk"],
        "multilevel_min_stability": multilevel.min_stability,
        "single_min_stability": single["min_stability"],
        "multilevel_feasible_rate": multilevel.feasible_rate,
        "single_feasible_rate": single["feasible_rate"],
    }


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "benchmark_scenarios.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))

    aggregate = aggregate_rows(rows)
    with (output_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in aggregate.items():
            writer.writerow([key, f"{value:.4f}" if isinstance(value, float) else value])


def format_row(row: dict) -> dict:
    formatted = {}
    for key, value in row.items():
        formatted[key] = f"{value:.4f}" if isinstance(value, float) else value
    return formatted


def aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    paired_success = [row for row in rows if row["multilevel_success"] and row["single_success"]]
    speedups = [row["runtime_speedup"] for row in paired_success]
    expanded_ratios = [row["expanded_ratio"] for row in paired_success]
    return {
        "scenarios": len(rows),
        "paired_success_scenarios": len(paired_success),
        "multilevel_success_rate": float(np.mean([row["multilevel_success"] for row in rows])),
        "single_success_rate": float(np.mean([row["single_success"] for row in rows])),
        "mean_multilevel_runtime_ms": float(np.mean([row["multilevel_runtime_ms"] for row in rows])),
        "mean_single_runtime_ms": float(np.mean([row["single_runtime_ms"] for row in rows])),
        "mean_runtime_speedup": float(np.mean(speedups)) if speedups else 0.0,
        "mean_expanded_ratio": float(np.mean(expanded_ratios)) if expanded_ratios else 0.0,
        "mean_multilevel_max_risk": float(np.mean([row["multilevel_max_risk"] for row in rows])),
        "mean_multilevel_min_stability": float(np.mean([row["multilevel_min_stability"] for row in rows])),
        "mean_multilevel_feasible_rate": float(np.mean([row["multilevel_feasible_rate"] for row in rows])),
    }


def print_summary(output_dir: Path, rows: list[dict]) -> None:
    aggregate = aggregate_rows(rows)
    table = Table(title="Multi-Scenario Benchmark")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in aggregate.items():
        table.add_row(key, f"{value:.3f}" if isinstance(value, float) else str(value))
    Console().print(table)
    Console().print(f"[green]Wrote benchmark outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
