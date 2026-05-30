from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.planner import PlanningResult, plan_multilevel
from traversability.terrain import generate_large_rough_terrain, make_pyramid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-window terrain-aware replanning benchmark.")
    parser.add_argument("--output", type=Path, default=Path("runs/realtime"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--step-distance", type=float, default=4.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    height, obstacle, origin = generate_large_rough_terrain(args.size, args.resolution, args.seed)
    start = (origin[0] + 2.5, origin[1] + 3.0)
    goal = (origin[0] + args.size * args.resolution - 3.0, origin[1] + args.size * args.resolution - 4.0)

    current = start
    trajectory = [current]
    results: list[PlanningResult] = []
    dynamic_obstacle = obstacle.copy()

    for cycle in range(args.cycles):
        if cycle == args.cycles // 2:
            add_dynamic_block(dynamic_obstacle, origin, args.resolution, center=(0.5, 5.5), radius_m=1.1)
        pyramid = make_pyramid(height, dynamic_obstacle, args.resolution, origin, levels=args.levels)
        result = plan_multilevel(pyramid, current, goal)
        results.append(result)
        if not result.success or len(result.path_xy) < 2:
            break
        current = advance_along_path(result.path_xy, args.step_distance)
        trajectory.append(current)
        if math.dist(current, goal) < args.step_distance:
            trajectory.append(goal)
            break

    write_outputs(args.output, results, trajectory)
    plot_outputs(args.output, height, dynamic_obstacle, origin, args.resolution, results, trajectory, goal)
    print_summary(args.output, results, trajectory, goal)


def add_dynamic_block(
    obstacle: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    center: tuple[float, float],
    radius_m: float,
) -> None:
    rows, cols = obstacle.shape
    xs = origin[0] + np.arange(cols) * resolution
    ys = origin[1] + np.arange(rows) * resolution
    xx, yy = np.meshgrid(xs, ys)
    obstacle |= (xx - center[0]) ** 2 + (yy - center[1]) ** 2 < radius_m**2


def advance_along_path(path: list[tuple[float, float]], distance: float) -> tuple[float, float]:
    remaining = distance
    for start, end in zip(path[:-1], path[1:]):
        segment = math.dist(start, end)
        if segment >= remaining:
            ratio = remaining / max(segment, 1e-9)
            return (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)
        remaining -= segment
    return path[-1]


def write_outputs(output_dir: Path, results: list[PlanningResult], trajectory: list[tuple[float, float]]) -> None:
    with (output_dir / "replanning_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "cycle",
                "success",
                "runtime_ms",
                "expanded_nodes",
                "path_length_m",
                "mean_risk",
                "max_risk",
                "min_stability",
                "feasible_rate",
            ]
        )
        for cycle, result in enumerate(results):
            writer.writerow(
                [
                    cycle,
                    int(result.success),
                    f"{result.runtime_ms:.3f}",
                    result.expanded_nodes,
                    f"{result.path_length_m:.3f}",
                    f"{result.mean_risk:.4f}",
                    f"{result.max_risk:.4f}",
                    f"{result.min_stability:.4f}",
                    f"{result.feasible_rate:.4f}",
                ]
            )

    with (output_dir / "executed_trajectory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for x, y in trajectory:
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])


def plot_outputs(
    output_dir: Path,
    height: np.ndarray,
    obstacle: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    results: list[PlanningResult],
    trajectory: list[tuple[float, float]],
    goal: tuple[float, float],
) -> None:
    extent = [origin[0], origin[0] + resolution * height.shape[1], origin[1], origin[1] + resolution * height.shape[0]]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(height, cmap="terrain", origin="lower", extent=extent)
    ax.contour(obstacle.astype(float), levels=[0.5], colors="black", origin="lower", extent=extent)
    for idx, result in enumerate(results):
        if result.path_xy:
            xs, ys = zip(*result.path_xy)
            ax.plot(xs, ys, linewidth=1.1, alpha=0.35, label="planned" if idx == 0 else None)
    if trajectory:
        tx, ty = zip(*trajectory)
        ax.plot(tx, ty, color="#1f77b4", linewidth=3.0, marker="o", label="executed")
        ax.scatter([trajectory[0][0]], [trajectory[0][1]], color="#2ca02c", s=55, label="start")
    ax.scatter([goal[0]], [goal[1]], color="#d62728", s=65, marker="*", label="goal")
    ax.set_title("Rolling-window replanning")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.86)
    fig.savefig(output_dir / "replanning.png", dpi=170)
    plt.close(fig)


def print_summary(
    output_dir: Path,
    results: list[PlanningResult],
    trajectory: list[tuple[float, float]],
    goal: tuple[float, float],
) -> None:
    runtimes = [result.runtime_ms for result in results]
    success_rate = np.mean([result.success for result in results]) if results else 0.0
    table = Table(title="Rolling Replanning Benchmark")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("cycles", str(len(results)))
    table.add_row("success rate", f"{success_rate:.1%}")
    table.add_row("mean runtime", f"{np.mean(runtimes):.1f} ms" if runtimes else "nan")
    table.add_row("p95 runtime", f"{np.percentile(runtimes, 95):.1f} ms" if runtimes else "nan")
    table.add_row("final distance", f"{math.dist(trajectory[-1], goal):.2f} m" if trajectory else "nan")
    Console().print(table)
    Console().print(f"[green]Wrote realtime benchmark outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
