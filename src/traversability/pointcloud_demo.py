from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.planner import plan_multilevel
from traversability.pointcloud import point_cloud_to_elevation_grid, sample_point_cloud_from_grid
from traversability.terrain import generate_large_rough_terrain, make_pyramid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan from a point-cloud-derived elevation map.")
    parser.add_argument("--output", type=Path, default=Path("runs/pointcloud"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--levels", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    height, obstacle, origin = generate_large_rough_terrain(args.size, args.resolution, args.seed)
    points = sample_point_cloud_from_grid(height, obstacle, args.resolution, origin, seed=args.seed)
    grid = point_cloud_to_elevation_grid(points, args.resolution)
    pyramid = make_pyramid(grid.height, grid.obstacle, grid.resolution, grid.origin_xy, levels=args.levels)

    start = (grid.origin_xy[0] + 2.5, grid.origin_xy[1] + 3.0)
    goal = (grid.origin_xy[0] + grid.height.shape[1] * grid.resolution - 3.0, grid.origin_xy[1] + grid.height.shape[0] * grid.resolution - 4.0)
    result = plan_multilevel(pyramid, start, goal)

    write_outputs(args.output, result, points, grid)
    plot_outputs(args.output, grid, result.path_xy, start, goal)
    print_summary(args.output, result, points, grid)


def write_outputs(output_dir: Path, result, points: np.ndarray, grid) -> None:
    with (output_dir / "pointcloud_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "success",
                "points",
                "grid_rows",
                "grid_cols",
                "observed_cells",
                "runtime_ms",
                "expanded_nodes",
                "path_length_m",
                "max_risk",
                "min_stability",
                "feasible_rate",
            ]
        )
        writer.writerow(
            [
                int(result.success),
                len(points),
                grid.height.shape[0],
                grid.height.shape[1],
                int(np.count_nonzero(grid.points_per_cell)),
                f"{result.runtime_ms:.3f}",
                result.expanded_nodes,
                f"{result.path_length_m:.3f}",
                f"{result.max_risk:.4f}",
                f"{result.min_stability:.4f}",
                f"{result.feasible_rate:.4f}",
            ]
        )


def plot_outputs(output_dir: Path, grid, path: list[tuple[float, float]], start: tuple[float, float], goal: tuple[float, float]) -> None:
    extent = [
        grid.origin_xy[0],
        grid.origin_xy[0] + grid.resolution * grid.height.shape[1],
        grid.origin_xy[1],
        grid.origin_xy[1] + grid.resolution * grid.height.shape[0],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    height_image = axes[0].imshow(grid.height, cmap="terrain", origin="lower", extent=extent)
    axes[0].contour(grid.obstacle.astype(float), levels=[0.5], colors="black", origin="lower", extent=extent)
    axes[0].set_title("Point-cloud elevation map")
    fig.colorbar(height_image, ax=axes[0], label="height [m]")

    density_image = axes[1].imshow(grid.points_per_cell, cmap="viridis", origin="lower", extent=extent)
    axes[1].set_title("Points per grid cell")
    fig.colorbar(density_image, ax=axes[1], label="points")

    for axis in axes:
        if path:
            px, py = zip(*path)
            axis.plot(px, py, color="#1f77b4", linewidth=2.4, label="planned")
        axis.scatter([start[0]], [start[1]], color="#2ca02c", s=55, label="start")
        axis.scatter([goal[0]], [goal[1]], color="#d62728", s=65, marker="*", label="goal")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.legend(loc="upper left", framealpha=0.86)

    fig.savefig(output_dir / "pointcloud_plan.png", dpi=170)
    plt.close(fig)


def print_summary(output_dir: Path, result, points: np.ndarray, grid) -> None:
    table = Table(title="Point Cloud Planning")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("success", str(result.success))
    table.add_row("points", str(len(points)))
    table.add_row("grid", f"{grid.height.shape[0]} x {grid.height.shape[1]}")
    table.add_row("observed cells", str(int(np.count_nonzero(grid.points_per_cell))))
    table.add_row("runtime", f"{result.runtime_ms:.1f} ms")
    table.add_row("path length", f"{result.path_length_m:.2f} m")
    table.add_row("max risk", f"{result.max_risk:.3f}")
    table.add_row("feasible rate", f"{result.feasible_rate:.1%}")
    Console().print(table)
    Console().print(f"[green]Wrote point-cloud planning outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()

