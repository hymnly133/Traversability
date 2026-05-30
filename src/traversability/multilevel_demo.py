from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.planner import plan_multilevel
from traversability.terrain import generate_large_rough_terrain, make_pyramid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multilevel terrain-aware planning reproduction demo.")
    parser.add_argument("--output", type=Path, default=Path("runs/multilevel"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--levels", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    height, obstacle, origin = generate_large_rough_terrain(
        size=args.size,
        resolution=args.resolution,
        seed=args.seed,
    )
    pyramid = make_pyramid(height, obstacle, args.resolution, origin, levels=args.levels)

    start = (origin[0] + 2.5, origin[1] + 3.0)
    goal = (origin[0] + args.size * args.resolution - 3.0, origin[1] + args.size * args.resolution - 4.0)
    result = plan_multilevel(pyramid, start, goal)

    write_outputs(args.output, result)
    plot_outputs(args.output, pyramid.finest, result, start, goal)
    print_summary(result, args.output)


def write_outputs(output_dir: Path, result) -> None:
    with (output_dir / "planned_path.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for x, y in result.path_xy:
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])

    with (output_dir / "stability_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "slope", "roughness", "step", "risk", "stability"])
        for sample in result.samples:
            writer.writerow(
                [
                    f"{sample.x:.4f}",
                    f"{sample.y:.4f}",
                    f"{sample.slope:.4f}",
                    f"{sample.roughness:.4f}",
                    f"{sample.step:.4f}",
                    f"{sample.risk:.4f}",
                    f"{sample.stability:.4f}",
                ]
            )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "success",
                "runtime_ms",
                "expanded_nodes",
                "path_length_m",
                "mean_risk",
                "max_risk",
                "min_stability",
            ]
        )
        writer.writerow(
            [
                int(result.success),
                f"{result.runtime_ms:.3f}",
                result.expanded_nodes,
                f"{result.path_length_m:.3f}",
                f"{result.mean_risk:.4f}",
                f"{result.max_risk:.4f}",
                f"{result.min_stability:.4f}",
            ]
        )


def plot_outputs(output_dir: Path, layer, result, start: tuple[float, float], goal: tuple[float, float]) -> None:
    extent = [
        layer.origin_xy[0],
        layer.origin_xy[0] + layer.resolution * layer.height.shape[1],
        layer.origin_xy[1],
        layer.origin_xy[1] + layer.resolution * layer.height.shape[0],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    terrain_image = axes[0].imshow(layer.height, cmap="terrain", origin="lower", extent=extent)
    axes[0].contour(layer.obstacle.astype(float), levels=[0.5], colors="black", origin="lower", extent=extent)
    draw_path(axes[0], result, start, goal)
    axes[0].set_title("Height map and multilevel plan")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(terrain_image, ax=axes[0], label="height [m]")

    risk = np.ma.masked_invalid(layer.risk)
    risk_image = axes[1].imshow(risk, cmap="magma", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    axes[1].contour(layer.obstacle.astype(float), levels=[0.5], colors="white", origin="lower", extent=extent)
    draw_path(axes[1], result, start, goal)
    axes[1].set_title("Terrain risk map")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    fig.colorbar(risk_image, ax=axes[1], label="risk")

    fig.savefig(output_dir / "multilevel_plan.png", dpi=170)
    plt.close(fig)


def draw_path(axis, result, start: tuple[float, float], goal: tuple[float, float]) -> None:
    if result.coarse_path_xy:
        cx, cy = zip(*result.coarse_path_xy)
        axis.plot(cx, cy, color="#ffffff", linewidth=1.2, linestyle="--", alpha=0.85, label="coarse")
    if result.path_xy:
        px, py = zip(*result.path_xy)
        axis.plot(px, py, color="#1f77b4", linewidth=2.5, label="refined")
    axis.scatter([start[0]], [start[1]], color="#2ca02c", s=55, marker="o", label="start")
    axis.scatter([goal[0]], [goal[1]], color="#d62728", s=65, marker="*", label="goal")
    axis.legend(loc="upper left", framealpha=0.86)


def print_summary(result, output_dir: Path) -> None:
    table = Table(title="Multilevel Terrain-Aware Planning")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("success", str(result.success))
    table.add_row("runtime", f"{result.runtime_ms:.1f} ms")
    table.add_row("expanded nodes", str(result.expanded_nodes))
    table.add_row("path length", f"{result.path_length_m:.2f} m")
    table.add_row("mean risk", f"{result.mean_risk:.3f}")
    table.add_row("max risk", f"{result.max_risk:.3f}")
    table.add_row("min stability", f"{result.min_stability:.3f}")
    Console().print(table)
    Console().print(f"[green]Wrote reproduction outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()

