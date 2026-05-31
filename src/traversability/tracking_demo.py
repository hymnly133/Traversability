from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.implicit_map import ImplicitTerrainMap
from traversability.planner import plan_multilevel
from traversability.terrain import generate_large_rough_terrain, make_pyramid
from traversability.tracking import simulate_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a multilevel terrain-aware plan with a pure-pursuit robot model.")
    parser.add_argument("--output", type=Path, default=Path("runs/tracking"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--levels", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    height, obstacle, origin = generate_large_rough_terrain(args.size, args.resolution, args.seed)
    pyramid = make_pyramid(height, obstacle, args.resolution, origin, levels=args.levels)
    start = (origin[0] + 2.5, origin[1] + 3.0)
    goal = (origin[0] + args.size * args.resolution - 3.0, origin[1] + args.size * args.resolution - 4.0)
    plan = plan_multilevel(pyramid, start, goal)
    terrain_map = ImplicitTerrainMap(pyramid.finest)
    tracking = simulate_tracking(terrain_map, plan.path_xy)

    write_outputs(args.output, plan, tracking)
    plot_outputs(args.output, pyramid.finest, plan.path_xy, tracking, start, goal)
    print_summary(args.output, plan, tracking)


def write_outputs(output_dir: Path, plan, tracking) -> None:
    with (output_dir / "tracking_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "plan_success",
                "tracking_success",
                "duration_s",
                "mean_error_m",
                "max_error_m",
                "final_error_m",
                "mean_risk",
                "max_risk",
                "min_stability",
                "feasible_rate",
            ]
        )
        writer.writerow(
            [
                int(plan.success),
                int(tracking.success),
                f"{tracking.duration_s:.3f}",
                f"{tracking.mean_error_m:.4f}",
                f"{tracking.max_error_m:.4f}",
                f"{tracking.final_error_m:.4f}",
                f"{tracking.mean_risk:.4f}",
                f"{tracking.max_risk:.4f}",
                f"{tracking.min_stability:.4f}",
                f"{tracking.feasible_rate:.4f}",
            ]
        )

    with (output_dir / "tracking_states.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "x", "y", "yaw", "target_index", "cross_track_error", "risk", "stability", "feasible"])
        for state in tracking.states:
            writer.writerow(
                [
                    f"{state.t:.3f}",
                    f"{state.x:.4f}",
                    f"{state.y:.4f}",
                    f"{state.yaw:.4f}",
                    state.target_index,
                    f"{state.cross_track_error:.4f}",
                    f"{state.risk:.4f}",
                    f"{state.stability:.4f}",
                    int(state.feasible),
                ]
            )


def plot_outputs(output_dir: Path, layer, path, tracking, start: tuple[float, float], goal: tuple[float, float]) -> None:
    extent = [
        layer.origin_xy[0],
        layer.origin_xy[0] + layer.resolution * layer.height.shape[1],
        layer.origin_xy[1],
        layer.origin_xy[1] + layer.resolution * layer.height.shape[0],
    ]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(layer.risk), cmap="magma", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    ax.contour(layer.obstacle.astype(float), levels=[0.5], colors="white", origin="lower", extent=extent)
    if path:
        px, py = zip(*path)
        ax.plot(px, py, color="#ffffff", linewidth=1.8, linestyle="--", label="planned")
    if tracking.states:
        tx = [state.x for state in tracking.states]
        ty = [state.y for state in tracking.states]
        ax.plot(tx, ty, color="#1f77b4", linewidth=2.6, label="tracked")
    ax.scatter([start[0]], [start[1]], color="#2ca02c", s=55, label="start")
    ax.scatter([goal[0]], [goal[1]], color="#ffdf4d", edgecolor="#111111", s=70, marker="*", label="goal")
    ax.set_title("Closed-loop path tracking on terrain risk map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.86)
    fig.savefig(output_dir / "tracking.png", dpi=170)
    plt.close(fig)


def print_summary(output_dir: Path, plan, tracking) -> None:
    table = Table(title="Path Tracking")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("plan success", str(plan.success))
    table.add_row("tracking success", str(tracking.success))
    table.add_row("duration", f"{tracking.duration_s:.1f} s")
    table.add_row("mean error", f"{tracking.mean_error_m:.2f} m")
    table.add_row("max error", f"{tracking.max_error_m:.2f} m")
    table.add_row("final error", f"{tracking.final_error_m:.2f} m")
    table.add_row("max risk", f"{tracking.max_risk:.3f}")
    table.add_row("min stability", f"{tracking.min_stability:.3f}")
    table.add_row("feasible rate", f"{tracking.feasible_rate:.1%}")
    Console().print(table)
    Console().print(f"[green]Wrote tracking outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
