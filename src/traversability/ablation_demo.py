from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.implicit_map import ImplicitTerrainMap
from traversability.planner import (
    PlannerWeights,
    astar,
    evaluate_stability,
    nearest_free,
    plan_multilevel,
    polyline_length,
    shortcut_path,
)
from traversability.terrain import generate_large_rough_terrain, grid_to_world, make_pyramid, world_to_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multilevel planning against a single-level baseline.")
    parser.add_argument("--output", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=180)
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

    multilevel = plan_multilevel(pyramid, start, goal, weights=PlannerWeights(enable_optimization=False))
    single = plan_single_level(pyramid.finest, start, goal)

    write_outputs(args.output, multilevel, single)
    plot_outputs(args.output, pyramid.finest, multilevel.path_xy, single["path_xy"], start, goal)
    print_summary(args.output, multilevel, single)


def plan_single_level(layer, start: tuple[float, float], goal: tuple[float, float]) -> dict:
    begin = time.perf_counter()
    weights = PlannerWeights()
    start_cell = nearest_free(layer, world_to_grid(layer, start), allowed_mask=None)
    goal_cell = nearest_free(layer, world_to_grid(layer, goal), allowed_mask=None)
    if start_cell is None or goal_cell is None:
        return empty_single_result(begin)

    cells, expanded = astar(layer, start_cell, goal_cell, weights)
    if not cells:
        return empty_single_result(begin, expanded)

    terrain_map = ImplicitTerrainMap(layer)
    raw_path = [grid_to_world(layer, cell) for cell in cells]
    path = shortcut_path(terrain_map, raw_path, max_risk=0.92)
    samples = evaluate_stability(terrain_map, path)
    risks = np.array([sample.risk for sample in samples], dtype=np.float64)
    stabilities = np.array([sample.stability for sample in samples], dtype=np.float64)
    feasible_rate = float(np.mean([sample.feasible for sample in samples]))
    runtime_ms = (time.perf_counter() - begin) * 1000.0

    return {
        "name": "single_level",
        "path_xy": path,
        "success": bool(
            np.all(np.isfinite(risks))
            and np.max(risks) <= weights.lethal_risk
            and np.min(stabilities) > 0.08
            and feasible_rate >= 0.80
        ),
        "runtime_ms": runtime_ms,
        "expanded_nodes": expanded,
        "path_length_m": polyline_length(path),
        "mean_risk": float(np.mean(risks)),
        "max_risk": float(np.max(risks)),
        "min_stability": float(np.min(stabilities)),
        "feasible_rate": feasible_rate,
    }


def empty_single_result(begin: float, expanded: int = 0) -> dict:
    return {
        "name": "single_level",
        "path_xy": [],
        "success": False,
        "runtime_ms": (time.perf_counter() - begin) * 1000.0,
        "expanded_nodes": expanded,
        "path_length_m": 0.0,
        "mean_risk": float("inf"),
        "max_risk": float("inf"),
        "min_stability": 0.0,
        "feasible_rate": 0.0,
    }


def write_outputs(output_dir: Path, multilevel, single: dict) -> None:
    with (output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "planner",
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
        writer.writerow(planner_row("multilevel", multilevel))
        writer.writerow(planner_row("single_level", single))


def planner_row(name: str, result) -> list:
    if isinstance(result, dict):
        return [
            name,
            int(result["success"]),
            f"{result['runtime_ms']:.3f}",
            result["expanded_nodes"],
            f"{result['path_length_m']:.3f}",
            f"{result['mean_risk']:.4f}",
            f"{result['max_risk']:.4f}",
            f"{result['min_stability']:.4f}",
            f"{result['feasible_rate']:.4f}",
        ]
    return [
        name,
        int(result.success),
        f"{result.runtime_ms:.3f}",
        result.expanded_nodes,
        f"{result.path_length_m:.3f}",
        f"{result.mean_risk:.4f}",
        f"{result.max_risk:.4f}",
        f"{result.min_stability:.4f}",
        f"{result.feasible_rate:.4f}",
    ]


def plot_outputs(
    output_dir: Path,
    layer,
    multilevel_path: list[tuple[float, float]],
    single_path: list[tuple[float, float]],
    start: tuple[float, float],
    goal: tuple[float, float],
) -> None:
    extent = [
        layer.origin_xy[0],
        layer.origin_xy[0] + layer.resolution * layer.height.shape[1],
        layer.origin_xy[1],
        layer.origin_xy[1] + layer.resolution * layer.height.shape[0],
    ]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(layer.risk), cmap="magma", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    ax.contour(layer.obstacle.astype(float), levels=[0.5], colors="white", origin="lower", extent=extent)
    if multilevel_path:
        mx, my = zip(*multilevel_path)
        ax.plot(mx, my, color="#1f77b4", linewidth=2.5, label="multilevel")
    if single_path:
        sx, sy = zip(*single_path)
        ax.plot(sx, sy, color="#2ca02c", linewidth=1.8, linestyle="--", label="single-level")
    ax.scatter([start[0]], [start[1]], color="#ffffff", edgecolor="#111111", s=55, label="start")
    ax.scatter([goal[0]], [goal[1]], color="#ffdf4d", edgecolor="#111111", s=70, marker="*", label="goal")
    ax.set_title("Planner ablation on terrain risk map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.86)
    fig.savefig(output_dir / "ablation.png", dpi=170)
    plt.close(fig)


def print_summary(output_dir: Path, multilevel, single: dict) -> None:
    speedup = single["runtime_ms"] / max(multilevel.runtime_ms, 1e-9)
    expansion_ratio = single["expanded_nodes"] / max(multilevel.expanded_nodes, 1)
    table = Table(title="Planning Ablation")
    table.add_column("Metric")
    table.add_column("Multilevel", justify="right")
    table.add_column("Single-level", justify="right")
    table.add_row("success", str(multilevel.success), str(single["success"]))
    table.add_row("runtime", f"{multilevel.runtime_ms:.1f} ms", f"{single['runtime_ms']:.1f} ms")
    table.add_row("expanded", str(multilevel.expanded_nodes), str(single["expanded_nodes"]))
    table.add_row("path length", f"{multilevel.path_length_m:.2f} m", f"{single['path_length_m']:.2f} m")
    table.add_row("max risk", f"{multilevel.max_risk:.3f}", f"{single['max_risk']:.3f}")
    table.add_row("feasible rate", f"{multilevel.feasible_rate:.1%}", f"{single['feasible_rate']:.1%}")
    table.add_row("runtime speedup", f"{speedup:.2f}x", "")
    table.add_row("expanded ratio", f"{expansion_ratio:.2f}x", "")
    Console().print(table)
    Console().print(f"[green]Wrote ablation outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
