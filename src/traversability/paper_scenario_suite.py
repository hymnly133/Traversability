from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.ndt_map import NDTConfig, NDTImplicitMap, adaptive_ndt_config
from traversability.ndt_planner import plan_ndt_global
from traversability.paper_pipeline_demo import sample_points


@dataclass(frozen=True)
class Scenario:
    name: str
    height: np.ndarray
    obstacle: np.ndarray
    resolution: float
    origin: tuple[float, float]
    start: tuple[float, float, float]
    goal: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style rough-terrain scenario suite.")
    parser.add_argument("--output", type=Path, default=Path("runs/paper_scenarios"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_scenario(scenario, args.output) for scenario in make_scenarios()]
    write_summary(args.output, rows)
    print_summary(args.output, rows)


def run_scenario(scenario: Scenario, output_dir: Path) -> dict:
    points = sample_points(scenario.height, scenario.obstacle, scenario.resolution, scenario.origin)
    ndt_map = NDTImplicitMap(points, ndt_config(points, scenario.resolution))
    result = plan_ndt_global(ndt_map, scenario.start, scenario.goal)
    metric_summary = summarize_metrics(ndt_map)
    plot_scenario(output_dir, scenario, result.path_xyz)
    return {
        "scenario": scenario.name,
        "success": int(result.success),
        "runtime_ms": result.runtime_ms,
        "expanded_nodes": result.expanded_nodes,
        "path_length_m": result.path_length_m,
        "max_traversal_cost": result.max_traversal_cost,
        "mean_traversal_cost": result.mean_traversal_cost,
        "traversable_voxels": result.traversable_voxels,
        "connected_components": result.connected_components,
        "path_points": len(result.path_xyz),
        **metric_summary,
    }


def ndt_config(points: np.ndarray, resolution: float) -> NDTConfig:
    return adaptive_ndt_config(
        points,
        resolution,
        min_voxel_multiplier=2.0,
        density_multiplier=3.8,
        max_voxel_size=0.26,
        vertical_multiplier=0.78,
        min_vertical_multiplier=1.15,
        fusion_multiplier=3.0,
        complexity_threshold=0.96,
        robot_radius=0.30,
        robot_height=0.58,
    )


def make_scenarios() -> list[Scenario]:
    return [
        terrain_stairs(),
        terrain_rubble(),
        terrain_grass(),
        terrain_hill(),
        terrain_bridge(),
        terrain_field(),
    ]


def terrain_base(name: str, size: int = 96, resolution: float = 0.060):
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    xs = origin[0] + np.arange(size) * resolution
    ys = origin[1] + np.arange(size) * resolution
    xx, yy = np.meshgrid(xs, ys)
    height = 0.045 * np.sin(2.1 * xx) + 0.035 * np.cos(2.4 * yy) + 0.025 * np.sin(5.6 * (xx - 0.25 * yy))
    obstacle = np.zeros((size, size), dtype=bool)
    start = (origin[0] + 0.50, origin[1] + 0.55, 0.0)
    goal = (origin[0] + size * resolution - 0.58, origin[1] + size * resolution - 0.58, 0.0)
    return xx, yy, height, obstacle, origin, resolution, start, goal


def terrain_stairs() -> Scenario:
    xx, yy, height, obstacle, origin, resolution, start, goal = terrain_base("stairs")
    stair_band = (xx > -0.62) & (xx < 0.75) & (yy > -1.95) & (yy < 1.85)
    height += np.where(stair_band, 0.105 * np.floor((yy + 1.95) / 0.18), 0.0)
    cross_steps = (xx > 0.85) & (xx < 1.55) & (yy > -1.55) & (yy < 1.20)
    height += np.where(cross_steps, 0.075 * np.floor((xx - 0.85) / 0.16), 0.0)
    obstacle |= (np.abs(xx - 0.10) < 0.09) & (yy > -1.35) & (yy < 1.10)
    obstacle |= ((xx + 1.10) ** 2 / 0.055 + ((yy - 0.65) ** 2) / 0.16) < 1.0
    height[obstacle] += 0.72
    return Scenario("stairs", height, obstacle, resolution, origin, start, goal)


def terrain_rubble() -> Scenario:
    xx, yy, height, obstacle, origin, resolution, start, goal = terrain_base("rubble")
    rng = np.random.default_rng(5)
    for _ in range(58):
        cx, cy = rng.uniform(-2.35, 2.35), rng.uniform(-2.15, 2.25)
        radius = rng.uniform(0.04, 0.18)
        bump = np.exp(-(((xx - cx) / radius) ** 2 + ((yy - cy) / (radius * 0.75)) ** 2))
        height += rng.uniform(0.06, 0.34) * bump
        obstacle |= bump > 0.84
    height += 0.035 * rng.normal(size=height.shape)
    corridor = np.abs(yy - (0.45 * xx - 0.10)) < 0.24
    obstacle &= ~corridor
    return Scenario("rubble", height, obstacle, resolution, origin, start, goal)


def terrain_grass() -> Scenario:
    xx, yy, height, obstacle, origin, resolution, start, goal = terrain_base("grass")
    texture = 0.085 * np.sin(24.0 * xx + 5.0 * np.sin(yy)) * np.cos(18.0 * yy)
    texture += 0.040 * np.sin(41.0 * (xx + 0.30 * yy))
    tall_patch = ((xx + 0.45) ** 2 / 1.10 + ((yy - 0.35) ** 2) / 0.50) < 1.0
    second_patch = ((xx - 1.15) ** 2 / 0.50 + ((yy + 0.95) ** 2) / 0.38) < 1.0
    height += np.where(tall_patch | second_patch, texture + 0.065, 0.42 * texture)
    obstacle |= ((xx - 0.45) ** 2 / 0.045 + ((yy + 0.35) ** 2) / 0.09) < 1.0
    obstacle |= ((xx + 1.55) ** 2 / 0.060 + ((yy - 1.00) ** 2) / 0.12) < 1.0
    height[obstacle] += 0.58
    return Scenario("grass", height, obstacle, resolution, origin, start, goal)


def terrain_hill() -> Scenario:
    xx, yy, height, obstacle, origin, resolution, start, goal = terrain_base("hill")
    height += 0.72 * np.exp(-(((xx + 0.45) / 1.15) ** 2 + ((yy - 0.30) / 1.25) ** 2))
    height += 0.36 * np.exp(-(((xx - 1.20) / 0.78) ** 2 + ((yy - 1.05) / 0.85) ** 2))
    height -= 0.46 * np.exp(-(((xx - 0.88) / 0.38) ** 2 + ((yy + 0.55) / 0.48) ** 2))
    height -= 0.28 * np.exp(-(((xx + 1.28) / 0.42) ** 2 + ((yy + 1.15) / 0.52) ** 2))
    obstacle |= ((xx + 0.15) ** 2 / 0.05 + ((yy + 0.75) ** 2) / 0.24) < 1.0
    obstacle |= (np.abs(yy - 0.25 * xx - 0.65) < 0.065) & (xx > -1.80) & (xx < 1.65)
    height[obstacle] += 0.72
    return Scenario("hill", height, obstacle, resolution, origin, start, goal)


def terrain_bridge() -> Scenario:
    xx, yy, height, obstacle, origin, resolution, start, goal = terrain_base("bridge")
    trench = (np.abs(yy) < 0.46) & (xx > -2.35) & (xx < 2.35)
    bridge = (np.abs(yy) < 0.15) & (xx > -0.70) & (xx < 0.70)
    side_bridge = (np.abs(yy + 0.26) < 0.12) & (xx > 1.15) & (xx < 1.85)
    height -= np.where(trench & ~(bridge | side_bridge), 0.58, 0.0)
    height -= np.where((np.abs(yy - 1.18) < 0.20) & (xx > -2.0) & (xx < 1.6), 0.26, 0.0)
    obstacle |= ((xx - 0.2) ** 2 / 0.045 + ((yy - 0.92) ** 2) / 0.08) < 1.0
    obstacle |= ((xx + 1.45) ** 2 / 0.06 + ((yy + 0.88) ** 2) / 0.09) < 1.0
    height[obstacle] += 0.78
    return Scenario("bridge", height, obstacle, resolution, origin, start, goal)


def terrain_field() -> Scenario:
    xx, yy, height, obstacle, origin, resolution, start, goal = terrain_base("field", size=116, resolution=0.065)
    start = (-3.25, -2.85, 0.0)
    goal = (3.05, 2.85, 0.0)
    height += 0.13 * np.sin(4.4 * xx + 0.7 * np.cos(yy)) * np.cos(3.2 * yy)
    height += 0.08 * np.sin(9.5 * (xx - 0.35 * yy))

    low_stairs = (xx > -2.15) & (xx < -0.80) & (yy > -0.35) & (yy < 2.05)
    height += np.where(low_stairs, 0.070 * np.floor((yy + 0.35) / 0.18), 0.0)

    high_stairs = (xx > 0.72) & (xx < 1.65) & (yy > -2.05) & (yy < -0.20)
    height += np.where(high_stairs, 0.135 * np.floor((yy + 2.05) / 0.20), 0.0)

    ramp = (xx > -0.45) & (xx < 0.95) & (yy > 0.18) & (yy < 2.05)
    height += np.where(ramp, 0.56 * np.clip((yy - 0.18) / 1.87, 0.0, 1.0), 0.0)
    trench = (np.abs(yy + 0.52 * xx - 0.48) < 0.22) & (xx > -2.85) & (xx < 2.65)
    bridge = (np.abs(yy + 0.52 * xx - 0.48) < 0.09) & (xx > -0.15) & (xx < 0.75)
    height -= np.where(trench & ~bridge, 0.34, 0.0)

    rng = np.random.default_rng(19)
    for _ in range(72):
        cx, cy = rng.uniform(-3.20, 3.05), rng.uniform(-2.95, 2.95)
        radius = rng.uniform(0.04, 0.14)
        bump = np.exp(-(((xx - cx) / radius) ** 2 + ((yy - cy) / (radius * rng.uniform(0.7, 1.4))) ** 2))
        height += rng.uniform(0.05, 0.30) * bump
        obstacle |= bump > 0.88

    obstacle |= ((xx + 0.25) ** 2 / 0.07 + ((yy + 0.95) ** 2) / 0.10) < 1.0
    obstacle |= ((xx - 1.95) ** 2 / 0.06 + ((yy - 0.72) ** 2) / 0.22) < 1.0
    obstacle |= (np.abs(xx + 0.05) < 0.11) & (yy > -1.75) & (yy < -0.25)
    obstacle |= (np.abs(yy - 0.15 * xx - 1.72) < 0.08) & (xx > -2.80) & (xx < 2.20)
    corridor = np.abs(yy - (0.82 * xx - 0.10)) < 0.42
    obstacle &= ~corridor
    height = np.where(
        corridor,
        0.07 * np.sin(2.6 * xx) + 0.04 * np.cos(3.1 * yy) + 0.14 * np.clip((xx + 3.5) / 6.8, 0.0, 1.0),
        height,
    )
    height[obstacle] += 0.72
    return Scenario("field", height, obstacle, resolution, origin, start, goal)


def summarize_metrics(ndt_map: NDTImplicitMap) -> dict:
    metrics = list(ndt_map.compute_metrics().values())
    finite = [metric for metric in metrics if np.isfinite(metric.traversal_cost)]
    return {
        "mean_roughness": float(np.mean([metric.roughness for metric in metrics])) if metrics else float("nan"),
        "mean_slope_rad": float(np.mean([metric.slope for metric in metrics])) if metrics else float("nan"),
        "mean_sparsity": float(np.mean([metric.sparsity for metric in metrics])) if metrics else float("nan"),
        "mean_complexity": float(np.mean([metric.complexity for metric in metrics])) if metrics else float("nan"),
        "terrain_risk_voxels": int(sum(metric.terrain_risk for metric in metrics)),
        "collision_risk_voxels": int(sum(metric.collision_risk for metric in metrics)),
        "falling_risk_voxels": int(sum(metric.falling_risk for metric in metrics)),
        "finite_cost_voxels": len(finite),
    }


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    with (output_dir / "paper_scenarios.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(value) for key, value in row.items()})
    aggregate = aggregate_rows(rows)
    with (output_dir / "paper_scenarios_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in aggregate.items():
            writer.writerow([key, format_value(value)])


def aggregate_rows(rows: list[dict]) -> dict:
    return {
        "scenarios": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in rows])),
        "mean_expanded_nodes": float(np.mean([row["expanded_nodes"] for row in rows])),
        "mean_path_length_m": float(np.mean([row["path_length_m"] for row in rows])),
        "max_traversal_cost": float(max(row["max_traversal_cost"] for row in rows)),
        "mean_traversable_voxels": float(np.mean([row["traversable_voxels"] for row in rows])),
        "mean_roughness": float(np.mean([row["mean_roughness"] for row in rows])),
        "mean_slope_rad": float(np.mean([row["mean_slope_rad"] for row in rows])),
        "mean_sparsity": float(np.mean([row["mean_sparsity"] for row in rows])),
        "mean_complexity": float(np.mean([row["mean_complexity"] for row in rows])),
    }


def format_value(value):
    return f"{value:.4f}" if isinstance(value, float) else value


def plot_scenario(output_dir: Path, scenario: Scenario, path_xyz: list[tuple[float, float, float]]) -> None:
    extent = [
        scenario.origin[0],
        scenario.origin[0] + scenario.resolution * scenario.height.shape[1],
        scenario.origin[1],
        scenario.origin[1] + scenario.resolution * scenario.height.shape[0],
    ]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.imshow(scenario.height, cmap="terrain", origin="lower", extent=extent)
    ax.contour(scenario.obstacle.astype(float), levels=[0.5], colors="black", origin="lower", extent=extent)
    if path_xyz:
        xs = [point[0] for point in path_xyz]
        ys = [point[1] for point in path_xyz]
        ax.plot(xs, ys, color="#ffffff", linewidth=2.0, label="NDT global")
    ax.scatter([scenario.start[0]], [scenario.start[1]], color="#2ca02c", s=45, label="start")
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], color="#d62728", marker="*", s=65, label="goal")
    ax.set_title(scenario.name)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.86)
    fig.savefig(output_dir / f"{scenario.name}.png", dpi=160)
    plt.close(fig)


def print_summary(output_dir: Path, rows: list[dict]) -> None:
    table = Table(title="Paper Terrain Scenario Suite")
    table.add_column("Scenario")
    table.add_column("Success", justify="right")
    table.add_column("Runtime", justify="right")
    table.add_column("Path", justify="right")
    table.add_column("Max cost", justify="right")
    table.add_column("Complexity", justify="right")
    for row in rows:
        table.add_row(
            row["scenario"],
            str(bool(row["success"])),
            f"{row['runtime_ms']:.1f} ms",
            f"{row['path_length_m']:.2f} m",
            f"{row['max_traversal_cost']:.3f}",
            f"{row['mean_complexity']:.3f}",
        )
    Console().print(table)
    Console().print(f"[green]Wrote paper scenario outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
