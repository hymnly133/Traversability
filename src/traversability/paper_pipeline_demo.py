from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.hybrid_local_planner import HybridLocalPlannerConfig, NDTLocalTraversabilityGuide, plan_hybrid_local
from traversability.implicit_map import ImplicitTerrainMap
from traversability.ndt_map import NDTConfig, NDTImplicitMap
from traversability.ndt_planner import plan_ndt_global
from traversability.terrain import analyze_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-style NDT global + Hybrid A* local planning pipeline.")
    parser.add_argument("--output", type=Path, default=Path("runs/paper_pipeline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_pipeline()
    write_outputs(args.output, result)
    plot_outputs(args.output, result)
    print_summary(args.output, result)


def run_pipeline() -> dict:
    height, obstacle, resolution, origin = make_pipeline_terrain()
    points = sample_points(height, obstacle, resolution, origin)
    ndt_map = NDTImplicitMap(
        points,
        NDTConfig(
            voxel_size=0.18,
            fusion_radius=0.36,
            saturation_count=2,
            slope_threshold_rad=np.deg2rad(35.0),
            complexity_threshold=0.82,
            robot_radius=0.28,
            robot_height=0.55,
        ),
    )
    start = (-1.55, -0.75, 0.0)
    goal = (1.55, 0.72, 0.0)
    global_result = plan_ndt_global(ndt_map, start, goal)
    global_xy = [(x, y) for x, y, _ in global_result.path_xyz]
    terrain_map = ImplicitTerrainMap(analyze_layer("paper_pipeline", height, obstacle, resolution, origin))
    traversability_guide = NDTLocalTraversabilityGuide.from_ndt_map(ndt_map)
    local_result = plan_hybrid_local(
        terrain_map,
        start=(global_xy[0][0], global_xy[0][1], 0.0),
        global_path=global_xy,
        config=HybridLocalPlannerConfig(
            step_length=0.22,
            local_window_radius=3.0,
            goal_tolerance=0.32,
            min_stability=0.28,
            max_iterations=7000,
            global_waypoint_limit=22,
        ),
        global_traversability=traversability_guide,
    )
    return {
        "height": height,
        "obstacle": obstacle,
        "resolution": resolution,
        "origin": origin,
        "points": points,
        "global": global_result,
        "local": local_result,
        "global_xy": global_xy,
        "shared_traversable_voxels": len(traversability_guide.traversable_keys),
    }


def make_pipeline_terrain() -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    resolution = 0.05
    size = 80
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    xs = origin[0] + np.arange(size) * resolution
    ys = origin[1] + np.arange(size) * resolution
    xx, yy = np.meshgrid(xs, ys)
    height = 0.04 * np.sin(2.2 * xx) + 0.03 * np.cos(2.0 * yy)
    height += 0.08 * np.exp(-(((xx + 0.9) / 0.32) ** 2 + ((yy - 0.55) / 0.24) ** 2))
    obstacle = (np.abs(xx) < 0.18) & (yy > -0.55) & (yy < 0.70)
    height = height.copy()
    height[obstacle] += 0.85
    return height.astype(np.float64), obstacle, resolution, origin


def sample_points(
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
) -> np.ndarray:
    rng = np.random.default_rng(41)
    rows, cols = height.shape
    xs = origin[0] + np.arange(cols) * resolution
    ys = origin[1] + np.arange(rows) * resolution
    points = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            samples = 2 + int(obstacle[row, col])
            jitter = rng.uniform(-0.42 * resolution, 0.42 * resolution, size=(samples, 2))
            z = height[row, col] + rng.normal(0.0, 0.012, size=samples)
            points.append(np.column_stack([x + jitter[:, 0], y + jitter[:, 1], z]))
    return np.vstack(points).astype(np.float64)


def write_outputs(output_dir: Path, result: dict) -> None:
    global_result = result["global"]
    local_result = result["local"]
    with (output_dir / "paper_pipeline_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "global_success",
                "local_success",
                "global_runtime_ms",
                "local_runtime_ms",
                "global_expanded",
                "local_expanded",
                "global_path_length_m",
                "local_path_length_m",
                "global_max_traversal_cost",
                "local_mean_risk",
                "local_min_stability",
                "shared_traversable_voxels",
            ]
        )
        writer.writerow(
            [
                int(global_result.success),
                int(local_result.success),
                f"{global_result.runtime_ms:.3f}",
                f"{local_result.runtime_ms:.3f}",
                global_result.expanded_nodes,
                local_result.expanded_nodes,
                f"{global_result.path_length_m:.3f}",
                f"{local_result.path_length_m:.3f}",
                f"{global_result.max_traversal_cost:.4f}",
                f"{local_result.mean_risk:.4f}",
                f"{local_result.min_stability:.4f}",
                result["shared_traversable_voxels"],
            ]
        )

    with (output_dir / "global_path.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for x, y, z in global_result.path_xyz:
            writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])

    with (output_dir / "local_path.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "yaw"])
        for x, y, yaw in local_result.path:
            writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{yaw:.4f}"])


def plot_outputs(output_dir: Path, result: dict) -> None:
    height = result["height"]
    obstacle = result["obstacle"]
    resolution = result["resolution"]
    origin = result["origin"]
    extent = [origin[0], origin[0] + resolution * height.shape[1], origin[1], origin[1] + resolution * height.shape[0]]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(height, cmap="terrain", origin="lower", extent=extent)
    ax.contour(obstacle.astype(float), levels=[0.5], colors="black", origin="lower", extent=extent)
    if result["global_xy"]:
        gx, gy = zip(*result["global_xy"])
        ax.plot(gx, gy, color="#ffffff", linewidth=1.8, linestyle="--", label="NDT global")
    if result["local"].path:
        lx = [state[0] for state in result["local"].path]
        ly = [state[1] for state in result["local"].path]
        ax.plot(lx, ly, color="#1f77b4", linewidth=2.5, label="Hybrid local")
    ax.set_title("Paper-style NDT global + Hybrid local pipeline")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.86)
    fig.savefig(output_dir / "paper_pipeline.png", dpi=170)
    plt.close(fig)


def print_summary(output_dir: Path, result: dict) -> None:
    table = Table(title="Paper-Style Planning Pipeline")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("global success", str(result["global"].success))
    table.add_row("local success", str(result["local"].success))
    table.add_row("global runtime", f"{result['global'].runtime_ms:.1f} ms")
    table.add_row("local runtime", f"{result['local'].runtime_ms:.1f} ms")
    table.add_row("global path length", f"{result['global'].path_length_m:.2f} m")
    table.add_row("local path length", f"{result['local'].path_length_m:.2f} m")
    table.add_row("local min stability", f"{result['local'].min_stability:.3f}")
    table.add_row("shared traversable voxels", str(result["shared_traversable_voxels"]))
    Console().print(table)
    Console().print(f"[green]Wrote paper pipeline outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
