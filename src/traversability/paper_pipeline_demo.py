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
from traversability.ndt_map import NDTImplicitMap, adaptive_ndt_config
from traversability.ndt_planner import plan_ndt_global
from traversability.pointcloud import point_cloud_to_elevation_grid
from traversability.terrain import TerrainLayer, analyze_layer, crop_local_window


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
    reference_height, reference_obstacle, resolution, origin = make_pipeline_terrain()
    points = sample_points(reference_height, reference_obstacle, resolution, origin)
    pointcloud_layer = point_cloud_layer_from_points("paper_pipeline_pointcloud", points, resolution)
    ndt_map = NDTImplicitMap(points, adaptive_ndt_config(points, resolution))
    start = (-2.25, -1.35, 0.0)
    goal = (2.25, 1.35, 0.0)
    global_result = plan_ndt_global(ndt_map, start, goal)
    global_xy = [(x, y) for x, y, _ in global_result.path_xyz]
    local_layer = crop_pointcloud_local_window(
        "paper_pipeline_local",
        points,
        resolution,
        center_xy=(global_xy[0][0], global_xy[0][1]),
        radius=3.0,
    )
    terrain_map = ImplicitTerrainMap(local_layer)
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
        "height": pointcloud_layer.height,
        "obstacle": pointcloud_layer.obstacle,
        "reference_height": reference_height,
        "reference_obstacle": reference_obstacle,
        "resolution": resolution,
        "origin": pointcloud_layer.origin_xy,
        "reference_origin": origin,
        "points": points,
        "global": global_result,
        "local": local_result,
        "global_xy": global_xy,
        "shared_traversable_voxels": len(traversability_guide.traversable_keys),
        "global_cells": int(pointcloud_layer.height.size),
        "local_cells": int(local_layer.height.size),
        "planning_map_source": "point_cloud_derived",
    }


def make_pipeline_terrain() -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    resolution = 0.05
    size = 112
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    xs = origin[0] + np.arange(size) * resolution
    ys = origin[1] + np.arange(size) * resolution
    xx, yy = np.meshgrid(xs, ys)
    rng = np.random.default_rng(43)
    height = (
        0.08 * np.sin(2.4 * xx)
        + 0.07 * np.cos(2.2 * yy)
        + 0.055 * np.sin(5.0 * (xx + 0.35 * yy))
        + 0.035 * np.cos(8.5 * (xx - yy))
    )
    height += 0.20 * np.exp(-(((xx + 1.25) / 0.42) ** 2 + ((yy - 0.82) / 0.30) ** 2))
    height -= 0.16 * np.exp(-(((xx - 1.15) / 0.32) ** 2 + ((yy + 0.72) / 0.26) ** 2))
    height += np.where((xx > -1.45) & (xx < -0.58) & (yy > -0.15) & (yy < 1.42), 0.065 * np.floor((yy + 0.15) / 0.18), 0.0)
    height += np.where((xx > 0.62) & (xx < 1.52) & (yy > -1.56) & (yy < -0.20), 0.085 * np.floor((yy + 1.56) / 0.20), 0.0)
    height += np.where((xx > -0.22) & (xx < 0.82) & (yy > 0.20) & (yy < 1.62), 0.44 * np.clip((yy - 0.20) / 1.42, 0.0, 1.0), 0.0)
    for _ in range(24):
        cx, cy = rng.uniform(-2.2, 2.1), rng.uniform(-2.0, 2.0)
        radius = rng.uniform(0.055, 0.18)
        bump = np.exp(-(((xx - cx) / radius) ** 2 + ((yy - cy) / (radius * rng.uniform(0.65, 1.35))) ** 2))
        height += rng.uniform(0.035, 0.20) * bump
    obstacle = (np.abs(xx) < 0.20) & (yy > -0.70) & (yy < 0.90)
    obstacle |= ((xx + 1.42) ** 2 / 0.060 + (yy + 0.72) ** 2 / 0.10) < 1.0
    obstacle |= ((xx - 1.48) ** 2 / 0.050 + (yy - 0.62) ** 2 / 0.15) < 1.0
    obstacle |= (np.abs(yy - (0.55 * xx + 0.10)) < 0.055) & (xx > -1.75) & (xx < 1.55)
    main_corridor = np.abs(yy - (0.62 * xx + 0.03)) < 0.46
    lower_bypass = (np.abs(yy - (0.28 * xx - 0.92)) < 0.38) & (xx > -1.65) & (xx < 0.60)
    upper_bypass = (np.abs(yy - (0.28 * xx + 0.92)) < 0.38) & (xx > -0.60) & (xx < 1.65)
    corridor = main_corridor | lower_bypass | upper_bypass
    corridor |= ((xx < -1.7) & (yy < -0.95)) | ((xx > 1.7) & (yy > 0.95))
    obstacle &= ~corridor
    obstacle |= (np.abs(xx) < 0.20) & (yy > -0.70) & (yy < 0.90)
    height = height.copy()
    corridor_height = 0.055 * np.sin(2.8 * xx) + 0.045 * np.cos(3.4 * yy) + 0.16 * np.clip((xx + 2.25) / 4.50, 0.0, 1.0)
    height = np.where(corridor, corridor_height, height)
    height[obstacle] += 1.05
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


def point_cloud_layer_from_points(
    name: str,
    points: np.ndarray,
    resolution: float,
) -> TerrainLayer:
    grid = point_cloud_to_elevation_grid(
        points,
        resolution=resolution,
        min_points_per_cell=1,
        obstacle_height_percentile=97.0,
        obstacle_relief_threshold=0.35,
    )
    return analyze_layer(name, grid.height, grid.obstacle, grid.resolution, grid.origin_xy)


def crop_pointcloud_local_window(
    name: str,
    points: np.ndarray,
    resolution: float,
    center_xy: tuple[float, float],
    radius: float,
) -> TerrainLayer:
    points = np.asarray(points, dtype=np.float64)
    offset = points[:, :2] - np.asarray(center_xy, dtype=np.float64)
    inside = np.linalg.norm(offset, axis=1) <= radius
    if not np.any(inside):
        raise ValueError("local point cloud window contains no points")
    layer = point_cloud_layer_from_points(name, points[inside], resolution)
    return crop_local_window(
        name,
        layer.height,
        layer.obstacle,
        layer.resolution,
        layer.origin_xy,
        center_xy=center_xy,
        radius=radius,
    )


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
                "local_raw_path_length_m",
                "local_path_states",
                "local_raw_path_states",
                "local_waypoint_states",
                "local_curvature_cost",
                "local_raw_curvature_cost",
                "global_max_traversal_cost",
                "local_mean_risk",
                "local_min_stability",
                "shared_traversable_voxels",
                "global_traversability_checks",
                "global_normal_initializations",
                "local_traversable_voxels",
                "local_traversable_queries",
                "global_cells",
                "local_cells",
                "planning_map_source",
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
                f"{local_result.raw_path_length_m:.3f}",
                local_result.path_states,
                local_result.raw_path_states,
                local_result.waypoint_states,
                f"{local_result.curvature_cost:.4f}",
                f"{local_result.raw_curvature_cost:.4f}",
                f"{global_result.max_traversal_cost:.4f}",
                f"{local_result.mean_risk:.4f}",
                f"{local_result.min_stability:.4f}",
                result["shared_traversable_voxels"],
                local_result.global_traversability_checks,
                local_result.global_normal_initializations,
                local_result.local_traversable_voxels,
                local_result.local_traversable_queries,
                result["global_cells"],
                result["local_cells"],
                result["planning_map_source"],
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
    table.add_row("local curvature", f"{result['local'].curvature_cost:.3f}")
    table.add_row("local min stability", f"{result['local'].min_stability:.3f}")
    table.add_row("shared traversable voxels", str(result["shared_traversable_voxels"]))
    table.add_row("local traversable voxels", str(result["local"].local_traversable_voxels))
    table.add_row("local map cells", str(result["local_cells"]))
    table.add_row("global normal initializations", str(result["local"].global_normal_initializations))
    Console().print(table)
    Console().print(f"[green]Wrote paper pipeline outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
