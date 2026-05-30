from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from traversability.hybrid_local_planner import HybridLocalPlannerConfig, NDTLocalTraversabilityGuide, plan_hybrid_local
from traversability.implicit_map import ImplicitTerrainMap
from traversability.ndt_map import NDTConfig, NDTImplicitMap
from traversability.ndt_planner import plan_ndt_global
from traversability.paper_pipeline_demo import make_pipeline_terrain, sample_points
from traversability.realtime_demo import advance_along_path
from traversability.terrain import analyze_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run receding-horizon NDT global + Hybrid local planning.")
    parser.add_argument("--output", type=Path, default=Path("runs/paper_receding"))
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--step-distance", type=float, default=0.72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_receding(args.cycles, args.step_distance)
    write_outputs(args.output, result)
    plot_outputs(args.output, result)
    print_summary(args.output, result)


def run_receding(cycles: int, step_distance: float) -> dict:
    base_height, base_obstacle, resolution, origin = make_pipeline_terrain()
    goal = (1.55, 0.72)
    current = (-1.55, -0.75)
    yaw = 0.0
    trajectory = [(current[0], current[1], yaw)]
    cycle_rows = []
    planned_paths: list[list[tuple[float, float]]] = []
    final_obstacle = base_obstacle.copy()

    for cycle in range(cycles):
        height = base_height.copy()
        obstacle = base_obstacle.copy()
        if cycle >= max(1, cycles // 2):
            inject_dynamic_obstacle(height, obstacle, resolution, origin)
        final_obstacle = obstacle
        points = sample_points(height, obstacle, resolution, origin)
        ndt_map = NDTImplicitMap(points, ndt_config())
        global_result = plan_ndt_global(ndt_map, (current[0], current[1], 0.0), (goal[0], goal[1], 0.0))
        global_xy = [(x, y) for x, y, _ in global_result.path_xyz]
        terrain_map = ImplicitTerrainMap(analyze_layer("paper_receding", height, obstacle, resolution, origin))
        traversability_guide = NDTLocalTraversabilityGuide.from_ndt_map(ndt_map)
        local_result = plan_hybrid_local(
            terrain_map,
            start=(current[0], current[1], yaw),
            global_path=global_xy,
            config=hybrid_config(),
            global_traversability=traversability_guide,
        )
        if global_xy:
            planned_paths.append(global_xy)
        if not global_result.success or not local_result.success or len(local_result.path) < 2:
            cycle_rows.append(
                cycle_row(cycle, global_result, local_result, current, goal, len(traversability_guide.traversable_keys), failed=True)
            )
            break

        local_xy = [(x, y) for x, y, _ in local_result.path]
        next_xy = advance_along_path(local_xy, step_distance)
        next_yaw = yaw_from_path(local_result.path, next_xy)
        current = next_xy
        yaw = next_yaw
        trajectory.append((current[0], current[1], yaw))
        cycle_rows.append(
            cycle_row(cycle, global_result, local_result, current, goal, len(traversability_guide.traversable_keys), failed=False)
        )
        if math.dist(current, goal) <= step_distance:
            trajectory.append((goal[0], goal[1], yaw))
            break

    return {
        "height": base_height,
        "obstacle": final_obstacle,
        "resolution": resolution,
        "origin": origin,
        "goal": goal,
        "trajectory": trajectory,
        "cycle_rows": cycle_rows,
        "planned_paths": planned_paths,
    }


def ndt_config() -> NDTConfig:
    return NDTConfig(
        voxel_size=0.18,
        fusion_radius=0.36,
        saturation_count=2,
        slope_threshold_rad=np.deg2rad(35.0),
        complexity_threshold=0.82,
        robot_radius=0.28,
        robot_height=0.55,
    )


def hybrid_config() -> HybridLocalPlannerConfig:
    return HybridLocalPlannerConfig(
        step_length=0.22,
        local_window_radius=2.6,
        goal_tolerance=0.32,
        min_stability=0.28,
        max_iterations=6500,
        global_waypoint_limit=20,
    )


def inject_dynamic_obstacle(
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
) -> None:
    rows, cols = height.shape
    xs = origin[0] + np.arange(cols) * resolution
    ys = origin[1] + np.arange(rows) * resolution
    xx, yy = np.meshgrid(xs, ys)
    dynamic = ((xx - 0.78) ** 2 / 0.055 + ((yy - 0.68) ** 2) / 0.028) < 1.0
    obstacle |= dynamic
    height[dynamic] += 0.75


def yaw_from_path(path: list[tuple[float, float, float]], point: tuple[float, float]) -> float:
    nearest = min(range(len(path)), key=lambda idx: math.dist(point, path[idx][:2]))
    if nearest < len(path) - 1:
        a = path[nearest]
        b = path[nearest + 1]
    else:
        a = path[max(0, nearest - 1)]
        b = path[nearest]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def cycle_row(cycle, global_result, local_result, current, goal, shared_traversable_voxels: int, failed: bool) -> dict:
    return {
        "cycle": cycle,
        "success": int(not failed and global_result.success and local_result.success),
        "global_runtime_ms": global_result.runtime_ms,
        "local_runtime_ms": local_result.runtime_ms,
        "global_expanded": global_result.expanded_nodes,
        "local_expanded": local_result.expanded_nodes,
        "global_path_length_m": global_result.path_length_m,
        "local_path_length_m": local_result.path_length_m,
        "local_raw_path_length_m": local_result.raw_path_length_m,
        "local_path_states": local_result.path_states,
        "local_raw_path_states": local_result.raw_path_states,
        "local_waypoint_states": local_result.waypoint_states,
        "local_curvature_cost": local_result.curvature_cost,
        "local_raw_curvature_cost": local_result.raw_curvature_cost,
        "local_mean_risk": local_result.mean_risk,
        "local_min_stability": local_result.min_stability,
        "shared_traversable_voxels": shared_traversable_voxels,
        "global_traversability_checks": local_result.global_traversability_checks,
        "global_normal_initializations": local_result.global_normal_initializations,
        "distance_to_goal_m": math.dist(current, goal),
    }


def write_outputs(output_dir: Path, result: dict) -> None:
    with (output_dir / "receding_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "cycle",
            "success",
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
            "local_mean_risk",
            "local_min_stability",
            "shared_traversable_voxels",
            "global_traversability_checks",
            "global_normal_initializations",
            "distance_to_goal_m",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["cycle_rows"]:
            writer.writerow({key: format_value(row[key]) for key in fieldnames})

    with (output_dir / "executed_trajectory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "yaw"])
        for x, y, yaw in result["trajectory"]:
            writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{yaw:.4f}"])


def format_value(value):
    return f"{value:.4f}" if isinstance(value, float) else value


def plot_outputs(output_dir: Path, result: dict) -> None:
    height = result["height"]
    obstacle = result["obstacle"]
    resolution = result["resolution"]
    origin = result["origin"]
    extent = [origin[0], origin[0] + resolution * height.shape[1], origin[1], origin[1] + resolution * height.shape[0]]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(height, cmap="terrain", origin="lower", extent=extent)
    ax.contour(obstacle.astype(float), levels=[0.5], colors="black", origin="lower", extent=extent)
    for idx, path in enumerate(result["planned_paths"]):
        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="#ffffff", linewidth=1.0, alpha=0.5, label="global plans" if idx == 0 else None)
    trajectory = result["trajectory"]
    if trajectory:
        tx = [state[0] for state in trajectory]
        ty = [state[1] for state in trajectory]
        ax.plot(tx, ty, color="#1f77b4", linewidth=2.8, marker="o", label="executed")
        ax.scatter([tx[0]], [ty[0]], color="#2ca02c", s=60, label="start")
    gx, gy = result["goal"]
    ax.scatter([gx], [gy], color="#d62728", marker="*", s=75, label="goal")
    ax.set_title("Receding NDT global + Hybrid local planning")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.86)
    fig.savefig(output_dir / "paper_receding.png", dpi=170)
    plt.close(fig)


def print_summary(output_dir: Path, result: dict) -> None:
    rows = result["cycle_rows"]
    success_rate = np.mean([row["success"] for row in rows]) if rows else 0.0
    final_distance = math.dist(result["trajectory"][-1][:2], result["goal"]) if result["trajectory"] else float("inf")
    table = Table(title="Receding Paper-Style Pipeline")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("cycles", str(len(rows)))
    table.add_row("success rate", f"{success_rate:.1%}")
    table.add_row("final distance", f"{final_distance:.2f} m")
    table.add_row("mean global runtime", f"{np.mean([row['global_runtime_ms'] for row in rows]):.1f} ms" if rows else "nan")
    table.add_row("mean local runtime", f"{np.mean([row['local_runtime_ms'] for row in rows]):.1f} ms" if rows else "nan")
    Console().print(table)
    Console().print(f"[green]Wrote receding pipeline outputs to[/green] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
