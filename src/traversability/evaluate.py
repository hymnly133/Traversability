from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from rich.console import Console
from rich.table import Table


DEFAULT_PATHS = {
    "centerline_blocked": [
        (-2.25, -0.05),
        (-1.45, 0.02),
        (-0.65, 0.02),
        (0.10, 0.02),
        (0.90, 0.02),
        (1.65, 0.06),
        (2.25, 0.10),
    ],
    "lower_corridor": [
        (-2.25, -1.02),
        (-1.42, -1.15),
        (-0.54, -1.22),
        (0.36, -1.18),
        (1.42, -1.26),
        (2.15, -0.96),
        (2.25, 0.10),
    ],
    "upper_corridor": [
        (-2.25, -0.05),
        (-2.22, 1.18),
        (-1.26, 1.22),
        (-0.30, 1.26),
        (0.68, 1.17),
        (1.45, 0.90),
        (2.25, 0.10),
    ],
}


@dataclass(frozen=True)
class Sample:
    x: float
    y: float
    yaw: float
    clearance_m: float
    collided: bool


@dataclass(frozen=True)
class PathResult:
    name: str
    samples: list[Sample]
    path_length_m: float
    direct_length_m: float
    min_clearance_m: float
    p10_clearance_m: float
    mean_clearance_m: float
    collision_rate: float
    length_efficiency: float
    score: float


def interpolate_path(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float, float]]:
    if len(points) < 2:
        raise ValueError("A path needs at least two waypoints.")

    samples: list[tuple[float, float, float]] = []
    for start, end in zip(points[:-1], points[1:]):
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        segment_length = math.hypot(dx, dy)
        if segment_length == 0:
            continue
        yaw = math.atan2(dy, dx)
        steps = max(1, math.ceil(segment_length / spacing))
        for i in range(steps):
            t = i / steps
            samples.append((sx + dx * t, sy + dy * t, yaw))
    lx, ly = points[-1]
    px, py = points[-2]
    samples.append((lx, ly, math.atan2(ly - py, lx - px)))
    return samples


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def set_probe_pose(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, yaw: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qpos_address = model.jnt_qposadr[joint_id]
    data.qpos[qpos_address : qpos_address + 3] = (x, y, 0.08)
    data.qpos[qpos_address + 3 : qpos_address + 7] = (
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def named_geom_ids(model: mujoco.MjModel, prefix: str) -> list[int]:
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(prefix):
            geom_ids.append(geom_id)
    return geom_ids


def has_robot_obstacle_contact(data: mujoco.MjData, robot_id: int, obstacle_ids: set[int]) -> bool:
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {contact.geom1, contact.geom2}
        if robot_id in pair and pair.intersection(obstacle_ids):
            return True
    return False


def evaluate_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    waypoints: list[tuple[float, float]],
    sample_spacing: float,
    desired_clearance: float,
    distance_limit: float,
) -> PathResult:
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_footprint")
    obstacle_ids = named_geom_ids(model, "obstacle_")
    obstacle_id_set = set(obstacle_ids)

    samples: list[Sample] = []
    for x, y, yaw in interpolate_path(waypoints, sample_spacing):
        set_probe_pose(model, data, x, y, yaw)
        min_distance = distance_limit
        for obstacle_id in obstacle_ids:
            from_to = np.zeros(6, dtype=np.float64)
            distance = mujoco.mj_geomDistance(model, data, robot_id, obstacle_id, distance_limit, from_to)
            min_distance = min(min_distance, float(distance))

        collided = has_robot_obstacle_contact(data, robot_id, obstacle_id_set) or min_distance <= 0.0
        if collided:
            min_distance = min(min_distance, 0.0)
        samples.append(Sample(x=x, y=y, yaw=yaw, clearance_m=min_distance, collided=collided))

    clearances = np.array([sample.clearance_m for sample in samples], dtype=np.float64)
    collided = np.array([sample.collided for sample in samples], dtype=bool)
    normalized_clearance = np.clip(clearances / desired_clearance, 0.0, 1.0)

    length = path_length(waypoints)
    direct = math.dist(waypoints[0], waypoints[-1])
    length_efficiency = direct / max(length, 1e-9)
    collision_rate = float(np.mean(collided))
    passable_fraction = 1.0 - collision_rate
    score = passable_fraction * (
        0.55 * float(np.mean(normalized_clearance))
        + 0.25 * float(np.percentile(normalized_clearance, 10))
        + 0.20 * length_efficiency
    )

    return PathResult(
        name=name,
        samples=samples,
        path_length_m=length,
        direct_length_m=direct,
        min_clearance_m=float(np.min(clearances)),
        p10_clearance_m=float(np.percentile(clearances, 10)),
        mean_clearance_m=float(np.mean(clearances)),
        collision_rate=collision_rate,
        length_efficiency=length_efficiency,
        score=float(np.clip(score, 0.0, 1.0)),
    )


def write_summary_csv(results: list[PathResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "path_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "path",
                "score",
                "collision_rate",
                "min_clearance_m",
                "p10_clearance_m",
                "mean_clearance_m",
                "path_length_m",
                "length_efficiency",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    f"{result.score:.4f}",
                    f"{result.collision_rate:.4f}",
                    f"{result.min_clearance_m:.4f}",
                    f"{result.p10_clearance_m:.4f}",
                    f"{result.mean_clearance_m:.4f}",
                    f"{result.path_length_m:.4f}",
                    f"{result.length_efficiency:.4f}",
                ]
            )

    with (output_dir / "path_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "x", "y", "yaw_rad", "clearance_m", "collided"])
        for result in results:
            for sample in result.samples:
                writer.writerow(
                    [
                        result.name,
                        f"{sample.x:.4f}",
                        f"{sample.y:.4f}",
                        f"{sample.yaw:.4f}",
                        f"{sample.clearance_m:.4f}",
                        int(sample.collided),
                    ]
                )


def plot_scene(model: mujoco.MjModel, results: list[PathResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title("MuJoCo path traversability score")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.85, 2.85)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    box_type = mujoco.mjtGeom.mjGEOM_BOX
    cylinder_type = mujoco.mjtGeom.mjGEOM_CYLINDER
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not (name.startswith("obstacle_") or name.startswith("wall_")):
            continue
        x, y = model.geom_pos[geom_id][:2]
        if model.geom_type[geom_id] == box_type:
            sx, sy = model.geom_size[geom_id][:2]
            rect = plt.Rectangle((x - sx, y - sy), sx * 2, sy * 2, color="#5b4636", alpha=0.85)
            ax.add_patch(rect)
        elif model.geom_type[geom_id] == cylinder_type:
            radius = model.geom_size[geom_id][0]
            circle = plt.Circle((x, y), radius, color="#7a4a2a", alpha=0.9)
            ax.add_patch(circle)

    colors = {
        "centerline_blocked": "#cf3f3f",
        "lower_corridor": "#2176ae",
        "upper_corridor": "#2f8f4e",
    }
    for result in results:
        xs = [sample.x for sample in result.samples]
        ys = [sample.y for sample in result.samples]
        color = colors.get(result.name, None)
        ax.plot(xs, ys, linewidth=2.4, color=color, label=f"{result.name}: {result.score:.2f}")
        collided_x = [sample.x for sample in result.samples if sample.collided]
        collided_y = [sample.y for sample in result.samples if sample.collided]
        if collided_x:
            ax.scatter(collided_x, collided_y, color="#111111", marker="x", s=42, zorder=5)

    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_dir / "traversability_paths.png", dpi=160)
    plt.close(fig)


def print_results(results: list[PathResult]) -> None:
    table = Table(title="Path Traversability")
    table.add_column("Path")
    table.add_column("Score", justify="right")
    table.add_column("Collision", justify="right")
    table.add_column("Min clearance", justify="right")
    table.add_column("P10 clearance", justify="right")
    table.add_column("Length", justify="right")
    table.add_column("Efficiency", justify="right")

    for result in sorted(results, key=lambda item: item.score, reverse=True):
        table.add_row(
            result.name,
            f"{result.score:.3f}",
            f"{result.collision_rate:.1%}",
            f"{result.min_clearance_m:.3f} m",
            f"{result.p10_clearance_m:.3f} m",
            f"{result.path_length_m:.2f} m",
            f"{result.length_efficiency:.2f}",
        )
    Console().print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify path traversability in a MuJoCo scene.")
    parser.add_argument("--scene", type=Path, default=Path("assets/traversability_scene.xml"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--sample-spacing", type=float, default=0.05)
    parser.add_argument("--desired-clearance", type=float, default=0.35)
    parser.add_argument("--distance-limit", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    results = [
        evaluate_path(
            model=model,
            data=data,
            name=name,
            waypoints=waypoints,
            sample_spacing=args.sample_spacing,
            desired_clearance=args.desired_clearance,
            distance_limit=args.distance_limit,
        )
        for name, waypoints in DEFAULT_PATHS.items()
    ]
    write_summary_csv(results, args.output)
    plot_scene(model, results, args.output)
    print_results(results)
    Console().print(f"[green]Wrote results to[/green] {args.output.resolve()}")


if __name__ == "__main__":
    main()
