from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from traversability.hybrid_local_planner import HybridLocalPlannerConfig, NDTLocalTraversabilityGuide, plan_hybrid_local
from traversability.implicit_map import ImplicitTerrainMap
from traversability.ndt_map import NDTConfig, NDTImplicitMap, adaptive_ndt_config
from traversability.ndt_planner import plan_ndt_global
from traversability.paper_pipeline_demo import crop_pointcloud_local_window, make_pipeline_terrain, point_cloud_layer_from_points, sample_points
from traversability.paper_receding_demo import inject_dynamic_obstacle, yaw_from_path
from traversability.paper_scenario_suite import Scenario, make_scenarios
from traversability.paper_visualization_frontend import FRONTEND_INDEX_PATH, INDEX_HTML, load_index_html
from traversability.realtime_demo import advance_along_path


DEFAULT_STEP_DISTANCE = 0.72
DEFAULT_LOCAL_WINDOW_RADIUS = 3.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动论文主线交互式实时规划演示。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dev", action="store_true", help="启用浏览器端热更新轮询，源码变更后自动刷新页面。")
    parser.add_argument("--once", action="store_true", help="只运行一个规划周期并打印 JSON 摘要，不启动界面。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = PaperInteractiveDemo()
    if args.once:
        payload = demo.plan()
        print(json.dumps(payload["plan"]["summary"], separators=(",", ":")))
        return

    server = make_server(args.host, args.port, demo, dev_mode=args.dev)
    url = f"http://{args.host}:{server.server_port}/"
    Console().print(f"[green]Interactive paper mainline demo:[/green] {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        Console().print("\n[yellow]Stopping interactive demo[/yellow]")
    finally:
        server.server_close()


def make_server(host: str, port: int, demo: "PaperInteractiveDemo", dev_mode: bool = False) -> ThreadingHTTPServer:
    class Handler(PaperDemoHandler):
        shared_demo = demo
        dev_enabled = dev_mode

    return ThreadingHTTPServer((host, port), Handler)


@dataclass
class DynamicBlock:
    x: float
    y: float
    radius: float
    delta: float = 0.75
    obstacle: bool = True
    roughness: float = 0.0


@dataclass(frozen=True)
class InteractiveScenario:
    key: str
    label: str
    height: np.ndarray
    obstacle: np.ndarray
    resolution: float
    origin: tuple[float, float]
    start: tuple[float, float]
    goal: tuple[float, float]
    description: str


@dataclass
class PaperInteractiveDemo:
    base_height: np.ndarray = field(init=False)
    base_obstacle: np.ndarray = field(init=False)
    resolution: float = field(init=False)
    origin: tuple[float, float] = field(init=False)
    scenarios: dict[str, InteractiveScenario] = field(init=False)
    scenario_key: str = "pipeline"
    start: tuple[float, float] = (-2.25, -1.35)
    goal: tuple[float, float] = (2.25, 1.35)
    yaw: float = 0.0
    cycle: int = 0
    step_distance: float = DEFAULT_STEP_DISTANCE
    local_window_radius: float = DEFAULT_LOCAL_WINDOW_RADIUS
    dynamic_blocks: list[DynamicBlock] = field(default_factory=list)
    trajectory: list[tuple[float, float, float]] = field(default_factory=lambda: [(-2.25, -1.35, 0.0)])
    last_plan: dict[str, Any] | None = None
    terrain_version: int = 0
    cached_terrain_version: int = -1
    cached_height: np.ndarray | None = None
    cached_obstacle: np.ndarray | None = None
    cached_layer_origin: tuple[float, float] | None = None
    cached_ndt_map: NDTImplicitMap | None = None
    cached_traversability_guide: NDTLocalTraversabilityGuide | None = None
    cached_metrics: dict[tuple[int, int, int], Any] | None = None
    active_job_id: int = 0
    active_job: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        scenarios = make_interactive_scenarios()
        object.__setattr__(self, "scenarios", scenarios)
        self.apply_scenario("pipeline")

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.apply_scenario(self.scenario_key)
            return self.state(include_terrain=True)

    def set_scenario(self, key: str) -> dict[str, Any]:
        with self.lock:
            if key not in self.scenarios:
                raise ValueError(f"unknown scenario: {key}")
            self.apply_scenario(key)
            return self.state(include_terrain=True)

    def apply_scenario(self, key: str) -> None:
        scenario = self.scenarios[key]
        self.scenario_key = key
        self.base_height = scenario.height.copy()
        self.base_obstacle = scenario.obstacle.copy()
        self.resolution = scenario.resolution
        self.origin = scenario.origin
        self.start = scenario.start
        self.goal = scenario.goal
        self.yaw = math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])
        self.cycle = 0
        self.step_distance = DEFAULT_STEP_DISTANCE
        self.local_window_radius = DEFAULT_LOCAL_WINDOW_RADIUS
        self.dynamic_blocks.clear()
        self.trajectory = [(self.start[0], self.start[1], self.yaw)]
        self.last_plan = None
        self.terrain_version += 1
        self.clear_cache()

    def state(self, include_terrain: bool = False) -> dict[str, Any]:
        with self.lock:
            payload: dict[str, Any] = {
                "sim": self.sim_payload(),
                "plan": self.last_plan,
                "job": self.active_job,
                "scenarios": scenario_options(self.scenarios),
            }
            if include_terrain:
                if self.cached_height is not None and self.cached_obstacle is not None and self.cached_layer_origin is not None:
                    height, obstacle, origin = self.cached_height, self.cached_obstacle, self.cached_layer_origin
                else:
                    height, obstacle = self.current_terrain()
                    origin = self.origin
                payload["terrain"] = terrain_payload(height, obstacle, self.resolution, origin, self.terrain_version)
            return payload

    def set_start(self, x: float, y: float) -> dict[str, Any]:
        with self.lock:
            self.start = (x, y)
            self.yaw = math.atan2(self.goal[1] - y, self.goal[0] - x)
            self.cycle = 0
            self.trajectory = [(x, y, self.yaw)]
            self.last_plan = None
            return self.state(include_terrain=True)

    def set_goal(self, x: float, y: float) -> dict[str, Any]:
        with self.lock:
            self.goal = (x, y)
            self.yaw = math.atan2(y - self.start[1], x - self.start[0])
            self.trajectory = [(self.start[0], self.start[1], self.yaw)]
            self.last_plan = None
            return self.state(include_terrain=True)

    def add_obstacle(
        self,
        x: float,
        y: float,
        radius: float = 0.23,
        delta: float = 0.75,
        obstacle: bool = True,
        roughness: float = 0.0,
    ) -> dict[str, Any]:
        with self.lock:
            self.dynamic_blocks.append(
                DynamicBlock(
                    x=x,
                    y=y,
                    radius=float(np.clip(radius, 0.06, 0.85)),
                    delta=float(np.clip(delta, -0.85, 0.95)),
                    obstacle=bool(obstacle),
                    roughness=float(np.clip(roughness, 0.0, 0.22)),
                )
            )
            self.terrain_version += 1
            self.last_plan = None
            self.clear_cache()
            return self.state(include_terrain=True)

    def smooth_patch(self, x: float, y: float, radius: float = 0.25) -> dict[str, Any]:
        with self.lock:
            self.dynamic_blocks.append(
                DynamicBlock(
                    x=x,
                    y=y,
                    radius=float(np.clip(radius, 0.08, 0.85)),
                    delta=0.0,
                    obstacle=False,
                    roughness=-1.0,
                )
            )
            self.terrain_version += 1
            self.last_plan = None
            self.clear_cache()
            return self.state(include_terrain=True)

    def clear_obstacles(self) -> dict[str, Any]:
        with self.lock:
            self.dynamic_blocks.clear()
            self.terrain_version += 1
            self.last_plan = None
            self.clear_cache()
            return self.state(include_terrain=True)

    def set_params(self, step_distance: float | None = None, local_window_radius: float | None = None) -> dict[str, Any]:
        with self.lock:
            if step_distance is not None:
                self.step_distance = float(np.clip(step_distance, 0.20, 1.20))
            if local_window_radius is not None:
                self.local_window_radius = float(np.clip(local_window_radius, 1.6, 3.2))
            self.last_plan = None
            return self.state(include_terrain=True)

    def plan(self) -> dict[str, Any]:
        with self.lock:
            return self.run_plan_locked()

    def start_async_plan(self) -> dict[str, Any]:
        with self.lock:
            self.active_job_id += 1
            job_id = self.active_job_id
            self.active_job = {
                "id": job_id,
                "status": "running",
                "stage": "cloud",
                "version": 0,
                "startedAt": time.time(),
                "updatedAt": time.time(),
                "stages": [],
                "partial": {},
            }
            payload = self.state(include_terrain=True)
        thread = threading.Thread(target=self.run_async_plan, args=(job_id,), daemon=True)
        thread.start()
        return payload

    def run_async_plan(self, job_id: int) -> None:
        try:
            self.run_plan_locked(job_id=job_id)
        except Exception as exc:
            with self.lock:
                if self.active_job is not None and self.active_job.get("id") == job_id:
                    self.active_job.update({"status": "failed", "stage": "failed", "error": str(exc), "updatedAt": time.time()})

    def run_plan_locked(self, job_id: int | None = None) -> dict[str, Any]:
        begin = time.perf_counter()
        height, obstacle, ndt_map, traversability_guide, metrics, ndt_ms = self.planning_context(job_id=job_id)
        terrain_origin = self.cached_layer_origin if self.cached_layer_origin is not None else self.origin
        self.update_job(job_id, "risk", {"implicitMap": implicit_map_payload(ndt_map, metrics), "voxels": voxel_payload(ndt_map, metrics)})
        t0 = time.perf_counter()
        global_result = plan_ndt_global(ndt_map, (self.start[0], self.start[1], 0.0), (self.goal[0], self.goal[1], 0.0))
        global_ms = elapsed_ms(t0)
        global_xy = [(x, y) for x, y, _ in global_result.path_xyz]
        self.update_job(
            job_id,
            "global",
            {
                "globalPath": [[round(x, 4), round(y, 4)] for x, y in global_xy],
                "globalPath3d": [[round(x, 4), round(y, 4), round(z, 4)] for x, y, z in global_result.path_xyz],
                "globalExpanded": global_result.expanded_nodes,
            },
        )
        local_result = None
        local_bounds = None
        temporary_elevation_map = None
        if global_result.success and len(global_xy) >= 2:
            local_layer = crop_pointcloud_local_window(
                "paper_interactive_local",
                ndt_map.points,
                self.resolution,
                center_xy=self.start,
                radius=self.local_window_radius,
            )
            local_bounds = layer_bounds(local_layer.height.shape, local_layer.resolution, local_layer.origin_xy)
            temporary_elevation_map = temporary_elevation_payload(
                local_layer,
                self.terrain_version,
                self.start,
                self.local_window_radius,
            )
            self.update_job(job_id, "local", {"localWindow": local_bounds, "temporaryElevationMap": temporary_elevation_map})
            local_result = plan_hybrid_local(
                ImplicitTerrainMap(local_layer),
                start=(self.start[0], self.start[1], self.yaw),
                global_path=global_xy,
                config=hybrid_config(self.local_window_radius),
                global_traversability=traversability_guide,
            )

        local_path = local_result.path if local_result is not None else []
        local_path_xyz = path_with_height(local_path, height, self.resolution, terrain_origin)
        summary = {
            "success": bool(global_result.success and local_result is not None and local_result.success),
            "scenario": self.scenario_key,
            "cycle": self.cycle,
            "total_ms": elapsed_ms(begin),
            "ndt_ms": ndt_ms,
            "global_ms": global_ms,
            "local_ms": local_result.runtime_ms if local_result is not None else 0.0,
            "raw_points": int(len(ndt_map.points)),
            "occupied_voxels": int(len(ndt_map.occupied)),
            "global_expanded": global_result.expanded_nodes,
            "local_expanded": local_result.expanded_nodes if local_result is not None else 0,
            "global_path_length_m": global_result.path_length_m,
            "local_path_length_m": local_result.path_length_m if local_result is not None else 0.0,
            "mean_traversal_cost": finite_or_none(global_result.mean_traversal_cost),
            "max_traversal_cost": finite_or_none(global_result.max_traversal_cost),
            "local_mean_risk": finite_or_none(local_result.mean_risk if local_result is not None else float("inf")),
            "local_min_stability": local_result.min_stability if local_result is not None else 0.0,
            "shared_traversable_voxels": len(traversability_guide.traversable_keys),
            "global_traversability_checks": local_result.global_traversability_checks if local_result is not None else 0,
            "global_normal_initializations": local_result.global_normal_initializations if local_result is not None else 0,
            "local_traversable_voxels": local_result.local_traversable_voxels if local_result is not None else 0,
            "local_cells": int((local_bounds["rows"] * local_bounds["cols"]) if local_bounds else 0),
            "global_cells": int(height.size),
            "planning_map_source": "point_cloud_derived",
        }
        self.last_plan = {
            "summary": summary,
            "globalPath": [[round(x, 4), round(y, 4)] for x, y in global_xy],
            "globalPath3d": [[round(x, 4), round(y, 4), round(z, 4)] for x, y, z in global_result.path_xyz],
            "localPath": [[round(x, 4), round(y, 4), round(yaw, 4)] for x, y, yaw in local_path],
            "localPath3d": local_path_xyz,
            "localWindow": local_bounds,
            "temporaryElevationMap": temporary_elevation_map,
            "pointCloud": point_cloud_payload(ndt_map.points),
            "voxels": voxel_payload(ndt_map, metrics),
            "implicitMap": implicit_map_payload(ndt_map, metrics),
            "distributions": distribution_payload(ndt_map, metrics, global_result, local_result),
            "stages": stage_payload(summary),
        }
        self.update_job(job_id, "done", {"plan": self.last_plan}, status="complete")
        return self.state(include_terrain=True)

    def step(self) -> dict[str, Any]:
        with self.lock:
            payload = self.plan()
            plan = self.last_plan
            local_path = plan["localPath"] if plan is not None else []
            if plan is not None and plan["summary"]["success"] and len(local_path) >= 2:
                path_xy = [(point[0], point[1]) for point in local_path]
                next_xy = advance_along_path(path_xy, self.step_distance)
                next_yaw = yaw_from_path([(p[0], p[1], p[2]) for p in local_path], next_xy)
                self.start = next_xy
                self.yaw = next_yaw
                self.cycle += 1
                self.trajectory.append((self.start[0], self.start[1], self.yaw))
                if math.dist(self.start, self.goal) <= self.step_distance:
                    self.start = self.goal
                    self.trajectory.append((self.goal[0], self.goal[1], self.yaw))
            payload["sim"] = self.sim_payload()
            return payload

    def inject_paper_update(self) -> dict[str, Any]:
        with self.lock:
            height, obstacle = self.current_terrain()
            inject_dynamic_obstacle(height, obstacle, self.resolution, self.origin)
            rows, cols = obstacle.shape
            xs = self.origin[0] + np.arange(cols) * self.resolution
            ys = self.origin[1] + np.arange(rows) * self.resolution
            xx, yy = np.meshgrid(xs, ys)
            dynamic = ((xx - 0.78) ** 2 / 0.055 + ((yy - 0.68) ** 2) / 0.028) < 1.0
            if np.any(dynamic):
                y_idx, x_idx = np.argwhere(dynamic).mean(axis=0)
                self.dynamic_blocks.append(
                    DynamicBlock(
                        x=float(self.origin[0] + x_idx * self.resolution),
                        y=float(self.origin[1] + y_idx * self.resolution),
                        radius=0.24,
                    )
                )
            self.terrain_version += 1
            self.last_plan = None
            self.clear_cache()
            return self.state(include_terrain=True)

    def current_terrain(self) -> tuple[np.ndarray, np.ndarray]:
        height = self.base_height.copy()
        obstacle = self.base_obstacle.copy()
        if self.dynamic_blocks:
            rows, cols = height.shape
            xs = self.origin[0] + np.arange(cols) * self.resolution
            ys = self.origin[1] + np.arange(rows) * self.resolution
            xx, yy = np.meshgrid(xs, ys)
            for block in self.dynamic_blocks:
                mask = (xx - block.x) ** 2 + (yy - block.y) ** 2 <= block.radius**2
                if block.roughness < 0.0:
                    smoothed = local_mean_height(height)
                    height[mask] = smoothed[mask]
                    obstacle[mask] = False
                    continue
                if block.obstacle:
                    obstacle |= mask
                else:
                    obstacle[mask] = False
                if abs(block.delta) > 1e-9:
                    height[mask] += block.delta
                if block.roughness > 0.0:
                    ripple = block.roughness * np.sin(34.0 * (xx - block.x)) * np.cos(31.0 * (yy - block.y))
                    height[mask] += ripple[mask]
        return height, obstacle

    def planning_context(
        self,
        job_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, NDTImplicitMap, NDTLocalTraversabilityGuide, dict[tuple[int, int, int], Any], float]:
        if (
            self.cached_terrain_version == self.terrain_version
            and self.cached_height is not None
            and self.cached_obstacle is not None
            and self.cached_layer_origin is not None
            and self.cached_ndt_map is not None
            and self.cached_traversability_guide is not None
            and self.cached_metrics is not None
        ):
            return (
                self.cached_height,
                self.cached_obstacle,
                self.cached_ndt_map,
                self.cached_traversability_guide,
                self.cached_metrics,
                0.0,
            )
        begin = time.perf_counter()
        reference_height, reference_obstacle = self.current_terrain()
        points = sample_points(reference_height, reference_obstacle, self.resolution, self.origin)
        self.update_job(
            job_id,
            "cloud",
            {
                "pointCloud": point_cloud_payload(points),
                "distributions": {"cloud": cloud_distribution(points)},
            },
        )
        pointcloud_layer = point_cloud_layer_from_points("paper_interactive_pointcloud", points, self.resolution)
        self.update_job(
            job_id,
            "ndt",
            {
                "terrain": terrain_payload(pointcloud_layer.height, pointcloud_layer.obstacle, self.resolution, pointcloud_layer.origin_xy, self.terrain_version),
            },
        )
        ndt_map = NDTImplicitMap(points, ndt_config(points, self.resolution))
        metrics = ndt_map.compute_metrics()
        traversability_guide = NDTLocalTraversabilityGuide.from_ndt_map(ndt_map)
        self.cached_terrain_version = self.terrain_version
        self.cached_height = pointcloud_layer.height
        self.cached_obstacle = pointcloud_layer.obstacle
        self.cached_layer_origin = pointcloud_layer.origin_xy
        self.cached_ndt_map = ndt_map
        self.cached_traversability_guide = traversability_guide
        self.cached_metrics = metrics
        return pointcloud_layer.height, pointcloud_layer.obstacle, ndt_map, traversability_guide, metrics, elapsed_ms(begin)

    def update_job(self, job_id: int | None, stage: str, partial: dict[str, Any], status: str = "running") -> None:
        with self.lock:
            if job_id is None or self.active_job is None or self.active_job.get("id") != job_id:
                return
            job_partial = self.active_job.setdefault("partial", {})
            merge_dict(job_partial, partial)
            stages = self.active_job.setdefault("stages", [])
            if stage not in [row.get("key") for row in stages]:
                stages.append({"key": stage, "label": stage, "at": round(time.time(), 3)})
            self.active_job.update(
                {
                    "status": status,
                    "stage": stage,
                    "version": int(self.active_job.get("version", 0)) + 1,
                    "updatedAt": time.time(),
                }
            )

    def clear_cache(self) -> None:
        self.cached_terrain_version = -1
        self.cached_height = None
        self.cached_obstacle = None
        self.cached_layer_origin = None
        self.cached_ndt_map = None
        self.cached_traversability_guide = None
        self.cached_metrics = None

    def sim_payload(self) -> dict[str, Any]:
        if self.cached_height is not None and self.cached_layer_origin is not None:
            height, origin = self.cached_height, self.cached_layer_origin
        else:
            height, _ = self.current_terrain()
            origin = self.origin
        start_z = sample_height_xy(height, self.resolution, origin, self.start)
        goal_z = sample_height_xy(height, self.resolution, origin, self.goal)
        return {
            "scenario": self.scenario_key,
            "start": [round(self.start[0], 4), round(self.start[1], 4), round(self.yaw, 4)],
            "startGround": [round(self.start[0], 4), round(self.start[1], 4), round(start_z, 4), round(self.yaw, 4)],
            "goal": [round(self.goal[0], 4), round(self.goal[1], 4)],
            "goalGround": [round(self.goal[0], 4), round(self.goal[1], 4), round(goal_z, 4)],
            "cycle": self.cycle,
            "stepDistance": self.step_distance,
            "localWindowRadius": self.local_window_radius,
            "dynamicBlocks": [
                [round(b.x, 4), round(b.y, 4), round(b.radius, 4), round(b.delta, 4), int(b.obstacle), round(b.roughness, 4)]
                for b in self.dynamic_blocks
            ],
            "trajectory": [[round(x, 4), round(y, 4), round(yaw, 4)] for x, y, yaw in self.trajectory],
        }


class PaperDemoHandler(BaseHTTPRequestHandler):
    shared_demo: PaperInteractiveDemo
    dev_enabled: bool = False

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index.html"):
            html = load_index_html().replace("__DEV_HOT_RELOAD__", "true" if self.dev_enabled else "false")
            self.send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/state"):
            self.send_json(self.shared_demo.state(include_terrain=True))
            return
        if self.path.startswith("/api/job"):
            self.send_json(self.shared_demo.state(include_terrain=True))
            return
        if self.path.startswith("/api/dev-version"):
            self.send_json({"version": dev_version(), "hotReload": self.dev_enabled})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        data = self.read_json()
        try:
            if self.path == "/api/plan":
                self.send_json(self.shared_demo.plan() if data.get("sync") else self.shared_demo.start_async_plan())
            elif self.path == "/api/step":
                self.send_json(self.shared_demo.step())
            elif self.path == "/api/reset":
                self.send_json(self.shared_demo.reset())
            elif self.path == "/api/scenario":
                self.send_json(self.shared_demo.set_scenario(str(data["key"])))
            elif self.path == "/api/start":
                self.send_json(self.shared_demo.set_start(float(data["x"]), float(data["y"])))
            elif self.path == "/api/goal":
                self.send_json(self.shared_demo.set_goal(float(data["x"]), float(data["y"])))
            elif self.path == "/api/obstacle":
                self.send_json(
                    self.shared_demo.add_obstacle(
                        float(data["x"]),
                        float(data["y"]),
                        float(data.get("radius", 0.23)),
                        delta=float(data.get("delta", 0.75)),
                        obstacle=bool(data.get("obstacle", True)),
                        roughness=float(data.get("roughness", 0.0)),
                    )
                )
            elif self.path == "/api/smooth":
                self.send_json(self.shared_demo.smooth_patch(float(data["x"]), float(data["y"]), float(data.get("radius", 0.25))))
            elif self.path == "/api/clear-obstacles":
                self.send_json(self.shared_demo.clear_obstacles())
            elif self.path == "/api/paper-update":
                self.send_json(self.shared_demo.inject_paper_update())
            elif self.path == "/api/params":
                self.send_json(
                    self.shared_demo.set_params(
                        step_distance=optional_float(data.get("stepDistance")),
                        local_window_radius=optional_float(data.get("localWindowRadius")),
                    )
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except (KeyError, TypeError, ValueError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8")

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def dev_version() -> str:
    paths = [Path(__file__), FRONTEND_INDEX_PATH]
    parts = []
    for path in paths:
        stat = path.stat()
        parts.append(f"{path.name}:{int(stat.st_mtime_ns)}:{stat.st_size}")
    return "|".join(parts)


def ndt_config(points: np.ndarray, resolution: float) -> NDTConfig:
    return adaptive_ndt_config(points, resolution)


def hybrid_config(local_window_radius: float) -> HybridLocalPlannerConfig:
    return HybridLocalPlannerConfig(
        step_length=0.22,
        local_window_radius=local_window_radius,
        goal_tolerance=0.32,
        min_stability=0.28,
        max_iterations=6500,
        global_waypoint_limit=20,
        global_traversability_radius_cells=2,
    )


def make_interactive_scenarios() -> dict[str, InteractiveScenario]:
    height, obstacle, resolution, origin = make_pipeline_terrain()
    scenarios = {
        "pipeline": InteractiveScenario(
            key="pipeline",
            label="论文主线",
            height=height,
            obstacle=obstacle,
            resolution=resolution,
            origin=origin,
            start=(-2.25, -1.35),
            goal=(2.25, 1.35),
            description="NDT 全局规划与 Hybrid A* 局部规划集成场景",
        )
    }
    descriptions = {
        "stairs": "多层台阶与窄通道，突出高程歧义和 3D 体素连通",
        "rubble": "碎石凸起与稠密障碍，突出粗糙度和碰撞风险",
        "grass": "稀疏草丛纹理，突出点云稀疏度和地形复杂度",
        "hill": "坡地和洼地组合，突出坡度代价",
        "bridge": "沟壑桥面结构，突出 falling risk",
        "field": "大范围混合地形，突出滚动重规划与局部窗口",
    }
    for scenario in make_scenarios():
        scenarios[scenario.name] = from_paper_scenario(scenario, descriptions.get(scenario.name, scenario.name))
    return scenarios


def from_paper_scenario(scenario: Scenario, description: str) -> InteractiveScenario:
    return InteractiveScenario(
        key=scenario.name,
        label=scenario.name,
        height=scenario.height.astype(np.float64),
        obstacle=scenario.obstacle.astype(bool),
        resolution=float(scenario.resolution),
        origin=scenario.origin,
        start=(float(scenario.start[0]), float(scenario.start[1])),
        goal=(float(scenario.goal[0]), float(scenario.goal[1])),
        description=description,
    )


def scenario_options(scenarios: dict[str, InteractiveScenario]) -> list[dict[str, str]]:
    return [
        {"key": scenario.key, "label": scenario.label, "description": scenario.description}
        for scenario in scenarios.values()
    ]


def terrain_payload(
    height: np.ndarray,
    obstacle: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    version: int,
) -> dict[str, Any]:
    finite = height[np.isfinite(height)]
    hmin = float(np.min(finite))
    hmax = float(np.max(finite))
    scale = max(hmax - hmin, 1e-9)
    normalized = np.round((np.clip((height - hmin) / scale, 0.0, 1.0) * 255.0)).astype(np.uint8)
    return {
        "version": version,
        "rows": int(height.shape[0]),
        "cols": int(height.shape[1]),
        "resolution": resolution,
        "origin": [origin[0], origin[1]],
        "heightMin": hmin,
        "heightMax": hmax,
        "height8": normalized.ravel().tolist(),
        "height": np.round(height.astype(np.float64), 4).ravel().tolist(),
        "obstacle": obstacle.astype(np.uint8).ravel().tolist(),
    }


def temporary_elevation_payload(
    layer: Any,
    version: int,
    center_xy: tuple[float, float],
    radius: float,
) -> dict[str, Any]:
    payload = terrain_payload(layer.height, layer.obstacle, layer.resolution, layer.origin_xy, version)
    payload["source"] = "local_point_cloud_window"
    payload["center"] = [round(float(center_xy[0]), 4), round(float(center_xy[1]), 4)]
    payload["radius"] = round(float(radius), 4)
    return payload


def local_mean_height(height: np.ndarray) -> np.ndarray:
    padded = np.pad(height, 1, mode="edge")
    result = np.zeros_like(height)
    for row in range(height.shape[0]):
        for col in range(height.shape[1]):
            result[row, col] = float(np.mean(padded[row : row + 3, col : col + 3]))
    return result


def voxel_payload(ndt_map: NDTImplicitMap, metrics: dict[tuple[int, int, int], Any], limit: int = 5200) -> list[list[float]]:
    voxels = []
    for key, metric in metrics.items():
        center = ndt_map.voxel_center(key)
        finite = np.isfinite(metric.traversal_cost)
        cost = 1.0 if not finite else float(np.clip(metric.traversal_cost, 0.0, 1.0))
        voxels.append(
            [
                round(float(center[0]), 4),
                round(float(center[1]), 4),
                round(float(center[2]), 4),
                round(cost, 4),
                0.0 if finite else 1.0,
                round(float(metric.roughness), 4),
                round(float(metric.slope), 4),
                round(float(metric.sparsity), 4),
                round(float(metric.normal[0]), 4),
                round(float(metric.normal[1]), 4),
                round(float(metric.normal[2]), 4),
            ]
        )
    return sample_evenly(voxels, limit)


def point_cloud_payload(points: np.ndarray, limit: int = 1800) -> list[list[float]]:
    if len(points) == 0:
        return []
    selected = points[np.linspace(0, len(points) - 1, min(limit, len(points)), dtype=np.int64)]
    return np.round(selected.astype(np.float64), 4).tolist()


def implicit_map_payload(ndt_map: NDTImplicitMap, metrics: dict[tuple[int, int, int], Any]) -> dict[str, Any]:
    values = list(metrics.values())
    finite = [metric for metric in values if np.isfinite(metric.traversal_cost)]
    risky = [metric for metric in values if metric.traversal_risk]
    normals = []
    gaussians = []
    sampled_metrics = sample_evenly(values, 220)
    for metric in sampled_metrics:
        center = ndt_map.voxel_center(metric.key)
        end = center + metric.normal * ndt_map.config.voxel_size * 0.9
        normals.append(
            [
                round(float(center[0]), 4),
                round(float(center[1]), 4),
                round(float(center[2]), 4),
                round(float(end[0]), 4),
                round(float(end[1]), 4),
                round(float(end[2]), 4),
            ]
        )
        cell = ndt_map.cells.get(metric.key)
        if cell is not None:
            axes = gaussian_axes(cell.covariance, ndt_map.config.voxel_size)
            gaussians.append(
                [
                    round(float(center[0]), 4),
                    round(float(center[1]), 4),
                    round(float(center[2]), 4),
                    round(float(axes[0][0]), 4),
                    round(float(axes[0][1]), 4),
                    round(float(axes[0][2]), 4),
                    round(float(axes[1][0]), 4),
                    round(float(axes[1][1]), 4),
                    round(float(axes[1][2]), 4),
                    round(float(axes[2][0]), 4),
                    round(float(axes[2][1]), 4),
                    round(float(axes[2][2]), 4),
                    round(float(metric.traversal_cost if np.isfinite(metric.traversal_cost) else 1.0), 4),
                    round(float(metric.roughness), 4),
                    round(float(metric.sparsity), 4),
                ]
            )
    return {
        "voxelSize": ndt_map.config.voxel_size,
        "verticalVoxelSize": ndt_map.config.z_voxel_size,
        "fusionRadius": ndt_map.config.fusion_radius,
        "octreeLeaves": ndt_map.octree.leaf_count,
        "octreeNodes": ndt_map.octree.node_count,
        "occupied": len(ndt_map.occupied),
        "finite": len(finite),
        "risk": len(risky),
        "terrainRisk": int(sum(metric.terrain_risk for metric in values)),
        "collisionRisk": int(sum(metric.collision_risk for metric in values)),
        "fallingRisk": int(sum(metric.falling_risk for metric in values)),
        "meanRoughness": float(np.mean([metric.roughness for metric in values])) if values else 0.0,
        "meanSlope": float(np.mean([metric.slope for metric in values])) if values else 0.0,
        "meanSparsity": float(np.mean([metric.sparsity for metric in values])) if values else 0.0,
        "normals": normals,
        "gaussians": gaussians,
        "distributions": ndt_distribution(ndt_map, metrics),
    }


def distribution_payload(ndt_map, metrics, global_result, local_result) -> dict[str, Any]:
    return {
        "cloud": cloud_distribution(ndt_map.points),
        "ndt": ndt_distribution(ndt_map, metrics),
        "risk": metric_distribution(metrics),
        "global": {
            "expanded": int(global_result.expanded_nodes),
            "pathPoints": len(global_result.path_xyz),
            "pathLength": round(float(global_result.path_length_m), 4),
            "maxCost": finite_or_none(global_result.max_traversal_cost),
        },
        "local": {
            "expanded": int(local_result.expanded_nodes) if local_result is not None else 0,
            "pathStates": int(local_result.path_states) if local_result is not None else 0,
            "minStability": round(float(local_result.min_stability), 4) if local_result is not None else 0.0,
            "meanRisk": finite_or_none(local_result.mean_risk if local_result is not None else float("inf")),
        },
    }


def cloud_distribution(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    return {
        "count": int(len(points)),
        "z": histogram_payload(points[:, 2], bins=10),
    }


def ndt_distribution(ndt_map: NDTImplicitMap, metrics: dict[tuple[int, int, int], Any]) -> dict[str, Any]:
    counts = np.array([ndt_map.cells[key].count for key in ndt_map.occupied if key in ndt_map.cells], dtype=np.float64)
    z_layers = np.array([key[2] for key in ndt_map.occupied], dtype=np.float64)
    return {
        "occupied": int(len(ndt_map.occupied)),
        "xySize": round(float(ndt_map.config.voxel_size), 4),
        "zSize": round(float(ndt_map.config.z_voxel_size), 4),
        "pointsPerVoxel": histogram_payload(counts, bins=8),
        "zLayers": histogram_payload(z_layers, bins=8),
        **metric_distribution(metrics),
    }


def metric_distribution(metrics: dict[tuple[int, int, int], Any]) -> dict[str, Any]:
    values = list(metrics.values())
    if not values:
        return {"cost": histogram_payload(np.array([]), bins=8), "roughness": histogram_payload(np.array([]), bins=8), "slope": histogram_payload(np.array([]), bins=8), "sparsity": histogram_payload(np.array([]), bins=8)}
    costs = np.array([metric.traversal_cost if np.isfinite(metric.traversal_cost) else 1.0 for metric in values], dtype=np.float64)
    return {
        "cost": histogram_payload(costs, bins=8, lo=0.0, hi=1.0),
        "roughness": histogram_payload(np.array([metric.roughness for metric in values], dtype=np.float64), bins=8, lo=0.0, hi=1.0),
        "slope": histogram_payload(np.array([metric.slope for metric in values], dtype=np.float64), bins=8),
        "sparsity": histogram_payload(np.array([metric.sparsity for metric in values], dtype=np.float64), bins=8, lo=0.0, hi=1.0),
    }


def histogram_payload(values: np.ndarray, bins: int, lo: float | None = None, hi: float | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"min": 0.0, "max": 0.0, "bins": [0 for _ in range(bins)]}
    low = float(np.min(values)) if lo is None else lo
    high = float(np.max(values)) if hi is None else hi
    if high <= low:
        high = low + 1e-6
    counts, _ = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    return {"min": round(low, 4), "max": round(high, 4), "bins": [int(v) for v in counts.tolist()]}


def merge_dict(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_dict(target[key], value)
        else:
            target[key] = value


def gaussian_axes(covariance: np.ndarray, voxel_size: float) -> list[np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-8)
    eigenvectors = eigenvectors[:, order]
    axes = []
    for index in range(3):
        length = float(np.clip(2.0 * math.sqrt(eigenvalues[index]), voxel_size * 0.18, voxel_size * 1.45))
        axes.append(eigenvectors[:, index] * length)
    return axes


def path_with_height(
    path: list[tuple[float, float, float]],
    height: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
) -> list[list[float]]:
    if not path:
        return []
    result = []
    rows, cols = height.shape
    for x, y, yaw in path:
        col = int(np.clip(round((x - origin[0]) / resolution), 0, cols - 1))
        row = int(np.clip(round((y - origin[1]) / resolution), 0, rows - 1))
        z = float(height[row, col]) + 0.05
        result.append([round(x, 4), round(y, 4), round(z, 4), round(yaw, 4)])
    return result


def sample_height_xy(
    height: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    xy: tuple[float, float],
) -> float:
    rows, cols = height.shape
    col = int(np.clip(round((xy[0] - origin[0]) / resolution), 0, cols - 1))
    row = int(np.clip(round((xy[1] - origin[1]) / resolution), 0, rows - 1))
    return float(height[row, col])


def sample_evenly(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=np.int64)
    return [items[int(index)] for index in indices]


def layer_bounds(shape: tuple[int, int], resolution: float, origin: tuple[float, float]) -> dict[str, Any]:
    rows, cols = shape
    return {
        "x0": origin[0],
        "y0": origin[1],
        "x1": origin[0] + cols * resolution,
        "y1": origin[1] + rows * resolution,
        "rows": rows,
        "cols": cols,
    }


def stage_payload(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": "cloud", "label": "点云输入", "detail": f"{summary['raw_points']} xyz samples", "ms": 0.0, "ok": summary["raw_points"] > 0},
        {
            "key": "ndt",
            "label": "隐式 NDT 体素地图",
            "detail": f"{summary['occupied_voxels']} occupied / {summary['shared_traversable_voxels']} traversable",
            "ms": summary["ndt_ms"],
            "ok": summary["shared_traversable_voxels"] > 0,
        },
        {
            "key": "risk",
            "label": "roughness/slope/sparsity 风险",
            "detail": f"max cost {summary['max_traversal_cost'] if summary['max_traversal_cost'] is not None else '--'}",
            "ms": 0.0,
            "ok": summary["max_traversal_cost"] is not None,
        },
        {
            "key": "global",
            "label": "3D voxel A* 全局规划",
            "detail": f"{summary['global_expanded']} expanded nodes",
            "ms": summary["global_ms"],
            "ok": summary["global_path_length_m"] > 0,
        },
        {
            "key": "local",
            "label": "Hybrid A* 局部轨迹",
            "detail": f"{summary['local_expanded']} expanded states",
            "ms": summary["local_ms"],
            "ok": summary["local_path_length_m"] > 0,
        },
        {
            "key": "stable",
            "label": "构型稳定性过滤",
            "detail": f"min stability {summary['local_min_stability']:.2f}",
            "ms": 0.0,
            "ok": summary["local_min_stability"] >= 0.28,
        },
        {
            "key": "recede",
            "label": "滚动窗口重规划",
            "detail": f"cycle {summary['cycle']}",
            "ms": summary["total_ms"],
            "ok": summary["success"],
        },
    ]


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def elapsed_ms(begin: float) -> float:
    return (time.perf_counter() - begin) * 1000.0


if __name__ == "__main__":
    main()
