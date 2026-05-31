from __future__ import annotations

import json
import subprocess
import sys

from traversability.paper_interactive_demo import INDEX_HTML, PaperInteractiveDemo


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "traversability.paper_interactive_demo", "--once"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    verify_summary(summary)
    verify_payload()
    verify_frontend_source()
    print("paper interactive demo verification passed")


def verify_payload() -> None:
    demo = PaperInteractiveDemo()
    payload = demo.plan()
    plan = payload["plan"]
    summary = plan["summary"]
    verify_summary(summary)
    sim = payload["sim"]
    require(len(sim["startGround"]) == 4 and len(sim["goalGround"]) == 3, "ground-projected start/goal markers are missing")
    require(abs(sim["startGround"][2] - terrain_height_at(payload["terrain"], sim["startGround"])) < 0.08, "start marker is not attached to terrain height")
    require(abs(sim["goalGround"][2] - terrain_height_at(payload["terrain"], sim["goalGround"])) < 0.08, "goal marker is not attached to terrain height")
    require(len(payload["scenarios"]) >= 7, "interactive demo is missing preset maps")
    require({"pipeline", "stairs", "rubble", "grass", "hill", "bridge", "field"} <= {row["key"] for row in payload["scenarios"]}, "preset map list is incomplete")
    require(len(plan["globalPath3d"]) >= 8, "3D global path is missing")
    require(all(len(point) == 3 for point in plan["globalPath3d"]), "global path is not xyz")
    require(len(plan["localPath3d"]) >= 2, "3D local path is missing")
    require(all(len(point) == 4 for point in plan["localPath3d"]), "local path is not xyz-yaw")
    require(len(plan["pointCloud"]) >= 1000, "point-cloud visualization sample is too small")
    require(all(len(point) == 3 for point in plan["pointCloud"][:20]), "point cloud is not xyz")
    require(len(plan["voxels"]) >= 150, "NDT voxel visualization sample is too small")
    require(all(len(voxel) == 11 for voxel in plan["voxels"][:20]), "voxel payload lacks xyz/cost/risk/metrics/normal")
    implicit = plan["implicitMap"]
    require(implicit["octreeLeaves"] > 0 and implicit["octreeNodes"] >= implicit["octreeLeaves"], "implicit octree summary is invalid")
    require(implicit["occupied"] >= implicit["finite"] > 0, "implicit traversability summary is invalid")
    require(len(implicit["normals"]) >= 20, "implicit normal-vector visualization is missing")
    require(len(implicit["gaussians"]) >= 20, "implicit Gaussian probability visualization is missing")
    require(all(len(gaussian) == 15 for gaussian in implicit["gaussians"][:20]), "Gaussian payload lacks center/axes/cost metrics")
    require({stage["key"] for stage in plan["stages"]} == {"cloud", "ndt", "risk", "global", "local", "stable", "recede"}, "pipeline stages do not match paper mainline")
    verify_editing_api(demo)


def verify_editing_api(demo: PaperInteractiveDemo) -> None:
    version = demo.terrain_version
    edited = demo.add_obstacle(-0.3, -0.2, radius=0.18, delta=-0.25, obstacle=False, roughness=0.12)
    require(edited["terrain"]["version"] > version, "terrain edit did not update the terrain version")
    require(len(edited["sim"]["dynamicBlocks"]) == 1, "terrain edit was not recorded")
    smoothed = demo.smooth_patch(-0.3, -0.2, radius=0.20)
    require(len(smoothed["sim"]["dynamicBlocks"]) == 2, "smooth edit was not recorded")
    cleared = demo.clear_obstacles()
    require(len(cleared["sim"]["dynamicBlocks"]) == 0, "terrain edits were not cleared")


def verify_frontend_source() -> None:
    require("Terrain-Aware Planning Workbench" in INDEX_HTML, "frontend title was not replaced with the rewritten workbench")
    require("class RenderQueue" in INDEX_HTML and "q.flush()" in INDEX_HTML, "frontend does not use a centralized depth-sorted render queue")
    require("const stageLayerMap" in INDEX_HTML and "focusStage" in INDEX_HTML, "frontend lacks stage-focused layer presets")
    require("主流程单步分解" in INDEX_HTML and "stage-strip" in INDEX_HTML, "frontend lacks clear pipeline step navigation")
    require("体素与可通行性" in INDEX_HTML and "costbar" in INDEX_HTML, "frontend lacks voxel/traversability presentation")
    require("function screenToTerrain" in INDEX_HTML, "frontend does not project screen points onto the 3D terrain surface")
    require("function fromScreenAtHeight" in INDEX_HTML, "frontend lacks height-aware screen unprojection")
    require("function updatePivot" in INDEX_HTML and "view.pivot" in INDEX_HTML, "frontend camera does not orbit around the terrain center")
    require("view.ox = canvas.width / 2" in INDEX_HTML, "frontend camera is not anchored to the viewport center")
    require("function rebuildCamera" in INDEX_HTML and "view.yaw" in INDEX_HTML and "view.pitch" in INDEX_HTML, "frontend camera does not use explicit orthographic yaw/pitch state")
    require("arcballVector" not in INDEX_HTML and "applyArcball" not in INDEX_HTML and "rotateBasis" not in INDEX_HTML, "frontend still allows free-roll arcball rotation")
    require("view.pitch = clamp" in INDEX_HTML and "82 * Math.PI / 180" in INDEX_HTML, "frontend does not constrain pitch without roll")
    require("return [p[0] * view.scale + view.ox, -p[1] * view.scale + view.oy]" in INDEX_HTML, "frontend projection still draws height upside down")
    require("up = [-sy * sp, cy * sp, cp]" in INDEX_HTML, "frontend camera basis still flips the default view")
    require("const capped = Math.min(18" in INDEX_HTML and "ordered = normals.map" in INDEX_HTML, "normal rendering is not depth-sorted and length-capped")
    require("function pushTerrainCell" in INDEX_HTML and "cells.sort((a, b) => b.order - a.order)" in INDEX_HTML, "terrain renderer is not using stable footprint-sorted heightfield cells")
    require("zRange" not in INDEX_HTML and "高度夸张" not in INDEX_HTML, "frontend still exposes an extra camera tilt/height exaggeration control")
    require("function drawVoxels" in INDEX_HTML and "costColor(v[3], v[4]" in INDEX_HTML, "voxel renderer does not encode traversal cost and risk")
    require("drawHover" in INDEX_HTML and "hoverTerrain" in INDEX_HTML, "frontend lacks terrain-pick hover preview")
    require('canvas.addEventListener("pointerup"' in INDEX_HTML and "applyEditAt(hoverTerrain.x" in INDEX_HTML, "frontend terrain editing is not bound to direct single-click picking")
    require('canvas.addEventListener("dblclick"' in INDEX_HTML, "frontend terrain editing is not bound to double-click picking")
    require("__DEV_HOT_RELOAD__" in INDEX_HTML and "pollDevVersion" in INDEX_HTML, "frontend hot reload polling is missing")


def verify_summary(summary: dict) -> None:
    require(summary["success"], "interactive mainline planning failed")
    require(summary["total_ms"] > 0.0, "missing total runtime")
    require(summary["global_ms"] > 0.0, "missing global runtime")
    require(summary["local_ms"] > 0.0, "missing local runtime")
    require(summary["raw_points"] >= 1000, "point-cloud input was not reported")
    require(summary["occupied_voxels"] >= 150, "occupied voxel count was not reported")
    require(summary["global_path_length_m"] >= 3.0, "global path is unexpectedly short")
    require(summary["local_path_length_m"] >= 1.4, "local path is unexpectedly short")
    require(summary["max_traversal_cost"] < 0.82, "global path crossed high traversal cost")
    require(summary["local_mean_risk"] < 0.65, "local path risk too high")
    require(summary["local_min_stability"] >= 0.28, "local path stability too low")
    require(summary["shared_traversable_voxels"] >= 150, "global traversability was not shared")
    require(summary["global_traversability_checks"] > 0, "local planner did not query global traversability")
    require(summary["global_normal_initializations"] > 0, "stability did not use NDT normals")
    require(summary["local_traversable_voxels"] > 0, "local traversable set was not built")
    require(summary["local_cells"] < summary["global_cells"], "local window was not cropped")


def terrain_height_at(terrain: dict, point: list[float]) -> float:
    col = max(0, min(terrain["cols"] - 1, round((point[0] - terrain["origin"][0]) / terrain["resolution"])))
    row = max(0, min(terrain["rows"] - 1, round((point[1] - terrain["origin"][1]) / terrain["resolution"])))
    return float(terrain["height"][row * terrain["cols"] + col])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"paper interactive verification failed: {message}")


if __name__ == "__main__":
    main()
