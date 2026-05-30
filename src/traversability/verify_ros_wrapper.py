from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console

from traversability.pointcloud import sample_point_cloud_from_grid, save_point_cloud
from traversability.terrain import generate_large_rough_terrain


REPO_ROOT = Path(__file__).resolve().parents[2]
ROS_PACKAGE = REPO_ROOT / "ros" / "traversability_planner"


def main() -> None:
    verify_manifest()
    verify_cmake()
    verify_launch()
    verify_node_cli()
    Console().print("[green]ROS wrapper verification passed[/green]")


def verify_manifest() -> None:
    package_xml = ROS_PACKAGE / "package.xml"
    require(package_xml.exists(), "missing ROS package.xml")
    root = ET.parse(package_xml).getroot()
    require(root.findtext("name") == "traversability_planner", "unexpected ROS package name")
    deps = {element.text for element in root.findall("exec_depend")}
    for dep in ["rospy", "nav_msgs", "geometry_msgs", "sensor_msgs"]:
        require(dep in deps, f"missing ROS dependency {dep}")


def verify_cmake() -> None:
    cmake = ROS_PACKAGE / "CMakeLists.txt"
    require(cmake.exists(), "missing CMakeLists.txt")
    content = cmake.read_text(encoding="utf-8")
    require("catkin_install_python" in content, "CMakeLists does not install Python node")
    require("terrain_aware_planner_node.py" in content, "planner node not listed in CMakeLists")


def verify_launch() -> None:
    launch = ROS_PACKAGE / "launch" / "offline_pointcloud_planner.launch"
    require(launch.exists(), "missing offline launch file")
    root = ET.parse(launch).getroot()
    nodes = root.findall("node")
    require(any(node.attrib.get("type") == "terrain_aware_planner_node.py" for node in nodes), "launch file missing planner node")


def verify_node_cli() -> None:
    height, obstacle, origin = generate_large_rough_terrain(size=80, resolution=0.25, seed=7)
    points = sample_point_cloud_from_grid(height, obstacle, 0.25, origin, seed=7)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cloud = root / "terrain.xyz"
        output = root / "out"
        save_point_cloud(cloud, points)
        node = ROS_PACKAGE / "scripts" / "terrain_aware_planner_node.py"
        subprocess.run(
            [
                sys.executable,
                str(node),
                "--pointcloud-file",
                str(cloud),
                "--output-dir",
                str(output),
                "--resolution",
                "0.25",
                "--levels",
                "4",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        summary = output / "ros_summary.csv"
        path = output / "ros_planned_path.csv"
        require(summary.exists(), "ROS wrapper did not write summary")
        require(path.exists(), "ROS wrapper did not write path")
        with summary.open(encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        require(row["success"] == "1", "ROS wrapper planner failed in CLI mode")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ROS wrapper verification failed: {message}")


if __name__ == "__main__":
    main()

