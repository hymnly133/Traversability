from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


EXPECTED_SCENARIOS = {"stairs", "rubble", "grass", "hill", "bridge", "field"}


def main() -> None:
    output = Path("runs/paper_scenarios_verify")
    subprocess.run([sys.executable, "-m", "traversability.paper_scenario_suite", "--output", str(output)], check=True)

    scenarios_path = output / "paper_scenarios.csv"
    summary_path = output / "paper_scenarios_summary.csv"
    require(scenarios_path.exists(), "missing paper scenario rows")
    require(summary_path.exists(), "missing paper scenario summary")

    with scenarios_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    with summary_path.open(encoding="utf-8") as file:
        summary = {row["metric"]: row["value"] for row in csv.DictReader(file)}

    names = {row["scenario"] for row in rows}
    require(names == EXPECTED_SCENARIOS, f"unexpected scenario set: {names}")
    require(int(float(summary["scenarios"])) == len(EXPECTED_SCENARIOS), "scenario count mismatch")
    require(float(summary["success_rate"]) >= 1.0, "paper scenario success rate too low")
    require(float(summary["mean_runtime_ms"]) <= 2200.0, "paper scenario mean runtime too high")
    require(float(summary["max_traversal_cost"]) < 0.90, "paper scenario traversal cost too high")
    require(float(summary["mean_traversable_voxels"]) >= 250.0, "too few traversable voxels")
    require(0.0 <= float(summary["mean_roughness"]) <= 1.0, "scenario roughness summary is invalid")
    require(float(summary["mean_slope_rad"]) >= 0.0, "scenario slope summary is invalid")
    require(0.0 <= float(summary["mean_sparsity"]) <= 1.0, "scenario sparsity summary is invalid")
    require(float(summary["mean_complexity"]) > 0.0, "scenario complexity summary is invalid")

    for row in rows:
        require(row["success"] == "1", f"{row['scenario']} failed")
        require(float(row["path_length_m"]) >= 4.0, f"{row['scenario']} path is too short")
        require(int(float(row["path_points"])) >= 10, f"{row['scenario']} path has too few points")
        require(int(float(row["expanded_nodes"])) >= 40, f"{row['scenario']} expanded too few nodes")
        require(float(row["runtime_ms"]) <= 4200.0, f"{row['scenario']} runtime too high")
        require(0.0 <= float(row["mean_roughness"]) <= 1.0, f"{row['scenario']} roughness invalid")
        require(float(row["mean_slope_rad"]) >= 0.0, f"{row['scenario']} slope invalid")
        require(0.0 <= float(row["mean_sparsity"]) <= 1.0, f"{row['scenario']} sparsity invalid")
        require(float(row["mean_complexity"]) > 0.0, f"{row['scenario']} complexity invalid")
        require(int(float(row["finite_cost_voxels"])) > 0, f"{row['scenario']} has no finite-cost voxels")
        image = output / f"{row['scenario']}.png"
        require(image.exists() and image.stat().st_size > 10_000, f"missing or empty {image}")

    print("paper scenario suite verification passed")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"paper scenario verification failed: {message}")


if __name__ == "__main__":
    main()
