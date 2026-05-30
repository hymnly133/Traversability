from __future__ import annotations

import subprocess
import sys

from rich.console import Console


def main() -> None:
    modules = [
        "traversability.verify_ndt_map",
        "traversability.verify_ndt_planner",
        "traversability.verify_tracked_stability",
        "traversability.verify_wheeled_stability",
        "traversability.verify_hybrid_local_planner",
        "traversability.verify_paper_pipeline",
        "traversability.verify_paper_receding",
        "traversability.verify_paper_scenarios",
    ]
    for module in modules:
        subprocess.run([sys.executable, "-m", module], check=True)
    Console().print("[green]paper mainline verification passed[/green]")


if __name__ == "__main__":
    main()
