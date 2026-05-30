from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from rich.console import Console

from traversability.pointcloud import load_point_cloud, point_cloud_to_elevation_grid, sample_point_cloud_from_grid, save_point_cloud
from traversability.terrain import generate_large_rough_terrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify point cloud file IO formats.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    height, obstacle, origin = generate_large_rough_terrain(size=28, resolution=0.25, seed=args.seed)
    points = sample_point_cloud_from_grid(height, obstacle, 0.25, origin, seed=args.seed)
    points = points[:500]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for suffix in [".csv", ".xyz", ".npy", ".ply"]:
            path = root / f"points{suffix}"
            save_point_cloud(path, points)
            loaded = load_point_cloud(path)
            require(loaded.shape == points.shape, f"{suffix} shape mismatch")
            require(np.allclose(loaded, points, atol=1e-5), f"{suffix} points mismatch")
            grid = point_cloud_to_elevation_grid(loaded, resolution=0.25)
            require(grid.height.size > 0, f"{suffix} produced empty grid")
            require(np.count_nonzero(grid.points_per_cell) > 0, f"{suffix} produced no observed cells")

    Console().print("[green]point cloud IO verification passed[/green]")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"point cloud IO verification failed: {message}")


if __name__ == "__main__":
    main()

