from __future__ import annotations

import argparse

import numpy as np
from rich.console import Console

from traversability.implicit_map import ImplicitTerrainMap
from traversability.terrain import generate_large_rough_terrain, grid_to_world, make_pyramid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify continuous implicit terrain map queries.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    height, obstacle, origin = generate_large_rough_terrain(seed=args.seed)
    layer = make_pyramid(height, obstacle, 0.25, origin).finest
    terrain_map = ImplicitTerrainMap(layer)
    rng = np.random.default_rng(args.seed)

    for _ in range(args.samples):
        row = int(rng.integers(0, layer.height.shape[0]))
        col = int(rng.integers(0, layer.height.shape[1]))
        xy = grid_to_world(layer, (row, col))
        query = terrain_map.query(xy)
        require(abs(query.height - layer.height[row, col]) < 1e-9, "height mismatch at grid point")
        require(abs(query.slope - layer.slope[row, col]) < 1e-9, "slope mismatch at grid point")
        if layer.obstacle[row, col]:
            require(query.obstacle, "obstacle grid point was not marked as obstacle")

    finite_queries = 0
    for _ in range(args.samples):
        x = origin[0] + rng.random() * layer.resolution * (layer.height.shape[1] - 1)
        y = origin[1] + rng.random() * layer.resolution * (layer.height.shape[0] - 1)
        query = terrain_map.query((x, y))
        require(np.isfinite(query.height), "continuous height query produced non-finite value")
        if not query.obstacle:
            require(np.isfinite(query.risk), "free continuous query produced non-finite risk")
            finite_queries += 1

    require(finite_queries > args.samples * 0.6, "too few finite free-space continuous queries")
    Console().print("[green]implicit map verification passed[/green]")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"implicit map verification failed: {message}")


if __name__ == "__main__":
    main()

