from __future__ import annotations

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap
from traversability.terrain import analyze_layer
from traversability.tracked_stability import estimate_tracked_configuration_stability


def main() -> None:
    flat_map = make_map("flat")
    flat = estimate_tracked_configuration_stability(flat_map, (0.0, 0.0), 0.0)
    require(flat.feasible, "flat terrain should be feasible")
    require(flat.stable_without_flippers, "flat terrain should be stable on main tracks")
    require(flat.main_contact_count >= 12, "flat terrain should have broad main-track contact")
    require(flat.support_area > 0.20, "flat support polygon area is too small")
    require(abs(flat.front_flipper_angle) < 0.15, "flat terrain should not need front flipper support")
    require(abs(flat.rear_flipper_angle) < 0.15, "flat terrain should not need rear flipper support")

    ledge_map = make_map("ledge")
    ledge = estimate_tracked_configuration_stability(ledge_map, (0.0, 0.0), 0.0)
    require(ledge.stable_with_flippers, "ledge terrain should become stable with flipper support")
    require(ledge.feasible, "ledge terrain should be feasible after flipper support")
    require(abs(ledge.front_flipper_angle) > 0.20, "ledge terrain should request a front flipper angle")
    require(ledge.support_area > flat.support_area * 0.55, "flipper support polygon is too small")

    collision_map = make_map("body_collision")
    collision = estimate_tracked_configuration_stability(collision_map, (0.0, 0.0), 0.0)
    require(collision.body_collision, "body obstacle should trigger collision")
    require(not collision.feasible, "body collision should make the configuration infeasible")
    require(collision.stability < flat.stability, "collision should reduce stability score")

    print("tracked stability verification passed")


def make_map(kind: str) -> ImplicitTerrainMap:
    resolution = 0.04
    size = 96
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    xs = origin[0] + np.arange(size) * resolution
    ys = origin[1] + np.arange(size) * resolution
    xx, yy = np.meshgrid(xs, ys)
    height = np.zeros((size, size), dtype=np.float64)

    if kind == "ledge":
        height += np.where(xx > 0.22, 0.22, 0.0)
    elif kind == "body_collision":
        height += 0.24 * np.exp(-((xx / 0.11) ** 2 + (yy / 0.11) ** 2))

    obstacle = np.zeros_like(height, dtype=bool)
    layer = analyze_layer(kind, height, obstacle, resolution, origin)
    return ImplicitTerrainMap(layer)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"tracked stability verification failed: {message}")


if __name__ == "__main__":
    main()
