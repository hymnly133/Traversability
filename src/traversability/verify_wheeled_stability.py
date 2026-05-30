from __future__ import annotations

import numpy as np

from traversability.implicit_map import ImplicitTerrainMap
from traversability.terrain import analyze_layer
from traversability.wheeled_stability import estimate_wheeled_configuration_stability


def main() -> None:
    flat = estimate_wheeled_configuration_stability(make_map("flat"), (0.0, 0.0), 0.0)
    require(flat.feasible, "flat terrain should be feasible")
    require(flat.wheel_contact_count >= 4, "flat terrain should contact all wheels")
    require(flat.com_inside_support, "flat terrain support polygon should contain CoM projection")
    require(flat.support_area > 0.16, "flat support polygon area is too small")

    side_step = estimate_wheeled_configuration_stability(make_map("side_step"), (0.0, 0.0), 0.0)
    require(side_step.feasible, "side step should still be feasible for the wheeled robot")
    require(side_step.wheel_contact_count >= 3, "side step should preserve at least three wheel contacts")
    require(abs(side_step.roll) > 0.05, "side step should induce a roll estimate")

    collision = estimate_wheeled_configuration_stability(make_map("body_collision"), (0.0, 0.0), 0.0)
    require(collision.body_collision, "body obstacle should trigger collision")
    require(not collision.feasible, "body collision should make wheeled configuration infeasible")

    trench = estimate_wheeled_configuration_stability(make_map("trench"), (0.0, 0.0), 0.0)
    require(trench.wheel_contact_count < 3 or not trench.com_inside_support, "trench should degrade wheel support")
    require(not trench.feasible, "trench should be infeasible")

    print("wheeled stability verification passed")


def make_map(kind: str) -> ImplicitTerrainMap:
    resolution = 0.04
    size = 96
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0)
    xs = origin[0] + np.arange(size) * resolution
    ys = origin[1] + np.arange(size) * resolution
    xx, yy = np.meshgrid(xs, ys)
    height = np.zeros((size, size), dtype=np.float64)

    if kind == "side_step":
        height += np.where(yy > 0.0, 0.08, 0.0)
    elif kind == "body_collision":
        height += 0.22 * np.exp(-((xx / 0.10) ** 2 + (yy / 0.10) ** 2))
    elif kind == "trench":
        height -= np.where((xx > -0.33) & (xx < 0.03) & (yy < -0.05), 0.22, 0.0)
        height -= np.where((xx > -0.33) & (xx < 0.03) & (yy > 0.05), 0.22, 0.0)
        height += np.where(xx > 0.12, 0.06, 0.0)

    obstacle = np.zeros_like(height, dtype=bool)
    return ImplicitTerrainMap(analyze_layer(kind, height, obstacle, resolution, origin))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"wheeled stability verification failed: {message}")


if __name__ == "__main__":
    main()
