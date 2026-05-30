from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traversability.terrain import TerrainLayer, world_to_grid


@dataclass(frozen=True)
class TerrainQuery:
    x: float
    y: float
    height: float
    slope: float
    roughness: float
    step: float
    risk: float
    obstacle: bool


class ImplicitTerrainMap:
    """Continuous query facade over a terrain layer.

    The paper describes an efficient terrain analysis and planning layer built on
    an implicit map representation. This class gives the reproduction a similar
    boundary: planners and stability estimators can query continuous world
    coordinates without depending directly on grid indexing.
    """

    def __init__(self, layer: TerrainLayer):
        self.layer = layer

    @property
    def resolution(self) -> float:
        return self.layer.resolution

    @property
    def origin_xy(self) -> tuple[float, float]:
        return self.layer.origin_xy

    def query(self, xy: tuple[float, float]) -> TerrainQuery:
        x, y = xy
        height = bilinear(self.layer.height, self.layer, xy)
        slope = bilinear(self.layer.slope, self.layer, xy)
        roughness = bilinear(self.layer.roughness, self.layer, xy)
        step = bilinear(self.layer.step, self.layer, xy)
        risk = bilinear_with_inf(self.layer.risk, self.layer, xy)
        row, col = world_to_grid(self.layer, xy)
        obstacle = bool(self.layer.obstacle[row, col]) or not np.isfinite(risk)
        return TerrainQuery(
            x=x,
            y=y,
            height=float(height),
            slope=float(slope),
            roughness=float(roughness),
            step=float(step),
            risk=float(risk),
            obstacle=obstacle,
        )

    def is_free(self, xy: tuple[float, float], max_risk: float = 0.90) -> bool:
        query = self.query(xy)
        return not query.obstacle and np.isfinite(query.risk) and query.risk < max_risk

    def footprint_heights(self, points: list[tuple[float, float]]) -> tuple[np.ndarray, bool]:
        heights = []
        collision = False
        for point in points:
            query = self.query(point)
            heights.append(query.height)
            collision = collision or query.obstacle
        return np.asarray(heights, dtype=np.float64), collision


def bilinear(values: np.ndarray, layer: TerrainLayer, xy: tuple[float, float]) -> float:
    x, y = xy
    col = (x - layer.origin_xy[0]) / layer.resolution
    row = (y - layer.origin_xy[1]) / layer.resolution
    row0 = int(np.floor(np.clip(row, 0, values.shape[0] - 1)))
    col0 = int(np.floor(np.clip(col, 0, values.shape[1] - 1)))
    row1 = min(row0 + 1, values.shape[0] - 1)
    col1 = min(col0 + 1, values.shape[1] - 1)
    tr = float(np.clip(row - row0, 0.0, 1.0))
    tc = float(np.clip(col - col0, 0.0, 1.0))
    v00 = values[row0, col0]
    v01 = values[row0, col1]
    v10 = values[row1, col0]
    v11 = values[row1, col1]
    top = (1.0 - tc) * v00 + tc * v01
    bottom = (1.0 - tc) * v10 + tc * v11
    return float((1.0 - tr) * top + tr * bottom)


def bilinear_with_inf(values: np.ndarray, layer: TerrainLayer, xy: tuple[float, float]) -> float:
    x, y = xy
    col = (x - layer.origin_xy[0]) / layer.resolution
    row = (y - layer.origin_xy[1]) / layer.resolution
    row0 = int(np.floor(np.clip(row, 0, values.shape[0] - 1)))
    col0 = int(np.floor(np.clip(col, 0, values.shape[1] - 1)))
    row1 = min(row0 + 1, values.shape[0] - 1)
    col1 = min(col0 + 1, values.shape[1] - 1)
    window = values[[row0, row0, row1, row1], [col0, col1, col0, col1]]
    if not np.all(np.isfinite(window)):
        return float("inf")
    return bilinear(values, layer, xy)

