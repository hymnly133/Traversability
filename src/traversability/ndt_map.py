from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from traversability.pointcloud import validate_points


@dataclass(frozen=True)
class NDTConfig:
    voxel_size: float = 0.4
    fusion_radius: float = 0.75
    saturation_count: int = 4
    roughness_weight: float = 0.35
    slope_weight: float = 0.45
    sparsity_weight: float = 0.20
    roughness_threshold: float = 0.55
    slope_threshold_rad: float = math.radians(35.0)
    sparsity_threshold: float = 0.65
    complexity_threshold: float = 1.0
    robot_radius: float = 0.45
    robot_height: float = 0.60


@dataclass(frozen=True)
class NDTCell:
    key: tuple[int, int, int]
    count: int
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class NDTMetric:
    key: tuple[int, int, int]
    center: np.ndarray
    normal: np.ndarray
    count: int
    roughness: float
    slope: float
    sparsity: float
    complexity: float
    terrain_risk: bool
    collision_risk: bool
    falling_risk: bool

    @property
    def traversal_risk(self) -> bool:
        return self.terrain_risk or self.collision_risk or self.falling_risk

    @property
    def traversal_cost(self) -> float:
        return float("inf") if self.traversal_risk else self.complexity


@dataclass
class OctreeNode:
    children: dict[int, "OctreeNode"]
    keys: set[tuple[int, int, int]]


class SparseOctreeIndex:
    """Sparse octree over integer voxel keys for the global volumetric map."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.root = OctreeNode(children={}, keys=set())
        self.keys: set[tuple[int, int, int]] = set()
        self.offset = 1 << (max_depth - 1)
        self.width = 1 << max_depth

    def insert(self, key: tuple[int, int, int]) -> None:
        if key in self.keys:
            return
        self.keys.add(key)
        node = self.root
        node.keys.add(key)
        ix, iy, iz = self.positive_key(key)
        for shift in range(self.max_depth - 1, -1, -1):
            child_index = ((ix >> shift) & 1) << 2 | ((iy >> shift) & 1) << 1 | ((iz >> shift) & 1)
            node = node.children.setdefault(child_index, OctreeNode(children={}, keys=set()))
            node.keys.add(key)

    def contains(self, key: tuple[int, int, int]) -> bool:
        return key in self.keys

    def neighborhood(
        self,
        center: tuple[int, int, int],
        radius: int,
    ) -> list[tuple[int, int, int]]:
        lower = (center[0] - radius, center[1] - radius, center[2] - radius)
        upper = (center[0] + radius, center[1] + radius, center[2] + radius)
        return [key for key in self.range_query(lower, upper)]

    def range_query(
        self,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        result: list[tuple[int, int, int]] = []
        self.collect_range(self.root, (0, 0, 0), self.width, lower, upper, result)
        return result

    def collect_range(
        self,
        node: OctreeNode,
        origin: tuple[int, int, int],
        size: int,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
        result: list[tuple[int, int, int]],
    ) -> None:
        node_min = (origin[0] - self.offset, origin[1] - self.offset, origin[2] - self.offset)
        node_max = (node_min[0] + size - 1, node_min[1] + size - 1, node_min[2] + size - 1)
        if any(node_max[axis] < lower[axis] or node_min[axis] > upper[axis] for axis in range(3)):
            return
        if all(lower[axis] <= node_min[axis] and node_max[axis] <= upper[axis] for axis in range(3)):
            result.extend(node.keys)
            return
        if not node.children:
            result.extend(key for key in node.keys if all(lower[axis] <= key[axis] <= upper[axis] for axis in range(3)))
            return
        half = size // 2
        for child_index, child in node.children.items():
            child_origin = (
                origin[0] + (half if child_index & 4 else 0),
                origin[1] + (half if child_index & 2 else 0),
                origin[2] + (half if child_index & 1 else 0),
            )
            self.collect_range(child, child_origin, half, lower, upper, result)

    def positive_key(self, key: tuple[int, int, int]) -> tuple[int, int, int]:
        shifted = (key[0] + self.offset, key[1] + self.offset, key[2] + self.offset)
        if any(value < 0 or value >= self.width for value in shifted):
            raise ValueError(f"voxel key {key} exceeds octree depth {self.max_depth}")
        return shifted

    @property
    def leaf_count(self) -> int:
        return len(self.keys)

    @property
    def node_count(self) -> int:
        return count_octree_nodes(self.root)


class NDTImplicitMap:
    """Sparse NDT voxel map matching the paper's global implicit-map boundary."""

    def __init__(self, points: np.ndarray | None = None, config: NDTConfig | None = None, origin: np.ndarray | None = None):
        self.config = config or NDTConfig()
        self.metrics: dict[tuple[int, int, int], NDTMetric] = {}
        self.connected_cache: dict[tuple[int, frozenset[tuple[int, int, int]]], object] = {}
        if points is None:
            if origin is None:
                raise ValueError("origin is required when constructing an empty NDT map")
            self.points = np.empty((0, 3), dtype=np.float64)
            self.origin = np.asarray(origin, dtype=np.float64)
            if self.origin.shape != (3,):
                raise ValueError("origin must be a 3-vector")
            self.cells: dict[tuple[int, int, int], NDTCell] = {}
            self.octree = SparseOctreeIndex()
        else:
            self.points = validate_points(points)
            self.origin = np.asarray(origin, dtype=np.float64) if origin is not None else np.min(self.points, axis=0)
            self.cells = {}
            self.octree = SparseOctreeIndex()
            self.integrate_points(self.points)
        self.occupied = self.occupied_keys()

    @classmethod
    def empty(cls, origin: tuple[float, float, float] | np.ndarray, config: NDTConfig | None = None) -> "NDTImplicitMap":
        return cls(points=None, config=config, origin=np.asarray(origin, dtype=np.float64))

    def integrate_points(self, points: np.ndarray) -> None:
        new_points = validate_points(points)
        for point in new_points:
            key = self.key_from_xyz(point)
            self.cells[key] = update_cell(self.cells.get(key), key, point)
            self.octree.insert(key)
        self.points = np.vstack([self.points, new_points]) if len(self.points) else new_points.copy()
        self.occupied = self.occupied_keys()
        self.metrics = {}
        self.connected_cache = {}

    def occupied_keys(self) -> set[tuple[int, int, int]]:
        return {key for key, cell in self.cells.items() if cell.count >= self.config.saturation_count}

    def key_from_xyz(self, xyz: np.ndarray) -> tuple[int, int, int]:
        index = np.floor((np.asarray(xyz, dtype=np.float64) - self.origin) / self.config.voxel_size).astype(np.int64)
        return int(index[0]), int(index[1]), int(index[2])

    def compute_metrics(self) -> dict[tuple[int, int, int], NDTMetric]:
        if self.metrics:
            return self.metrics
        partial_metrics = {}
        for key in sorted(self.occupied):
            cell = self.cells[key]
            fused = self.fuse_neighborhood(key)
            if fused is None:
                continue
            count, mean, covariance = fused
            normal, roughness, slope = roughness_and_slope(covariance)
            sparsity = self.visible_sparsity(key, normal)
            complexity = terrain_complexity(roughness, slope, sparsity, self.config)
            partial_metrics[key] = NDTMetric(
                key=key,
                center=mean,
                normal=normal,
                count=count,
                roughness=roughness,
                slope=slope,
                sparsity=sparsity,
                complexity=complexity,
                terrain_risk=False,
                collision_risk=False,
                falling_risk=False,
            )
        self.metrics = partial_metrics
        metrics = {}
        for key, metric in partial_metrics.items():
            terrain_risk, collision_risk, falling_risk = self.assess_traversal_risk(key, metric.complexity)
            metrics[key] = NDTMetric(
                key=metric.key,
                center=metric.center,
                normal=metric.normal,
                count=metric.count,
                roughness=metric.roughness,
                slope=metric.slope,
                sparsity=metric.sparsity,
                complexity=metric.complexity,
                terrain_risk=terrain_risk,
                collision_risk=collision_risk,
                falling_risk=falling_risk,
            )
        self.metrics = metrics
        return self.metrics

    def fuse_neighborhood(self, key: tuple[int, int, int]) -> tuple[int, np.ndarray, np.ndarray] | None:
        radius_cells = max(1, int(math.ceil(self.config.fusion_radius / self.config.voxel_size)))
        cells = []
        for neighbor_key in self.octree.neighborhood(key, radius_cells):
            cell = self.cells.get(neighbor_key)
            if cell is not None and cell.count > 0:
                cells.append(cell)
        if not cells:
            return None
        return fuse_cells(cells)

    def visible_sparsity(self, key: tuple[int, int, int], normal: np.ndarray) -> float:
        radius_cells = max(1, int(math.ceil(self.config.fusion_radius / self.config.voxel_size)))
        center = self.voxel_center(key)
        densities = []
        for neighbor_key in neighbor_keys(key, radius_cells):
            if neighbor_key == key:
                continue
            offset = self.voxel_center(neighbor_key) - center
            distance = float(np.linalg.norm(offset))
            if distance <= 1e-9 or distance > self.config.fusion_radius:
                continue
            if float(np.dot(offset / distance, normal)) > 0.15:
                continue
            count = self.cells.get(neighbor_key).count if neighbor_key in self.cells else 0
            densities.append(min(1.0, count / max(self.config.saturation_count, 1)))
        if not densities:
            return 1.0
        return float(np.clip(1.0 - np.mean(densities), 0.0, 1.0))

    def assess_traversal_risk(self, key: tuple[int, int, int], complexity: float) -> tuple[bool, bool, bool]:
        center = self.voxel_center(key)
        terrain_risk = complexity > self.config.complexity_threshold
        collision_risk = False
        falling_risk = False
        for checkpoint in circular_checkpoints(center, self.config.robot_radius, self.config.voxel_size):
            hit_key = self.cast_down(checkpoint, start_z=center[2] + self.config.robot_height)
            if hit_key is None:
                falling_risk = True
                continue
            hit_metric = self.metrics.get(hit_key)
            if hit_metric is not None and hit_metric.complexity > self.config.complexity_threshold:
                terrain_risk = True
            hit_center = self.voxel_center(hit_key)
            horizontal = max(float(np.linalg.norm(hit_center[:2] - center[:2])), self.config.voxel_size)
            tilt = abs(math.atan2(float(hit_center[2] - center[2]), horizontal))
            if tilt > self.config.slope_threshold_rad:
                collision_risk = True
        return terrain_risk, collision_risk, falling_risk

    def cast_down(self, xy: np.ndarray, start_z: float) -> tuple[int, int, int] | None:
        ix = int(math.floor((xy[0] - self.origin[0]) / self.config.voxel_size))
        iy = int(math.floor((xy[1] - self.origin[1]) / self.config.voxel_size))
        iz_start = int(math.floor((start_z - self.origin[2]) / self.config.voxel_size))
        min_z = min(key[2] for key in self.occupied)
        for iz in range(iz_start, min_z - 1, -1):
            key = (ix, iy, iz)
            if key in self.occupied:
                return key
        return None

    def voxel_center(self, key: tuple[int, int, int]) -> np.ndarray:
        return self.origin + (np.asarray(key, dtype=np.float64) + 0.5) * self.config.voxel_size


def build_cells(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> dict[tuple[int, int, int], NDTCell]:
    indices = np.floor((points - origin) / voxel_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for index, point in zip(indices, points):
        buckets.setdefault((int(index[0]), int(index[1]), int(index[2])), []).append(point)

    cells = {}
    for key, bucket in buckets.items():
        values = np.asarray(bucket, dtype=np.float64)
        mean = np.mean(values, axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(len(values), 1)
        covariance += np.eye(3) * 1e-8
        cells[key] = NDTCell(key=key, count=len(values), mean=mean, covariance=covariance)
    return cells


def count_octree_nodes(node: OctreeNode) -> int:
    return 1 + sum(count_octree_nodes(child) for child in node.children.values())


def update_cell(cell: NDTCell | None, key: tuple[int, int, int], point: np.ndarray) -> NDTCell:
    point = np.asarray(point, dtype=np.float64)
    if cell is None:
        return NDTCell(key=key, count=1, mean=point.copy(), covariance=np.zeros((3, 3), dtype=np.float64))
    old_count = cell.count
    count = old_count + 1
    delta = point - cell.mean
    mean = cell.mean + delta / count
    covariance = (old_count / count) * (cell.covariance + np.outer(delta, delta) / count)
    return NDTCell(key=key, count=count, mean=mean, covariance=covariance)


def fuse_cells(cells: Iterable[NDTCell]) -> tuple[int, np.ndarray, np.ndarray]:
    cells = list(cells)
    total_count = sum(cell.count for cell in cells)
    if total_count <= 0:
        raise ValueError("cannot fuse empty NDT cells")
    mean = sum(cell.count * cell.mean for cell in cells) / total_count
    covariance = np.zeros((3, 3), dtype=np.float64)
    for cell in cells:
        delta = cell.mean - mean
        covariance += cell.count * (cell.covariance + np.outer(delta, delta))
    covariance /= total_count
    covariance += np.eye(3) * 1e-8
    return total_count, mean, covariance


def roughness_and_slope(covariance: np.ndarray) -> tuple[np.ndarray, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    lambda_1 = float(eigenvalues[0])
    lambda_2 = float(eigenvalues[1])
    roughness = 1.0 - (lambda_2 - lambda_1) / max(lambda_2 + lambda_1, 1e-9)
    slope = math.acos(float(np.clip(np.dot(normal, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
    return normal, float(np.clip(roughness, 0.0, 1.0)), slope


def terrain_complexity(roughness: float, slope: float, sparsity: float, config: NDTConfig) -> float:
    return float(
        config.roughness_weight * roughness / max(config.roughness_threshold, 1e-9)
        + config.slope_weight * slope / max(config.slope_threshold_rad, 1e-9)
        + config.sparsity_weight * sparsity / max(config.sparsity_threshold, 1e-9)
    )


def neighbor_keys(key: tuple[int, int, int], radius: int) -> Iterable[tuple[int, int, int]]:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                yield key[0] + dx, key[1] + dy, key[2] + dz


def circular_checkpoints(center: np.ndarray, radius: float, voxel_size: float) -> list[np.ndarray]:
    circumference = 2.0 * math.pi * radius
    count = max(8, int(math.ceil(circumference / max(voxel_size, 1e-6))))
    checkpoints = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        checkpoints.append(center[:2] + radius * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64))
    return checkpoints
