from __future__ import annotations

import math

import numpy as np

from traversability.ndt_map import NDTConfig, NDTImplicitMap, roughness_and_slope


def main() -> None:
    points = make_test_cloud()
    ndt_map = NDTImplicitMap(
        points,
        NDTConfig(
            voxel_size=0.25,
            fusion_radius=0.55,
            saturation_count=3,
            slope_threshold_rad=math.radians(30.0),
            complexity_threshold=0.45,
            robot_radius=0.35,
            robot_height=0.55,
        ),
    )
    metrics = ndt_map.compute_metrics()
    require(len(ndt_map.cells) > 200, "too few NDT cells were created")
    require(len(metrics) > 150, "too few saturated occupied NDT cells were analyzed")

    flat_key = nearest_metric(metrics, np.array([-1.2, 0.0, 0.0]))
    ramp_key = nearest_metric(metrics, np.array([1.2, 0.2, 0.35]))
    edge_key = nearest_metric(metrics, np.array([0.0, 0.0, 0.9]))
    hole_key = nearest_metric(metrics, np.array([-0.45, 1.1, 0.0]))

    flat = metrics[flat_key]
    ramp = metrics[ramp_key]
    edge = metrics[edge_key]
    hole = metrics[hole_key]

    require(flat.slope < math.radians(15.0), "flat patch slope is too high")
    require(ramp.slope > math.radians(12.0), "ramp slope did not increase")
    require(edge.complexity > flat.complexity, "step edge did not increase terrain complexity")
    require(hole.sparsity > flat.sparsity, "undersampled patch did not increase sparsity")
    require(any(metric.terrain_risk for metric in metrics.values()), "terrain risk was never detected")
    require(any(metric.collision_risk for metric in metrics.values()), "collision risk was never detected")
    require(any(metric.falling_risk for metric in metrics.values()), "falling risk was never detected")
    verify_incremental_integration(points, ndt_map.config)

    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, 12), np.linspace(-1.0, 1.0, 12))
    plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    covariance = np.cov(plane.T) + np.eye(3) * 1e-8
    normal, roughness, slope = roughness_and_slope(covariance)
    require(abs(normal[2]) > 0.99, "SVD normal does not align with z for a flat line-supported patch")
    require(0.0 <= roughness <= 1.0, "roughness is outside normalized range")
    require(slope < math.radians(1.0), "SVD slope for flat patch is too high")

    print("NDT implicit map verification passed")


def verify_incremental_integration(points: np.ndarray, config: NDTConfig) -> None:
    origin = np.min(points, axis=0)
    batch = NDTImplicitMap(points, config, origin=origin)
    incremental = NDTImplicitMap.empty(origin, config)
    for frame in np.array_split(points, 4):
        incremental.integrate_points(frame)

    require(batch.occupied == incremental.occupied, "incremental occupied voxel set differs from batch map")
    require(set(batch.cells) == set(incremental.cells), "incremental cell set differs from batch map")
    for key in batch.cells:
        batch_cell = batch.cells[key]
        inc_cell = incremental.cells[key]
        require(batch_cell.count == inc_cell.count, f"incremental count mismatch at {key}")
        require(np.allclose(batch_cell.mean, inc_cell.mean, atol=1e-10), f"incremental mean mismatch at {key}")
        require(np.allclose(batch_cell.covariance, inc_cell.covariance, atol=1e-7), f"incremental covariance mismatch at {key}")

    batch_metrics = batch.compute_metrics()
    incremental_metrics = incremental.compute_metrics()
    require(set(batch_metrics) == set(incremental_metrics), "incremental metrics keys differ from batch map")
    sample_keys = sorted(batch_metrics)[:20]
    for key in sample_keys:
        require(
            abs(batch_metrics[key].complexity - incremental_metrics[key].complexity) < 1e-6,
            f"incremental complexity mismatch at {key}",
        )


def make_test_cloud() -> np.ndarray:
    rng = np.random.default_rng(11)
    xs = np.arange(-2.0, 2.01, 0.08)
    ys = np.arange(-2.0, 2.01, 0.08)
    points = []
    for x in xs:
        for y in ys:
            if -0.7 < x < -0.15 and 0.75 < y < 1.35:
                continue
            z = 0.03 * rng.normal()
            if x > 0.45:
                z += 0.35 * (x - 0.45)
            if -0.12 < x < 0.16 and -0.55 < y < 0.55:
                z += 0.75
            samples = 3 if (x < 0.2 or y < -0.2) else 2
            jitter = rng.normal(0.0, 0.018, size=(samples, 3))
            base = np.array([x, y, z])
            points.append(base + jitter)
    return np.vstack(points).astype(np.float64)


def nearest_metric(metrics, xyz: np.ndarray) -> tuple[int, int, int]:
    return min(metrics, key=lambda key: float(np.linalg.norm(metrics[key].center - xyz)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NDT map verification failed: {message}")


if __name__ == "__main__":
    main()
