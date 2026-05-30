# Reproduction Notes

## Source Status

Target paper:

```text
Real-Time Multilevel Terrain-Aware Path Planning for Ground Mobile Robots in Large-Scale Rough Terrains
IEEE Transactions on Robotics, 2025
DOI: 10.1109/TRO.2025.3577015
```

The authors' public GitHub repository is:

```text
https://github.com/HITSZ-NRSL/terrain-aware-planning
```

At the time of this reproduction pass, that repository contains only a README with dependencies, Docker section headings, and citation information. It does not expose buildable source code, launch files, sample rosbags, or parameter files.

## Implemented Reproduction Scope

This repository therefore implements an independent prototype that reproduces the paper's system shape rather than copying unavailable code:

- large rough terrain represented as an elevation grid
- implicit terrain-map facade for continuous height, slope, roughness, step, risk, and obstacle queries
- terrain pyramid with multiple map resolutions
- terrain-aware risk from slope, roughness, step height, and obstacle proximity
- coarse-to-fine A* planning through the pyramid
- robot footprint configuration-stability estimation from sampled local support geometry
- local iterative shortcut smoothing constrained by terrain risk and geometric feasibility
- stability sampling along the final path
- rolling-window replanning with a dynamic obstacle injected mid-run
- ablation benchmark comparing multilevel planning against a single-level full-resolution baseline
- pure-pursuit path tracking simulation with execution risk and stability checks
- point-cloud to elevation-grid import path that approximates the PCL/SLAM map input boundary
- multi-seed benchmark suite for aggregate success-rate, speed, risk, and feasibility statistics
- CSV and PNG artifacts for inspection
- verification script with numeric gates

## Current Verification

Run:

```powershell
.venv\Scripts\python.exe -m traversability.verify_reproduction
```

Expected verification gates:

- implicit map grid-point and continuous-query checks pass
- planner reports success
- `max_risk <= 0.90`
- `min_stability >= 0.08`
- `feasible_rate >= 0.80`
- path length is not trivially short
- summary, path, stability samples, and visualization files exist

Full verification with rolling replanning:

```powershell
.venv\Scripts\python.exe -m traversability.verify_reproduction --check-realtime --check-ablation --check-tracking --check-pointcloud --check-benchmark
```

Additional gates:

- at least four replanning cycles
- rolling success rate is at least 75%
- rolling p95 runtime is at most 5000 ms
- replanning summary, executed trajectory, and visualization files exist
- multilevel planner is at least 1.5x faster than the single-level baseline in the ablation scene
- multilevel planner expands at most two thirds as many nodes as the single-level baseline
- tracking mean error is at most 0.20 m and final error is at most 0.55 m
- tracked trajectory max risk is at most 0.95 and feasible rate is at least 75%
- point-cloud-derived elevation planning succeeds with max risk at most 0.90
- benchmark suite reaches at least 95% multilevel success and at least 1.5x mean speedup

## Known Gaps Against the Full Paper

The current prototype is not yet a full ROS Noetic reproduction. Missing parts include:

- real sensor or rosbag input
- full PCL/OpenCV/Ceres terrain mapping pipeline
- kinodynamic constraints and closed-loop tracking
- real low-level motor control and actuator limits
- online replanning benchmark on real large-scale rough-terrain data
