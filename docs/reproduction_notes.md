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
- terrain pyramid with multiple map resolutions
- terrain-aware risk from slope, roughness, step height, and obstacle proximity
- coarse-to-fine A* planning through the pyramid
- shortcut smoothing constrained by terrain risk
- stability sampling along the final path
- CSV and PNG artifacts for inspection
- verification script with numeric gates

## Current Verification

Run:

```powershell
.venv\Scripts\python.exe -m traversability.verify_reproduction
```

Expected verification gates:

- planner reports success
- `max_risk <= 0.90`
- `min_stability >= 0.08`
- path length is not trivially short
- summary, path, stability samples, and visualization files exist

## Known Gaps Against the Full Paper

The current prototype is not yet a full ROS Noetic reproduction. Missing parts include:

- real sensor or rosbag input
- PCL/OpenCV/Ceres terrain mapping pipeline
- configuration-stability estimator calibrated to a specific robot body
- kinodynamic constraints and closed-loop tracking
- online replanning benchmark on real large-scale rough-terrain data

