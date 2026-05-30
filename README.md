# Traversability 复现实验

本仓库包含两个相关实验：

1. MuJoCo 几何可通行性量化 demo：沿候选路径采样，计算机器人足迹与障碍物的 clearance 和碰撞比例。
2. 论文复刻原型：面向 `Real-Time Multilevel Terrain-Aware Path Planning for Ground Mobile Robots in Large-Scale Rough Terrains` 的多层地形感知路径规划 demo。

## 论文复刻说明

目标论文：

```text
Li, Yuxiang et al.
Real-Time Multilevel Terrain-Aware Path Planning for Ground Mobile Robots in Large-Scale Rough Terrains.
IEEE Transactions on Robotics, 2025, 41:4159-4179.
DOI: 10.1109/TRO.2025.3577015
```

我检查了作者公开仓库 `HITSZ-NRSL/terrain-aware-planning`，截至当前只有 README 和 citation，没有可运行源码。因此本项目没有复制官方实现，而是搭建一个工程化复刻原型，覆盖论文题目和关键词对应的核心链路：

- 大尺度崎岖地形高程图
- 论文 IV 对应的 NDT/implicit voxel map 原型：按体素维护点数、均值、协方差，并用邻域 Gaussian 融合做 SVD 地形分析
- 连续坐标 implicit terrain map 查询接口
- 多层/多分辨率 terrain pyramid
- 坡度、粗糙度、sparsity、局部台阶和障碍物 margin 代价
- checkpoint/ray-casting 风格的 terrain、collision、falling traversal risk 验证
- 论文 IV-D 对应的 3D voxel A* 全局规划原型，代价为路径长度与 traversal cost 的组合
- union-find connected traversable voxel set，用于把搜索限制在可通行连通体内
- 粗到细 A* 路径规划
- 矩形机器人 footprint 的 configuration-stability estimation
- 局部迭代几何平滑和风险验证
- 局部轨迹优化，可在路径质量模式启用，在实时 benchmark 中关闭以保持低延迟
- 滚动窗口实时重规划 benchmark，包含动态障碍注入
- CSV 与 PNG 可视化输出

## 环境

当前工作区已创建 `.venv`，并安装了：

- `mujoco==3.9.0`
- `numpy`
- `matplotlib`
- `rich`

如果需要重新安装：

```powershell
uv venv --python 3.10
uv pip install -e .
```

这里建议使用 Python 3.10 到 3.12。你机器上的默认 Python 是 3.14，很多机器人仿真依赖的 wheel 还不一定完整覆盖这个版本。

## 运行论文复刻 demo

```powershell
.venv\Scripts\python.exe -m traversability.multilevel_demo
```

输出会写到：

- `runs/multilevel/summary.csv`
- `runs/multilevel/planned_path.csv`
- `runs/multilevel/stability_samples.csv`
- `runs/multilevel/multilevel_plan.png`

验证复刻 demo：

```powershell
.venv\Scripts\python.exe -m traversability.verify_reproduction
```

验证项包括：规划成功、最大风险不超过阈值、最低稳定性不低于阈值、路径长度合理、结果文件和图片生成成功。

单独验证 implicit terrain map 查询：

```powershell
.venv\Scripts\python.exe -m traversability.verify_implicit_map
```

单独验证论文式 NDT implicit voxel map、SVD 地形指标和 traversal risk：

```powershell
.venv\Scripts\python.exe -m traversability.verify_ndt_map
```

单独验证论文式 NDT traversability map 上的 3D voxel A* 全局规划：

```powershell
.venv\Scripts\python.exe -m traversability.verify_ndt_planner
```

运行完整验证，包括滚动重规划、ablation、路径跟踪、点云输入和多场景 benchmark：

```powershell
.venv\Scripts\python.exe -m traversability.verify_reproduction --check-realtime --check-ablation --check-tracking --check-pointcloud --check-benchmark
```

单独运行实时重规划：

```powershell
.venv\Scripts\python.exe -m traversability.realtime_demo
```

单独运行 multi-level vs single-level 对比实验：

```powershell
.venv\Scripts\python.exe -m traversability.ablation_demo
```

单独运行闭环路径跟踪仿真：

```powershell
.venv\Scripts\python.exe -m traversability.tracking_demo
```

单独运行点云到高程图再规划：

```powershell
.venv\Scripts\python.exe -m traversability.pointcloud_demo
```

使用外部点云文件，支持 `.csv`、`.xyz`、`.txt`、`.npy`、ASCII `.ply`：

```powershell
.venv\Scripts\python.exe -m traversability.pointcloud_demo --input path\to\cloud.xyz
```

导出 demo 生成的点云再从文件规划：

```powershell
.venv\Scripts\python.exe -m traversability.pointcloud_demo --export-points runs\pointcloud\terrain.csv
.venv\Scripts\python.exe -m traversability.pointcloud_demo --input runs\pointcloud\terrain.csv
```

单独运行多场景 benchmark：

```powershell
.venv\Scripts\python.exe -m traversability.benchmark_suite
```

实时 benchmark 输出：

- `runs/realtime/replanning_summary.csv`
- `runs/realtime/executed_trajectory.csv`
- `runs/realtime/replanning.png`

Ablation 输出：

- `runs/ablation/ablation_summary.csv`
- `runs/ablation/ablation.png`

Tracking 输出：

- `runs/tracking/tracking_summary.csv`
- `runs/tracking/tracking_states.csv`
- `runs/tracking/tracking.png`

Point-cloud 输出：

- `runs/pointcloud/pointcloud_summary.csv`
- `runs/pointcloud/pointcloud_plan.png`

Benchmark 输出：

- `runs/benchmark/benchmark_scenarios.csv`
- `runs/benchmark/benchmark_summary.csv`

## 运行 MuJoCo clearance demo

```powershell
.venv\Scripts\python.exe -m traversability.evaluate
```

输出会写到：

- `runs/demo/path_scores.csv`：每条路径的汇总分数
- `runs/demo/path_samples.csv`：每个采样点的 clearance 和碰撞状态
- `runs/demo/traversability_paths.png`：路径、障碍物和碰撞点可视化

## MuJoCo clearance 指标定义

每条路径会计算：

- `min_clearance_m`：整条路径上机器人足迹到障碍物的最小几何距离，小于等于 0 表示碰撞或穿透。
- `p10_clearance_m`：10 分位 clearance，用来衡量路径瓶颈，而不是只看平均值。
- `collision_rate`：发生碰撞的采样点比例。
- `length_efficiency`：起终点直线距离 / 路径实际长度，绕行越多越低。
- `score`：综合分数，范围 `[0, 1]`。

当前综合分数为：

```text
score = passable_fraction * (
  0.55 * mean(normalized_clearance)
  + 0.25 * p10(normalized_clearance)
  + 0.20 * length_efficiency
)
```

其中 `normalized_clearance = clip(clearance / desired_clearance, 0, 1)`，默认期望安全间隙是 `0.35 m`。

## 项目结构

```text
assets/traversability_scene.xml          # MuJoCo MJCF 场景
src/traversability/evaluate.py           # MuJoCo clearance 评估
src/traversability/terrain.py            # 地形生成、地形分析、多层地图
src/traversability/ndt_map.py            # 论文式 NDT implicit voxel map 与 traversal risk
src/traversability/ndt_planner.py        # 论文式 3D voxel A* 与连通可通行体素集
src/traversability/implicit_map.py       # 连续地形查询/隐式地图 facade
src/traversability/planner.py            # 多层 terrain-aware A*
src/traversability/trajectory_optimizer.py # 局部轨迹优化
src/traversability/stability.py          # 机器人 footprint 配置稳定性估计
src/traversability/multilevel_demo.py    # 论文复刻原型入口
src/traversability/realtime_demo.py      # 滚动窗口重规划 benchmark
src/traversability/ablation_demo.py      # 多层规划 vs 单层规划对比实验
src/traversability/tracking.py           # pure-pursuit 路径跟踪仿真
src/traversability/tracking_demo.py      # 闭环跟踪验证入口
src/traversability/pointcloud.py         # 点云到 elevation grid 转换
src/traversability/pointcloud_demo.py    # 点云输入规划验证入口
src/traversability/verify_pointcloud_io.py # 点云文件 IO 验证
src/traversability/benchmark_suite.py    # 多场景 benchmark 聚合
src/traversability/verify_reproduction.py # 自动验证入口
src/traversability/verify_implicit_map.py # implicit map 查询验证
src/traversability/verify_ndt_map.py     # NDT voxel map 和论文 IV 指标验证
src/traversability/verify_ndt_planner.py # NDT voxel global planning 验证
runs/                                  # 运行后生成的结果
```

## 后续可以扩展的方向

- 把当前圆柱足迹替换为真实移动机器人底盘模型。
- 增加坡度、台阶高度、地面摩擦系数、能耗或姿态稳定性指标。
- 把 `DEFAULT_PATHS` 换成 A*/RRT/PRM 生成的候选路径。
- 加一个真实控制器，让机器人沿路径运动，再用是否脱轨、控制努力和接触冲击修正可通行性评分。
