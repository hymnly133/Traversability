# MuJoCo 路径可通行性量化实验

这个示例项目用 MuJoCo 做一个“路径可通行性”基线实验：给定静态场景、机器人足迹和若干候选路径，沿路径离散采样，把机器人足迹放到每个采样点，计算它与障碍物的几何距离和碰撞状态，最后输出 0 到 1 的可通行性分数。

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

## 运行

```powershell
.venv\Scripts\python.exe -m traversability.evaluate
```

输出会写到：

- `runs/demo/path_scores.csv`：每条路径的汇总分数
- `runs/demo/path_samples.csv`：每个采样点的 clearance 和碰撞状态
- `runs/demo/traversability_paths.png`：路径、障碍物和碰撞点可视化

## 指标定义

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
assets/traversability_scene.xml   # MuJoCo MJCF 场景
src/traversability/evaluate.py    # 评估脚本
runs/demo/                        # 运行后生成的结果
```

## 后续可以扩展的方向

- 把当前圆柱足迹替换为真实移动机器人底盘模型。
- 增加坡度、台阶高度、地面摩擦系数、能耗或姿态稳定性指标。
- 把 `DEFAULT_PATHS` 换成 A*/RRT/PRM 生成的候选路径。
- 加一个真实控制器，让机器人沿路径运动，再用是否脱轨、控制努力和接触冲击修正可通行性评分。

