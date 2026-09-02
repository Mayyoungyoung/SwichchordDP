# 环境现状记录与复现说明

> 记录时间：2026-09-02。本任务（SWDP 方案验证）遵循约束：优先复用现有 conda 环境，
> 仅当现有环境不可用或依赖冲突时才新建独立环境，不破坏 base 与其它已有环境。

## 硬件

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA GeForce 11GB（Driver 525.85.05, CUDA 12.0） |
| 内存 | 62 GB |
| 磁盘 | 474 GB 可用 |
| 系统 | Ubuntu 20.04 |

## 现有 conda 环境（安装前记录）

`/home/jia/miniconda3`，环境列表与关键包：

| 环境 | Python | 关键包 | 与本任务的关系 |
|---|---|---|---|
| base | - | conda 23.x | 不动 |
| robot_mujoco | 3.10.20 | mujoco 3.3.3（无 torch） | mujoco 3.x 与 metaworld 锁定的 2.x 冲突，**不复用** |
| turbovla-libero | 3.10.20 | torch 2.3.1+cu118, libero 0.1.0, robosuite 1.4.1, mujoco 2.3.2 | **LIBERO 增信实验直接复用，不做修改** |
| lerobot | 3.12.13 | torch 2.7.1+cu118 | 与本任务无关，不动 |
| qwen25vl / vlmrl / ffs / eps0 / grasp / RoboTwin | - | - | 与本任务无关，不动 |

## 新建环境 swdp（Meta-World 概念验证）

**新建理由**：现有环境均无 metaworld / diffusion_policy；robot_mujoco 的 mujoco 3.3.3
与 metaworld 需要的 mujoco 2.x 冲突；turbovla-libero 需为 LIBERO 实验保持完好。
故新建独立环境 `swdp`，不触碰其它环境。

```bash
conda create -n swdp python=3.10 -y
conda activate swdp
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install metaworld gymnasium einops h5py tqdm hydra-core numpy
```

安装完成后导出：

```bash
conda env export -n swdp > environment.yml
pip freeze > requirements.txt
```

## 数据集与模型资产（复用，不重新下载）

| 资产 | 位置 | 用途 |
|---|---|---|
| LIBERO 数据集（153 GB，含 libero_10/90/goal/object/spatial） | /home/jia/VLA/libbero_tmp/datasets | LIBERO 增信实验 |
| libero 包（pip 可编辑安装于 turbovla-libero） | /home/jia/VLA/libbero_tmp | LIBERO 训练/评测 |
| Meta-World 演示数据 | 训练脚本生成或官方数据集 | Meta-World 概念验证 |

## 执行记录(2026-09-02)

- swdp 环境安装: torch 2.3.1+cu118、metaworld 3.1.1(mujoco 3.3.0)、gymnasium 1.3.0、h5py、hydra-core。
- turbovla-libero 环境原样复用(LIBERO 分段提取、技能 DP 训练、离线拼接评测),未做任何修改。
- 数据资产: LIBERO-10 全部 10 任务 hdf5 已完成启发式分段(夹爪动作切换 + 速度谷值),
  结果在 results/libero/data/。
- 训练产物: results/metaworld/models/(pick-place 5 技能 DP、端到端 DP、door-open DP),
  results/libero/models/(LIBERO 统一技能 DP)。均在 .gitignore 中,不提交。
