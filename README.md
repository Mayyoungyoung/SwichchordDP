# SWDP: 机器人扩散策略的推理期技能组合（免训）

> SwitchChord DP —— 把 ChordEdit 的「Chord 控制场」从图像潜在空间平移到动作块空间，
> 实现技能 **切换 / 串联 / 叠加** 三种免训组合方式。

## 仓库结构

```
├── SWDP.md                  # 方案设计文档（Idea 9 技术方案）
├── docs/
│   ├── survey.md            # 相关论文/开源工作调研报告
│   ├── experiment_report.md # 实验记录、指标与可行性结论
│   └── environment.md       # 环境现状记录与复现说明
├── code/
│   ├── swdp/                # 核心库：技能条件扩散策略、ChordCompose、蒸馏、可行性投影
│   ├── metaworld/           # Meta-World 概念验证实验（主基准）
│   └── libero/              # LIBERO 增信实验
├── results/                 # 实验指标与日志（不提交大文件）
├── environment.yml          # swdp 环境导出
└── requirements.txt         # pip 依赖清单
```

## 快速开始

```bash
# 1. 环境（已装好则跳过）
conda env create -f environment.yml
conda activate swdp

# 2. Meta-World 概念验证
bash code/metaworld/scripts/run_all.sh

# 3. LIBERO 增信实验（使用 turbovla-libero 环境）
bash code/libero/scripts/run_all.sh
```

## 核心思路（一分钟版）

把 ChordEdit（CVPR 2026 Oral）的「Chord 控制场」从图像潜在空间平移到动作块空间：

- **Switch（切换）**：执行中从技能 s 换成 s′，用低能 Chord 场 `û = [δ·R(t−δ) + τ·R(t)] / (δ+τ)` 单步传输，替代高能发散的朴素硬切
- **Chain（串联）**：长程任务 = 技能序列，每次交接 = 技能动作分布间的最小能量耦合（Benamou–Brenier 最优传输视角）
- **Combine（叠加）**：多技能乘积专家叠加后做 Chord 平滑

相对 ChordEdit 的新增：**物理可行性投影**（运动学/动力学约束）与**时间掩码**（轨迹段保持），
并新增「零样本组合稳定性条件」定理回答「哪两个技能可以免训组合、误差多大」。

## 引用参考

- ChordEdit: One-Step Low-Energy Transport for Image Editing. CVPR 2026 Oral. arXiv:2602.19083
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023 / IJRR 2025
- Learning Diffusion Policy from Primitive Skills for Robot Manipulation. arXiv:2601.01948
