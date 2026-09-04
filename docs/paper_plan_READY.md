# 论文方案：Beyond Nominal Success — Policy-Ready Handoffs between Frozen Robot Skills

> 版本 v1.0（2026-09-04，第九轮方案定稿）
> 状态：方案收敛，Phase 0/1 实验启动
> 前序依据：experiment_report.md §11-§13（分水岭诊断 → 六臂消融 → Terminal-State
> Diversity 三对泛化 → grasp→lift 劣质抓取亚域发现）

---

## 1. 核心科学命题

> **A task-complete state is not necessarily a policy-ready state for the next frozen skill.**

形式化：对冻结技能 π_j 定义 successor critic

$$V_j(s) = P(\mathrm{Succ}(\pi_j) \mid s)$$

交接缺口（handoff gap）：

$$s \in \mathcal{S}_i^{\mathrm{succ}} \;\not\Rightarrow\; V_j(s) \ge \tau$$

**实证（已有数据，§13.5）**：grasp→lift 中 8.3% 的语义合法终态（过度闭合 grip
0.29 vs 正常 0.44 + 偏心 0.028 vs 0.005，特征空间线性可分）使 lift 成功率
塌到 0.0；同时位置型对（carry→place 0.972 / reach→grasp 0.990）无缺口——
现象**条件性存在**，判据是成功谓词的粒度（质量型 vs 位置型）。

**核心问题**：*How can frozen skills be composed when nominal skill success
does not guarantee downstream executability?*

## 2. 定位（准确表述，避免过度声明）

- **不说「空白」**：GSC/CDGS/RoboHarness 已在推理时处理技能组合/交接；
  本工作的创新是**执行后、局部、显式优化后继策略成功率的交接修复**。
- **不说「免训练」**：准确说法是 **frozen policies, no policy adaptation,
  lightweight offline successor critic**。
- **「policy-ready」而非「feasible」**：可行性必须相对后继策略定义
  （BOSS 的观测失配反例证明「物理可行 ≠ 策略可执行」）。

### 2.1 三类交接失败分型（intro 的问题分类学）

| 类型 | 例子 | 正确响应 | 本文范围 |
|---|---|---|---|
| 物理状态失配 | 抓取成功但姿态/夹爪开度使 lift 失败 | 受约束局部动作修复 | **主实验** |
| 观测失配 | 物理可移但后继视觉 DP OOD | 观测接口适配/重规划 | limitations |
| 语义/计划失配 | 修复会撤销前序目标 | 不修复，高层 replan | 三路决策之一 |

## 3. 方法框架：READY（三阶段闭环）

**方法名候选**：READY（**Re**pair via successor critics for policy-re**ADY**
handoffs）主投 AAAI/ICLR；SGR / HandoffGuard 备选。

```
Agent: [grasp] → [lift] → [carry] → [place]        (高层只选序列)
              │
              ▼ 每条边上：
     Handoff Manager (i → j)
     1. Evaluate LCB_δ(s_i, j)          ← Diagnosis
     2. Ready      → execute π_j
     3. Repairable → CEM repair (RH 执行) ← Correction
     4. 不可修复  → escalate 失败签名给高层重规划
```

### 3.1 Diagnosis：Successor Critic + LCB 门控

- \(V_j(s)\)：19 维几何特征 + 技能 one-hot 小 MLP（已有 success_model.py 基建）；
- **训练数据来源（关键卖点）—— composability probing**：冻结技能互探
  （π_i 自然执行产生终态 → π_j rollout × K 取 outcome），自监督零标注，
  每对 ~240 回合 ≈ 30 分钟仿真；
- ensemble M=5 折 → \(\bar{V}_j, \sigma_j\)，门控用
  \(\mathrm{LCB}_\delta(s) = \bar{V}_j(s) - \kappa\sigma_j(s)\)；
- **训练分布教训（两次验证，§12-§13）**：必须用目标分布自然数据训练
  （诊断扰动数据不覆盖抓取质量维度）。

### 3.2 Correction：不变量约束的动作序列搜索（统一 formulation）

$$a^*_{1:H} = \arg\min_{a_{1:H}} C(a_{1:H}; s) \quad \text{s.t.}\quad
\mathrm{LCB}_\delta(\hat{s}_H) \ge \tau,\;\; \mathcal{I}(\hat{s}_H)=\mathcal{I}(s),\;\; \hat{s}_t \in \mathcal{S}_{safe}$$

- 搜索空间恒为 **short-horizon action sequence**（H=10），跨 failure mode
  不换 formulation——regrasp 只是 \(a^*\) 恰好表现为重抓的实例；
- 求解：CEM（population 64 × 5 代）在 MuJoCo roll-forward 上；
- 任务不变量 \(\mathcal{I}\)：物体在手中、高度、关节限位（physics-valid 已有实现）；
- 执行：receding-horizon（每次执行 1-2 步重观测 re-gate），与 DP chunked
  执行哲学一致；PoC 先全序列，RH 做消融。

### 3.3 与 Chord 的统一（理论主线）

两个模块都是**冻结 DP 上的 test-time operator，作用空间不同**：

| | Chord（已有） | Repair（新增） |
|---|---|---|
| 作用空间 | 动作空间 | 状态空间 |
| 利用的结构 | score 网络（技能条件间 eps-残差场，能量 3×↓） | outcome landscape（successor critic） |
| 算子性质 | 免训解析算子（已有定理 1-3） | 搜索算子（约束保持） |
| 回答 | how to hand off | whether & where to hand off |

统一表述：**Test-time composition = action-space transition operator ⊕
state-space diagnosis/repair operator**，二者只读取冻结生成模型的结构，
零 policy adaptation。（NeurIPS 版可深化为统一能量视角
\(E_{total} = E_{chord} + \lambda E_{feasibility}\)。）

训练时方向（backward fine-tuning）不进主论文——与 Sequential Dexterity
正面撞车且违背 frozen 设定；discussion 一句话正交性声明。

## 4. 相关工作定位（检索核实后的对照）

| 工作 | 差异化 |
|---|---|
| Sequential Dexterity (CoRL'23) | 它训练时双向微调对齐；我们部署时诊断+修复，技能永久冻结 |
| GSC (CoRL'23) | 它执行前验证计划级 pre/post-condition；**计划级验证看不到执行后真实终态**（post-condition 满足恰是我们的反例）——结构性盲区 |
| Back to the Manifold (2022) | 它回到**密度流形**；我们回到**后继成功域**。论据 A：劣质态密度偏移极小（0.15 单维）但成功率 1.0→0.0，密度信号不必要也不充分 |
| CDGS (ICLR'26 Oral) | 它在计划空间搜索可行性；我们在状态空间搜索修复动作，目标是后继策略成功率（非数据似然） |
| BOSS (RA-L'26) | 观测失配 benchmark——证明「可行必须相对后继策略定义」，是 policy-ready 措辞的佐证 |
| FAR (2026) / BGR | FAR 测试时更新策略；我们不更新，只优化交接状态 |
| RoboHarness (2026) | 编排层；我们是每条边上的交接组件，可插拔 |
| PoCo / score composition 系 | 组合对象是生成组件；我们把 learned outcome landscape 当组合对象 |

## 5. 实验计划（五阶段，风险递进）

**Phase 0（扩样）**：grasp→lift collect 480 回合（预期 ~40 失败态）；
lift→carry termdiv 探测第二失配对。训练 5-member ensemble \(V_{lift}\)。

**Phase 1（PoC，决定性实验）**：失败态集上五臂对照——

| 臂 | 说明 | 检验 |
|---|---|---|
| direct handoff | 直接 lift | 基线锚点（≈0） |
| random action chunk | 同预算随机动作 | 排除「任何扰动都有用」 |
| re-execute grasp | 重执行前序技能 | **关键基线**：修复收益是否只是重试？ |
| READY repair | CEM on \(V_{lift}\) + 不变量约束 | 方法主体 |
| oracle regrasp | 脚本完美抓取 | 上界 |

指标：lift 成功率 ×10、不变量保持率、动作代价、墙钟、修复后 LCB。
**核心读出：READY vs re-execute-grasp 的差距** = 显式优化后继就绪超越
盲目重试的增量。

**Phase 2（诊断质量）**：AUROC/AUPRC/Brier/ECE/selective risk-coverage；
No-Go 两对成为特异度证据（FP≈0 → 选择性计算）。

**Phase 3（泛化）**：优化器/代价/阈值冻结，仅换 \(V_j\) one-hot 与状态来源；
外加 BTM 式密度基线（验证论据 A）。

**Phase 4（端到端）**：5 链 120 回合配对四臂：naive / chord / chord+gate /
chord+gate+repair(+escalate)。预期挽回 8-10 分 e2e。

**Phase 5（可选）**：密度对比全表、RH vs 全序列、LCB κ 敏感性。

## 6. 审稿人攻击点与应对（预案）

1. 「就是 success detector + retry」→ re-execute-grasp 正式基线（Phase 1）；
2. 「劣质态=OOD，密度检测即可」→ 论据 A + 密度基线实测（Phase 3）；
3. 「现象只有一对 n=1」→ 条件性发现 + 判据（质量型 vs 位置型谓词）+ 第二对/场景扩展；
4. 「只有 Meta-World 状态空间仿真」→ LIBERO 图像基建已有，critic 可吃后继策略观测输入；
5. 「critic 每对要 240 回合」→ 自监督零标注 30 分钟/对，跨执行摊销；
6. 「CEM 用仿真器」→ RH + 学到的短时动力学（FAR 同假设），scope 声明；
7. 「修复撤销已完成任务」→ 不变量约束 + 保持率实测；
8. 「为什么不微调」→ 不可变技能资产三类场景 + 正交性声明；
9. 「三模块拼贴」→ 3.3 统一叙事 + 逐模块消融；
10. 「GSC/CDGS 已做推理组合」→ 计划级验证的结构性盲区（论据 B）。

## 7. 投稿定位

- **AAAI**（首选）：三阶段闭环 + 能力矩阵 + 条件性现象，方法轻量系统完整；
- **ICLR/NeurIPS**：需补理论（统一能量视角 / CEM-on-landscape 收敛性）；
- **CoRL/RA-L**：加实机 RH 执行则为系统论文；
- 时间线：Phase 0/1（本周）→ Phase 2-4（2 周）→ 理论+写作（4 周）。

## 8. 复现索引（现有资产）

- 失败态数据：`results/metaworld/eval/termdiv_grasp_lift_states.npz`（19 态）
- Critic 基建：`code/swdp/success_model.py`（19 维特征 + MLP + 5 折 CV）
- 探测协议：`code/metaworld/diag_termdiv.py`（collect/rollout/restore 全套）
- Chord 基建：`code/swdp/{chord_compose,harness}.py` + 定理（theory_composability.md）
- 主表回归：`regress_chord/naive_mask_proj.json`（0.562/0.500，48eps bit 级）
