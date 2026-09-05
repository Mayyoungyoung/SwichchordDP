# 论文方案：Beyond Nominal Success — Policy-Ready Handoffs between Frozen Robot Skills

> 版本 v2.0（2026-09-04，第十轮路线修订：主路线从「事后修复 READY」切换为
> 「事前塑造 Downstream-Compatible Skill Learning」）
> v1.0 的 READY(CEM 修复)已在 Phase 1 五臂 PoC 被双重证伪
> （critic 对抗失效 + 执行器 OOD，见 experiment_report.md §14），
> Correction 模块整体废弃；Detection 成果保留为方法的核心组件。
> 前序依据：experiment_report.md §11-§14。

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

## 3. 方法框架（v2.0 修订：Downstream-Compatible Skill Learning）

### 3.0 路线决策（第十轮，基于 §14 负结果）

零策略适应的修复（CEM/重执行/预算扩展）被系统性证伪——修复需要闭环执行
能力，冻结 DP 库不具备。主路线切换为**训练时塑造上游终态分布**：

> **一个技能的「成功终态」并不唯一；不同成功终态对后续技能的可执行性不同。
> 能否利用下游技能的可行性，反向塑造上游技能的终态分布？**

优点：不需 recovery policy / CEM / runtime planner；与 DP 天然兼容；
直接利用已有数据；避开 §14 的双重失效根因（分布内加权而非外推搜索）。

### 3.1 方法：Trajectory Reweighting（非 ∇F_B 引导）

$$
\mathcal L = \mathcal L_{DP} + \lambda\,\mathcal L_{DC},
\qquad
w(s^+) = 1 + \lambda\, F_B(s^+)
$$

- 对上游技能轨迹按**终态下游可行性**加权（trajectory-level → step-level 广播）；
- 三个版本（实验设计）：
  1. **Outcome-weighted**（上界）：w = 1+λy（y=下游真实 rollout outcome）；
  2. **Feasibility-guided**（正式方法）：w = 1+λF_B(s⁺)（连续可行性）；
  3. **Quality-weighted**（对照基线）：w = 1+λ·grasp 自身质量（对中+闭合度）——
     证明「下游可行性」优于「当前技能质量」；
- **关键设计**：用 distribution 内 reweighting 而非 ∇_s F_B 或 CEM argmax——
  直接规避 §14 的 critic 对抗失效。

### 3.2 与 Chord 的统一

| 层 | 不兼容类型 | 算子 |
|---|---|---|
| Action Compatibility | 切换动作不连续 | Chord（免训传输场，已有） |
| Terminal-State Compatibility | 终态不被后继可执行 | F_B 判别 + 上游终态分布塑造（新） |

### 3.3 [废弃] READY 事后修复模块

CEM-on-V repair / release+re-execute / budget extension 全部废弃（§14 实证）。

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

## 5. 实验计划（v2.0 修订）

**主实验：Downstream-Compatible Skill Learning（grasp→lift，第一轮最小验证）**

1. 数据（dc_collect.py）：脚本 reach + DP grasp 自然轨迹 240 条，每条带
   lift×10 outcome 标签 y_i + grasp 质量标签 + 完整动作轨迹；
2. 训练（dc_train.py）：从 dp_pick-place-v3.pt 微调，权重方案
   {uniform(等权微调对照), outcome(y 加权, λ∈{1,2,4}), quality(grasp 质量
   加权), fb(F_B 连续加权，第二轮)}；
3. 评估（dc_eval.py）：每模型 120 次 rollout——P(grasp) / P(lift|grasp) /
   P(e2e) / 终态 F_B 分布 / grip 分布（Pareto 图 + 分布对比图）。

**Go/No-Go**（用户建议的判定）：若 outcome 加权使 P(e2e) 显著上升且
P(grasp) 无明显下降 → 升级 F_B 连续加权 + 泛化到第二对；否则此路价值
有限，回到「现象+检测」的 empirical study。

> **第十轮执行结果（2026-09-04，experiment_report.md §15）：NO-GO。**
> 240 轨迹收集（y pos rate=0.970）→ 5 模型微调 → 120 回合×6 臂评估：
> base 0.9525 / uniform 0.9558 / outcome λ=1,2 0.9600 / **λ=4 0.9683** /
> quality 0.9600——方向性正且随 λ 单调，但 Δe2e=+0.016 未达显著
> （McNemar 3-1 p=1.0，SE≈0.019）；P(grasp) 恒 1.0 无下降；
> 所有微调变体的 grip 终态分布显著偏移（KS p<0.005，0.433→0.436）。
> 根因：脚本 reach setup 下劣质抓取天然发生率仅 ~3%，加权信号质量小
> （§13.5 的 8.3% 在 DP 链 setup 宽散布下）。待决策：
> (a) DP 链 setup 重收数据再测（失败样本 2-3×）；(b) 按判据回到
> empirical study。fb 加权受 outcome 上界约束，暂不投入。

> **v2.1 路线修订（2026-09-05，第十一/十二轮，experiment_report.md §16-§17）：**
> 放弃「训练时塑终态」，改为主张 **Boundary Monitor + Best-of-K**（免训、
> 可插拔、不碰技能权重）。两轮证据链：
> (1) 第十一轮预言（§16）：三层交接触发机制（V 峰值切换/成功即停/
> 稳定夹持即停）被离线沙盘双重证伪——劣质终态在接触早期已定型；
> 传导检验定位失败源在**入口**（grasp 是条件放大器：入口 hp 1.6×→
> 终态 6.2×；三对边界全部复现入口传导）；
> (2) 第十二轮闭环（§17）：r2g（入口质量型缺口）3 个独立种子段稳定
> 显著（e2e +5~6pp，合并 p=0.0001，上游无损，θ/K 鲁棒）；l2c/c2p
> （任务难度型/下游瓶颈型）无效且根因干净归因。
> 部署判据：离线传导检验（入口特征比值/AUC）预筛边界类型。
> 论文形态：条件性发现（现象→机制→组件）的完整证据链。下一步：
> LIBERO 图像泛化或收口撰写。

**原 READY 五阶段计划（v1.0 §5）**：Phase 0/1 已执行（§14），
Phase 2-4 废弃（修复路线已关）。

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
