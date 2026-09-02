# 相关论文/开源工作调研报告：免训技能组合扩散策略（Idea 9）

> 调研时间：2026-09-02
> 调研对象：与「技能切换/衔接/组合」相关的论文与开源工作，覆盖扩散策略（Diffusion Policy）、
> 少步/一步蒸馏扩散模型、技能原语库、zero-shot 技能组合、技能交接/时序拼接、动作空间平滑与
> 最优传输（OT/Benamou–Brenier）等方向。
> 调研目标：逐条对照 /home/jia/DP/SWDP.md 的核心设计，找出（a）可引用来强化理论的部分、
> （b）可填补的对比实验、（c）可能撞车的相关工作，并明确相对 ChordEdit 的 novelty 增量。

---

## 0. 结论速览

1. **方法基础（ChordEdit）完全可靠**：ChordEdit 为 CVPR 2026 Oral（最佳学生论文荣誉提名），
   arXiv:2602.19083，代码已开源。其理论骨架（能量收缩、截断误差、Benamou–Brenier 间隙、
   单步稳定性条件）确实只依赖 L²/L^∞ 收缩、Lipschitz 与 Jensen 不等式，**可以逐条迁移到动作空间**。
2. **SWDP.md 存在两处需要修正的公式/符号问题**：
   - Chord 场公式权重写反：原文是 `û = [t·R(t−δ) + δ·R(t)]/(t+δ)`（更早时刻 t−δ 的测量权重为 t），
     SWDP.md 写成了 `[δ·R(t−δ) + τ·R(t)]/(δ+τ)`；
   - 「τ」在 ChordEdit 原文中并不存在，原文用 `λ` 表示步长缩放（`x_pred = x_in + λ·û`），
     SWDP.md 把 τ 同时用作平滑权重与步长，语义混淆。
3. **「技能串联/组合」不是空白领域，存在多个近距离工作**：Generative Skill Chaining（CoRL'23）、
   SCaR（NeurIPS'24）、SkillDiffuser（ICLR'24）、SDP（arXiv 2601.01948）、BOSS、DeCo、
   CCDP（IROS'25）等。本文必须与其明确区分——**区分点正是 ChordEdit 带来的增量：单步低能传输、
   免训、稳定性判据、物理可行性投影**。
4. **一个关键情报**：2026-08 的复现分析论文（Rethinking One-Step Image Editing through ChordEdit）
   发现 Chord 场在默认参数下 ≈ (6/7)·Δv(x,0.75) + (1/7)·Δv(x,0.9)，**主要由 t−δ 时刻主导**，
   并指出「核心增益来自选择合适有效时间步」。这既是机会（我们在动作域同样可以验证"有效时间步"解释），
   也是审稿风险（必须正面回应"Chord 只是时间步平移"的质疑）。
5. **投稿窗口**：当前 2026-09，AAAI 2027 摘要截止已过（约 2026-07/08），**现实目标是 CVPR 2027**
   （摘要通常 11 月初截止）。调研按 CVPR 2027 节奏规划。

---

## 1. 方法基石精读：ChordEdit

> Liangsi Lu, Xuhang Chen, Minzhe Guo, Shichu Li, Jingchao Wang, Yang Shi.
> *ChordEdit: One-Step Low-Energy Transport for Image Editing.* CVPR 2026 (Oral, Best Student Paper Honorable Mention). arXiv:2602.19083. 代码: github.com/ChordEdit/ChordEdit

### 1.1 核心机制（已核实原文）

- **问题形式化**（§3.1）：T2I 模型诱导条件概率流 `dx_t/dt = v(x_t, t, c)`；
  编辑 = 把源分布 `p1(x|c_src)` 传输到目标分布 `p0(x|c_tar)`。
- **可观测模型**（§3.2）：锚点 `x_τ := x_src`（干净源图），用正向加噪核 `K_t(·|x_τ)` 造带噪代理
  `z ~ K_t`，查询模型输出 Q，残差 `ΔQ(z,t) = Q(z,t,c_tar) − Q(z,t,c_src)`，
  代理场 `R(x_τ,t) = E_{z~K_t}[ B_t ΔQ(z,t) ]`（共享噪声 Monte Carlo）。
- **B_t 闭式系数**（附录 C，VP 调度下）：
  - 噪声预测模型：`A_t^(ε) = −α̇(t)/(α(t)σ(t)) = β(t)/(2σ(t))`
  - x0 预测模型：`A_t^(x0) = α̇(t)/σ(t)²`
  - v 预测模型：`A_t^(v) = −α̇(t)/σ(t)`
  - 得分模型：`A_t^(score) = β(t)`；速度/流匹配模型：`B_t ≡ I`
  - 数值实现用一阶差分近似 α̇, σ̇；查询时间远离 t=1 以保证良态。
- **Chord 控制场**（§4.2，Eq. 4.5，核心公式）：
  ```
  û_t(x_τ) = [ t·R(x_τ, t−δ) + δ·R(x_τ, t) ] / (t+δ)
  ```
  等价于对朴素场 R 做因果单边核平滑 `û = K_δ ∗ R`（K_δ ≥ 0, ∫K_δ = 1, supp ⊂ [0,δ]）。
  **注意权重方向**：更早时刻 (t−δ) 的测量权重为 t（大权重），当前时刻 t 的测量权重为 δ。
  默认 t=0.90、δ=0.15 时 û ≈ (6/7)·R(0.75) + (1/7)·R(0.90)。
- **单步传输**（Algorithm 1）：`x_pred = x_in + λ·û`（λ 为步长缩放，默认 λ=1.0 附近调优）；
  可选近端精炼 `x_tar = prox(x_pred, t_c, c_tar) = B_{t_c} Q(x_pred, t_c, c_tar)`（一次目标条件前向，默认 t_c=0.30）。

### 1.2 理论骨架（附录 D/E，逐条核实）

| 结论 | 出处 | 内容 | 迁移到动作空间的可行性 |
|---|---|---|---|
| L² 能量收缩 ‖û‖ ≤ ‖R‖ | Prop D.1 / Cor D.3 | Jensen 不等式，K_δ 为单位质量核 | 纯泛函分析，**直接成立** |
| L^∞ 收缩（场、时间导数、空间梯度） | Prop D.5 | 卷积与求导可交换 | 直接成立 |
| 局部截断误差更小 | Lemma D.4 + Prop D.5 | 一致性常数 C_cho ≤ C_nai | 直接成立 |
| 全局 O(h) 收敛、误差常数更小 | Thm D.6 | 离散 Grönwall，h=1 单步可行 | 直接成立 |
| 与 Benamou–Brenier 最优能量间隙 O(δ) | Thm E.4 | 弦空间投影近似理论（Jackson 型） | 直接成立 |
| 单步误差 + 稳定性条件 | Thm E.6 | C_cho ≤ C_nai，稳定性条件 h·L < 1 与场无关 | 直接成立 |
| Euler 稳定性不受控制设计影响 | Prop E.7 / Cor E.9 | 误差比 ≤ 1，δ=0 或 Ṙ≡0 时取等 | 直接成立 |

**结论**：SWDP.md 第 5 节「直接迁移」的判断正确。需要注意的两点：
- ChordEdit 的稳定性条件是 **Euler 显式格式的通用条件 h·L < 1**，而非「哪两个技能可以免训组合」
  的判据——SWDP.md 的「新增定理 1（零样本组合稳定性条件）」需要比 Thm E.6 更进一步：
  必须把 L（`sup‖∇_a v‖`）显式表达为**技能对 (s,s′) 的函数**，并给出「技能对可组合」的充分条件。
  这是本工作相对 ChordEdit 真正的理论增量，但也需要谨慎论证（见 §4.3 的风险）。
- ChordEdit 的「近端精炼」作用在图像域是**语义增强**；在动作域没有对应物，必须替换为
  **物理可行性投影**——SWDP.md 第 6.3 节的判断正确，这是论文独立性来源之一。

### 1.3 复现分析论文的重要情报

> *Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights.* arXiv 2026-08。

- **发现 1**：Chord 场更新主要由 t−δ 时刻的速度差主导（默认参数下权重 6/7 vs 1/7）；
  用「直接使用主导时间步（t=0.75, δ=0）」简化 ChordEdit 可获得近似性能。
  → **启示**：本工作应把「有效时间步平移」也作为消融项（δ 的非零作用 vs 纯时间步平移），
  并论证在动作域 Chord 平滑相比纯平移仍有独立价值（例如多时刻测量降低方差）。
- **发现 2**：复现中 Chord 场主要改善保真度（preservation），对语义对齐的增益弱于原文报告。
  → **启示**：在动作域预期对称：Chord 场主要改善「轨迹平滑/交接保真」，
  「目标技能达成」可能仍需可行性投影/后续正常推理完成——实验指标应分开报告。
- **发现 3**：提出「提示条件动态时间步选择」为未来方向。
  → **机会**：动作域可做「技能对条件的时间步选择」（根据技能对差异自动选 t、t_c），
    可作为本文的 secondary contribution 或 future work 呼应。

---

## 2. 文献分组调研

### 2.1 组 A：扩散策略基础（引用支撑）

| # | 文献 | 与 Idea 9 的关系 |
|---|---|---|
| A1 | Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023 / IJRR 2025 | 方法基石。其「动作块去噪生成 + receding horizon」即本文的基础策略范式；论文 §3 的 `da/dt = v(a,t,o,s)` 设定需对齐其实现。 |
| A2 | Liu et al., *Compositional Visual Generation with Composable Diffusion Models.* ECCV 2022 | 「乘积专家 + 分数/能量叠加」组合范式的源头。本文 Combine 算子的理论依据；GSC/CCDP 均基于此。可引用于 §4.3 Combine 形式化。 |
| A3 | Ajay et al., *Compositional Foundation Models for Hierarchical Planning*（DiP, NeurIPS 2023，可选） | 能量组合 + 扩散规划。Combine 的规划域对照。 |
| A4 | Ma et al., *Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation.* CVPR 2024 | 「可行性」进入扩散策略的先例：可微运动学对齐。本文可行性投影可引用其作为「动作域需要物理约束」的佐证，且需说明与本文的差异（他们是训练期/结构内嵌，本文是推理期免训投影）。 |

### 2.2 组 B：技能串联/衔接（最近距离工作，撞车风险最高，必须逐篇区分）

| # | 文献 | 核心方法 | 与 Idea 9 的关系 |
|---|---|---|---|
| B1 | Mishra et al., *Generative Skill Chaining (GSC): Long-Horizon Skill Planning with Diffusion Models.* CoRL 2023 | 为每个技能训练独立扩散模型；推理期按任务图用**能量组合（EBIL/乘积专家）+ 技能分布重叠**做拼接；在仿真 kitchen 等域验证。 | **最直接撞车对象**。区分点：(i) GSC 需要**每技能单独训练**扩散模型，本文用**冻结的共享技能条件策略**（免训）；(ii) GSC 依赖技能分布间的自然重叠/碰撞缓冲来做拼接，本文把拼接定义为**Chord 单步低能传输**（有理论保证）；(iii) GSC 无稳定性判据、无物理可行性投影、无少步蒸馏语境。对比实验：可在相同技能库上复现其能量组合拼接作为 baseline。 |
| B2 | Chen et al., *SCaR: Refining Skill Chaining for Long-Horizon Robotic Manipulation via Dual Regularization.* NeurIPS 2024 | **训练式**：对技能策略做「终止-初始状态匹配正则 + 光滑正则」双正则化，让交接平滑。 | 「训练式组合」的 SOTA 代表。本文与之的关系 = 「免训推理期组合 vs 训练期正则化」的对照。**必做 baseline/对比**：训练成本对比 + 交接平滑度指标对比。审稿人一定会问「为什么不训练」——SCaR 是回答这个问题的最佳对照物。 |
| B3 | Lee et al., *Adversarial Skill Chaining (T-STAR).* arXiv 2021 | 对抗正则化使前一技能终止分布匹配后一技能初始分布。 | 训练式组合的早期代表，第 2 节 related work 引用 + 表格对照即可。 |
| B4 | Agia et al., *STAP: Sequencing Task-Agnostic Policies.* ICRA 2023 | 用「状态到技能」映射与跳步控制器串行化任务无关策略。 | 免训序列化的另一路线（非扩散）。说明本文在扩散/少步语境下的独特性。 |
| B5 | Yang et al., *BOSS: Benchmark for Observation Space Shift in Long-Horizon Task.* arXiv 2502.15679 | 首次把「交接处的**观测空间偏移**（OOS）」列为长程技能串联失败的主因，给出 benchmark 与缓解（视觉特征对齐）。 | **重要引用**：为本文「切换失败模式」提供外部依据——硬切失败不仅来自高能场，也来自 OOS。本文时间掩码+可行性投影可视为对 OOS 的一种缓解；可在实验中加测 OOS 指标。 |
| B6 | Chen et al., *DeCo: Task Decomposition and Skill Composition for Zero-Shot Generalization in Long-Horizon 3D Manipulation.* arXiv 2505.00527 | 每个技能定义起始关键帧；工序完成后用**运动规划**自动转移到下一技能关键帧，实现自由组合。 | 免训组合的另一路线（基于关键帧 + 运动规划，非扩散）。与本文「动作分布间最优传输」形成方法论对照；审稿人可能建议把运动规划作为可行性投影的实现——可在 related work 中明确分工。 |
| B7 | Liang et al., *SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution.* ICLR 2024 | 端到端学习离散技能表示 + 技能条件扩散规划（Meta-World/LOReL 评测）。 | 与本文共享「技能条件扩散策略」设定；但 SkillDiffuser 是**训练式**、技能表示是学出来的、不做推理期组合。可作为「技能条件策略」的取法参考，实验中可用其技能库设置。 |
| B8 | Gu et al., *SDP: Learning Diffusion Policy from Primitive Skills for Robot Manipulation.* arXiv 2601.01948（HKU） | 8 个原语技能 + VLM 提取离散技能表示 + 轻量 router 分配技能 → 单技能条件 DP；CALVIN + LIBERO + 真机验证。 | SWDP.md 第 7 节钦定的「技能载体」。**注意**：SDP 是「每状态选一个技能的单技能策略」（router 训练式），并非「多技能组合执行」。本文 ChordCompose 可视为其**免训推理期组合升级**：冻结 SDP 式技能条件策略 + Chord 场做技能切换/串联。这是本文实验最自然的落点（LIBERO 增信实验直接对齐其 benchmark）。 |
| B9 | Zentner et al., *Conditionally Combining Robot Skills using LLMs.* ICRA 2024 | LLM 决定技能组合条件（when 切换）。 | 切换触发条件的高层规划侧（与本文低层动作交接互补），related work 引用。 |
| B10 | MOSAIC: A Skill-Centric Algorithmic Framework for Long-Horizon Manipulation Planning. arXiv 2504.16738 | 技能 + 扩散模型 + 运动规划混合规划框架。 | 免训组合大框架的竞争性方案，related work + 差异说明。 |

### 2.3 组 C：组合/并发约束扩散策略（Combine 相关）

| # | 文献 | 核心方法 | 与 Idea 9 的关系 |
|---|---|---|---|
| C1 | Razmjoo et al., *CCDP: Composition of Conditional Diffusion Policies with Guided Sampling.* IROS 2025 (arXiv 2503.15386) | 失败后**引导采样**：用「不成功动作」负样本修正采样分布；扩散分解把长程问题拆成子问题。 | 名字最像但方向不同：CCDP 是**失败恢复**的引导采样，不是多技能乘积专家组合。本文 Combine（乘积专家叠加 + Chord 平滑）与其互补；可作为 related work 区分，也可作为「组合失败后怎么办」的扩展讨论。 |
| C2 | Wang et al., *PoCo: Policy Composition from and for Heterogeneous Robot Learning.* RSS 2024 | 异质数据训练的策略通过**输出分布乘积**组合。 | 乘积专家组合的训练式代表；Combine 的对照。可引用其组合形式化。 |

### 2.4 组 D：少步/一步蒸馏（「单步价值」论证依据）

| # | 文献 | 核心方法 | 与 Idea 9 的关系 |
|---|---|---|---|
| D1 | Wang et al., *One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation.* ICRA 2024 | 扩散策略蒸馏到一步，实时控制。 | 本文「少步/一步蒸馏 DP」的直接实现参考（DDIM 蒸馏）；其「在快速模型上才有意义」的论证正是 ChordEdit 的镜像论点，可引用。 |
| D2 | Prasad et al., *Consistency Policy: Accelerating Visuomotor Policies via Consistency Distillation.* ICML 2024 | 一致性蒸馏加速机器人策略（1~4 步）。 | 本文蒸馏实现的第一选择（一致性蒸馏比 DDIM 蒸馏更稳、代码开源）；同时提供「少步策略更脆弱、更需要组合稳定化」的论据。 |
| D3 | Luo et al., *Latent Consistency Models.* ICLR 2024 | LCM 蒸馏。 | 背景引用。 |
| D4 | Kim et al., *Consistency Trajectory Models.* ICLR 2024 | CTM。 | 背景引用（可选）。 |

### 2.5 组 E：OT/Benamou–Brenier × 机器人

| # | 文献 | 核心方法 | 与 Idea 9 的关系 |
|---|---|---|---|
| E1 | Benamou & Brenier, *A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem.* Numerische Mathematik 2000 | 动态 OT 原始公式（最小动能 + 连续性方程）。 | 理论根基，直接引用。 |
| E2 | Le et al., *Accelerating Motion Planning via Optimal Transport.* NeurIPS 2023 | 用 OT 在演示分布间做几何插值加速运动规划。 | **最重要的 OT×机器人对照**：把 OT 用于「演示轨迹间的插值」。与本文「技能动作分布间的低能交接」同属 OT 机器人应用谱系；可作为理论 motivation 的引用，也需区分（他们是轨迹/配置空间插值，本文是条件分布间的推理期传输场）。 |
| E3 | Dhakan et al., *Concurrent Skill Composition using Ensemble of Primitive Skills.* IEEE TCDS 2022 | 原语技能集成做并发组合。 | Combine 的经典（非扩散）对照。 |
| E4 | Papadakis et al., *Optimal Transport with Proximal Splitting.* SIAM 2014 | 动态 OT 的凸优化求解。 | 背景引用（可选）。 |

---

## 3. 与 SWDP.md 各节的逐节对照

### 3.1 第 2 节（ChordEdit 迁移骨架）— 需要 2 处修正 + 1 处补充

- **修正 1（公式）**：`û = [δ·R(t−δ) + τ·R(t)]/(δ+τ)` → `û = [t·R(t−δ) + δ·R(t)]/(t+δ)`。
- **修正 2（符号）**：`x_pred = x_in + τ·û` 中的 τ 应为 λ（步长缩放），与平滑窗口 δ、查询时刻 t 分离。
- **补充（B_t 动作空间改写）**：附录 C 系数表已核实（§1.1）。动作空间 DP 通常用 **epsilon（噪声）预测
  或 x0 预测参数化**：`A_t^(ε) = −α̇/(ασ)`（VP）或 `A_t^(x0) = α̇/σ²`，仅依赖调度参数，**纯代数改写，零风险**。
  若用流匹配/速度参数化的 DP 则 `B_t ≡ I`，更简单。

### 3.2 第 3 节（问题形式化）— 基本成立，3 处细化建议

- 「技能条件扩散策略」应明确为 **SDP 式**（技能 one-hot/语言嵌入条件）或 **多任务 one-hot 条件**；
  实验上两者都做（Meta-World 用 one-hot，LIBERO 用语言/子目标）。
- 「动作块本身就是锚点」判断正确，但需补充：DP 推理中锚点是**当前待执行的动作块**
  （可以处于任意噪声水平，而非必须干净）——ChordEdit 的锚点是干净源图，
  动作域的对应物是「当前执行中的动作块 + 其干净估计 x0」。这会影响实现（锚点噪声水平选择），
  应作为超参（t 的选择）消融。
- Combine 的乘积专家形式化建议引用 A2（Composable Diffusion Models）与 C2（PoCo）。

### 3.3 第 4 节（Chord 组合场）— 机制成立，2 处补充

- Switch/Chain/Combine 三算子划分合理，与 ChordEdit「编辑=传输」一一对应。
- **补充**：应加入「有效时间步平移」消融（呼应复现分析论文），证明 δ>0 的平滑在动作域
  相对「直接用 t−δ 时刻查询」仍有增量（预期来自方差降低与时间掩码兼容）。
- 时间掩码在动作域的对应物更自然（动作块是序列，掩码 = 只传输交接窗口内的动作步），
  比图像域的「非编辑区域保持」实现更简洁——这是本文相对 ChordEdit 的一个真实增量，
  但**需要与 GSC 的「技能分布重叠」、SCaR 的「交接正则」做实验对照**才能立住。

### 3.4 第 5 节（理论）— 迁移判断正确，新增定理需重写表述

- 继承部分：逐条成立（§1.2 已核实）。
- **新增定理 1（零样本组合稳定性条件）**：Thm E.6 只给出「Chord 一致性常数 ≤ 朴素」，
  其稳定性条件 h·L<1 与「哪两个技能可组合」无关。要立住该定理，需要**新的数学内容**，例如：
  ```
  将 L = sup ‖∇_a v(a,t,o,s)‖ 分解为 L(s) + ΔL(s,s′)，
  其中 ΔL 为技能对间的场差异 Lipschitz 上界（可用技能嵌入距离 ||e_s − e_s'|| 上界化），
  导出可组合性充分条件：λ·(L(s) + c·||e_s − e_s'||) < 1 ⇒ 单步交接误差 ≤ ε(s,s′)。
  ```
  即把「技能对能否组合」绑定到**技能嵌入几何 + 局部 Lipschitz 估计**上。该条件可在实验中
  用沿交接轨迹的有限差分 Jacobian 范数估计验证（理论-实验闭环）。**风险**：若实验不支持，
  降级为「经验稳定性代理指标」而非硬定理——必须在实验中提前验证。
- **新增定理 2（组合最优性）**：直接继承 Thm E.4 的 O(δ) 间隙，无需新证明；表述为推论即可。
- **新增定理 3（可行性投影不破坏稳定性）**：Prop E.7 只证「平滑不收紧稳定性」。
  本文需要证明「投影算子 P_C（凸约束）后的 Euler 更新仍收缩」——
  标准的**非扩张（firmly non-expansive）投影论证**（P_C 是 1-Lipschitz 的，
  复合后 Jacobian 范数不增）可以补上。这是 ChordEdit 没有的，且有真实数学内容。**建议把这个
  作为头号新增定理**（比定理 1 更稳），定理 1 作为二号（带条件）。

### 3.5 第 7 节（验证方案）— 大方向对，需要落地与增补

- 基准落地：**Meta-World ML10（状态空间）主基准**（训练分钟级、消融全面）+
  **LIBERO-10（增信）**（复用本地 153GB 数据集 + turbovla-libero 环境 + SDP 对齐）。
- Baseline 增补：(a) 硬切换、(b) SCaR 式训练正则化组合、(c) 端到端多任务 DP、
  (d) GSC 式能量组合拼接、(e) Chord 场 vs 有效时间步平移（呼应复现论文）。
- 指标增补：除成功率/能量/NEF 外，加入 **OOS（观测空间偏移）指标**（BOSS 论文的度量）、
  交接处 jerk、分阶段技能达成率；理论一致性用「沿交接轨迹的 ‖∇_a v‖ 估计」预测成败。
- 少步/一步蒸馏：一致性蒸馏（D2）优先；蒸馏后复测「单步价值」——预期硬切在少步模型上失败更剧烈、
  Chord 收益更大（ChordEdit Figure 9 的动作域镜像）。

### 3.6 第 8 节（风险）— 更新

- 风险 1（少步蒸馏前置）维持：蒸馏已有开源实现（Consistency Policy），风险降低。
- 风险 2（低能=可行）维持，且按复现论文情报预期「Chord 主要改善平滑度/保真，
  目标达成需配合投影与后续推理」，指标分开报。
- 新增风险 4：**复现论文的「时间步平移」质疑**必须正面回应（§3.3 的消融）。
- 新增风险 5：**GSC/SCaR 撞车**——本文必须把「免训 + 少步 + 理论判据」三重差异讲清楚，
  且 baseline 要含训练式组合，证明免训路线在 unseen 组合上不输。

---

## 4. Novelty 增量定位（相对 ChordEdit 与技能组合邻域）

| 增量 | 相对 ChordEdit | 相对技能组合邻域（GSC/SCaR/SDP） | 强度评估 |
|---|---|---|---|
| **物理可行性投影**（运动学/动力学/动作界约束，非扩张投影 + 稳定性保持定理） | 新增（图像无物理约束） | GSC/SCaR 无推理期投影模块 | **强**（头号理论贡献候选） |
| **时间掩码**（交接窗口局部传输，轨迹段保持） | 对应「非编辑区域保持」，动作块序列使实现更自然 | 与 BOSS 的 OOS 问题形成呼应，可论证为 OOS 缓解 | 中强（需实验立住） |
| **零样本组合稳定性条件**（技能对可组合性判据） | 从 Thm E.6 再进一步，需新数学 | 无同类判据 | 中（有风险，需实验支持） |
| 免训 + 冻结少步策略上的组合 | ChordEdit 的免训性质继承 | 对 GSC（需训每技能）/SCaR（训练正则）形成方法论差异 | 强 |
| 有效时间步平移消融与回应 | 与复现论文对话 | — | 中（防御性贡献） |

---

## 5. 参考文献列表

1. Lu et al. ChordEdit: One-Step Low-Energy Transport for Image Editing. CVPR 2026. arXiv:2602.19083
2. Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights. arXiv 2026-08
3. Chi et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023 / IJRR 2025
4. Liu et al. Compositional Visual Generation with Composable Diffusion Models. ECCV 2022
5. Mishra et al. Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models. CoRL 2023
6. Chen et al. SCaR: Refining Skill Chaining for Long-Horizon Robotic Manipulation via Dual Regularization. NeurIPS 2024
7. Lee et al. Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization. arXiv:2111.07999, 2021
8. Agia et al. STAP: Sequencing Task-Agnostic Policies. ICRA 2023
9. Yang et al. BOSS: Benchmark for Observation Space Shift in Long-Horizon Task. arXiv:2502.15679
10. Chen et al. DeCo: Task Decomposition and Skill Composition for Zero-Shot Generalization. arXiv:2505.00527
11. Liang et al. SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions. ICLR 2024
12. Gu et al. SDP: Learning Diffusion Policy from Primitive Skills for Robot Manipulation. arXiv:2601.01948
13. Zentner et al. Conditionally Combining Robot Skills using LLMs. ICRA 2024
14. MOSAIC: A Skill-Centric Algorithmic Framework for Long-Horizon Manipulation Planning. arXiv:2504.16738
15. Razmjoo et al. CCDP: Composition of Conditional Diffusion Policies with Guided Sampling. IROS 2025. arXiv:2503.15386
16. Wang et al. PoCo: Policy Composition from and for Heterogeneous Robot Learning. RSS 2024
17. Wang et al. One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation. ICRA 2024
18. Prasad et al. Consistency Policy: Accelerating Visuomotor Policies via Consistency Distillation. ICML 2024
19. Luo et al. Latent Consistency Models. ICLR 2024
20. Ma et al. Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Manipulation. CVPR 2024
21. Benamou & Brenier. A Computational Fluid Mechanics Solution to the Monge-Kantorovich Mass Transfer Problem. Numerische Mathematik 2000
22. Le et al. Accelerating Motion Planning via Optimal Transport. NeurIPS 2023
23. Dhakan et al. Concurrent Skill Composition using Ensemble of Primitive Skills. IEEE TCDS 2022
24. Papadakis et al. Optimal Transport with Proximal Splitting. SIAM J. Imaging Sci. 2014
25. Ajay et al. Compositional Foundation Models for Hierarchical Planning (DiP). NeurIPS 2023

---

## 6. 补充调研(2026-09-03):工具对齐与长程 VLA 两篇

> 详见 [docs/design_optimization.md](design_optimization.md) 的完整提炼与 SWDP.md v3 对照。

26. Lei et al. *Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models (TAPT / VLAs-as-Tools).* arXiv:2605.13119
    - 高层 VLM 智能体 + 专用 VLA 工具家族;双向接口(调用 c=(g,z) / 进度反馈);
      TAPT = 调用对齐训练单元 + 工具家族残差适配器(参数 +9%);
      LIBERO-Long 97.2 / RoboTwin 62.5 / CF-Long 调用忠实度(Faithful/Non-biased Rate)。
    - 与 Idea 9:本文 = 冻结执行器上的**免训**工具交接(对照其训练式对齐),分水岭实验见 SWDP.md §7。
27. Zeng et al. *HELM: Harness-Enhanced Long-horizon Memory for Vision-Language-Action Manipulation.* arXiv:2604.18791
    - 反应式 VLA 三缺陷(记忆/验证/恢复)→ 情景记忆 EMM(CLIP 键值) + 学习验证器 + 恢复控制器;
      LIBERO-Long +23.1、扰动恢复 54.2%。
    - 与 Idea 9:本文的可行性投影 = 免训执行前验证,场能量 = 免训恢复触发信号(对标其验证/恢复环)。
