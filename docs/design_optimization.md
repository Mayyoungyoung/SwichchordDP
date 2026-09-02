# 设计优化说明：从「推理期动作平滑拼接」升级为「面向长程具身智能的工具对齐组合框架」

> 依据：对 Tool-Aligned VLA(TAPT, arXiv:2605.13119)与 Harness VLA(HELM, arXiv:2604.18791)
> 的深度调研，逐项对照 /home/jia/DP/SWDP.md 后给出的设计增强方案。
> 配套文档：SWDP.md(v3)、docs/survey.md、docs/experiment_report.md。

---

## 1. 两篇论文核心方法提炼

### 1.1 TAPT: Towards Long-horizon Embodied Agents with Tool-Aligned VLAs（VLAs-as-Tools）

- **问题定位**：VLA 在长程任务上受「扩展的闭环规划 + 多样的物理操作」双重负担所限；
  端到端 VLA 对演示偏差敏感、恢复失败率高。
- **核心策略**：把负担分配给「高层 VLM 智能体（时序推理：场景分析/全局规划/恢复）」
  与「一族专用 VLA 工具（有界局部物理操作）」。
- **关键模块**：
  1. **VLA 工具家族接口**（双向）：调用消息 `c_k = (g_k, z_k)`，`g_k` 为离散工具家族标签
     （grasp/open/place/rotate…），`z_k` 为场景接地子任务指令（对象/关系/期望局部效果）；
     反馈消息 `r_k` 含**执行中进度反馈** → 高层智能体**事件触发式重规划**
     （无需每步轮询，VLM 调用次数从 ~109/回合降至 ~2）。
  2. **TAPT（Tool-Aligned Post-Training）**：
     - **调用对齐训练单元**：把完整轨迹分段为有界窗口，每窗标注调用 `(g, z)` 与
       进度目标 `p*`；IL 与 RL 共享同一「有界调用」单元（RL 的 reward = 局部完成谓词
       ψ_{z,g} 在 horizon 内成立则 1）。
     - **工具家族残差适配器**：共享预训练 VLA 骨干 + 按 `g` 选择的低秩残差
       ΔW_g（参数仅 +9%）；`g` 控制执行路径而非仅是语言 token。
     - **目标**：动作克隆 + 进度回归双目标，让 VLA 学会接口两侧（执行调用 + 返回进度）。
- **评测基准**：LIBERO-Long（成功率 97.2，π0.5 骨干 +4.8）、RoboTwin（62.5，+23.1）、
  CALVIN（验证原生子任务结构下增益仍成立）；**调用忠实度**（invocation fidelity）在
  LIBERO-CF-Long 反事实套件上以 Faithful Rate / Non-biased Rate 度量
  （π0.5 +30.0 / +15.0）；消融证明「只包 planner 不训执行器不可靠」。
- **主要结论**：执行器必须被「对齐」成可靠工具（TAPT），高层规划才有意义；
  进度反馈使低开销事件触发重规划可行。

### 1.2 HELM: Harness-Enhanced Long-horizon Memory（Harness VLA）

- **问题定位**：反应式 VLA 的三个结构性缺陷——**记忆缺陷**（只看到最近几秒，忘记已完成步骤）、
  **验证缺陷**（不「三思而后行」，错误动作无执行前拦截）、**恢复缺陷**（犯错后无纠正上下文）。
- **核心策略**：模型无关的「缰绳」（harness）执行环，包裹任意 VLA 骨干（OpenVLA/Octo）。
- **关键模块**：
  1. **情景记忆模块（EMM）**：键值记忆 `{(k_i, v_i)}`，键 = 过去观测的 CLIP 压缩视觉签名，
     值 = 图像/正在执行的子目标/时间/机器人物理状态；检索相似历史注入当前决策。
  2. **学习到的状态验证器**：执行前对 VLA 提议的动作做失败预判，拦截不可行动作。
  3. **恢复控制器**：失败被预测/检测时触发恢复例程。
  4. **执行环**：retrieve → propose（VLA）→ verify → act / recover。
- **评测基准**：LIBERO-Long（+23.1 成功率）、扰动注入任务的恢复成功率（54.2%）。
- **主要结论**：单纯增大上下文的收益远小于「情景记忆 + 预执行验证 + 回滚重规划」；
  长程能力的短板不在生成而在**验证与恢复**。

---

## 2. 两篇论文 × Idea 9 映射与当前薄弱点

| 维度 | TAPT | HELM | 当前 SWDP（v2 状态） | 判定 |
|---|---|---|---|---|
| 高层任务规划/工具语义 | VLM 智能体 + 工具家族标签 g + 接地指令 z | 子目标驱动的记忆+重规划 | 无：仅 one-hot 技能 token 的底层切换 | **薄弱** |
| 工具/技能与执行器的对齐机制 | TAPT 训练对齐（IA 单元+残差适配器） | 外部 harness 不改骨干 | Chord 场免训组合（有理论保证） | 部分强项（免训），但缺「对齐」视角 |
| 语言条件融合 | z 接地指令 + 进度回归 | CLIP 视觉键 + 子目标文本 | 无语言条件（状态空间 MLP） | **薄弱** |
| 失败恢复/技能衔接 | 进度反馈 → 事件触发重规划 | 验证器 + 恢复控制器 | 可行性投影（执行前硬约束）+ 时间掩码 | 部分强项，缺「验证→恢复」闭环 |
| 评测 | LIBERO-Long/RoboTwin/CALVIN + CF-Long 忠实度 | LIBERO-Long + 扰动恢复 | Meta-World 状态空间 12 回合 + LIBERO 离线拼接 | **薄弱** |
| novelty 表述 | — | — | 「推理期动作平滑拼接」 | **不足以支撑顶会** |

**结论**：当前方案在「低层交接的数学机制」上独树一帜（Chord 场 + 稳定性理论），
但缺三块拼图：(1) 高层规划/工具语义；(2) VLA 骨干与语言条件；(3) 长程在线评测与
忠实度/恢复指标。升级路径是把 Chord 场重新定位为「**工具调用交接的免训对齐层**」，
与 TAPT（训练对齐）与 HELM（验证/恢复 harness）形成互补而非竞争。

---

## 3. 新增 novelty 增量说明（升级后）

1. **工具语义条件 Chord 场（头号）**：技能条件从 one-hot 升级为「工具家族标签 g +
   场景接地指令 z 的嵌入 e(g,z)」（冻结 VLA/CLIP 文本编码器提取）。残差场
   `R(a,τ) = E[B_τ(Q(z,τ,o,(g',z')) − Q(z,τ,o,(g,z)))]`，交接语义 = 两个工具调用的
   条件差。相对 TAPT 的**训练式**工具对齐，本文是**冻结策略上的免训工具交接**，
   且带 L² 收缩与稳定性理论；相对 ChordEdit 增加了工具语义（不止离散 token）。
2. **双层框架「规划-组合」（Plan-and-Compose）**：高层 VLM 产出调用序列
   `{(g_k, z_k)}`（复用 TAPT 接口定义），低层 ChordCompose 按调用序列做
   Switch/Chain/Combine 交接；**Chord 场能量 ‖û‖² 作为免训的交接风险信号**
   反馈给高层，触发事件式重规划——对应 TAPT 的进度反馈与 HELM 的验证器，
   但零额外训练。
3. **可行性投影 = 免训执行前验证**（与 HELM 验证器对照）：投影后的 Euler 稳定性定理 +
   实验证据（LIBERO 拼接 MSE -36%）。可进一步输出「投影残差」作为失败预判信号。
4. **工具对组合稳定性条件**：`L((g,z),(g',z')) = L(g) + c·‖e(g,z) − e(g',z')‖`，
   把「哪两个工具调用可以免训交接、误差多大」绑定到工具语义距离——TAPT/HELM 均无
   此类可组合性判据。
5. **少步工具执行**：工具 = 蒸馏到少步的 VLA 片段（TAPT 的 bounded invocation 天然
   适合少步），Chord 场在少步执行器上证明「单步价值」。

---

## 4. 需要补做的实验清单（按优先级）

1. **LIBERO-Long 在线评测**（BDDL 成功判定，复用本地数据与 turbovla-libero 环境）：
   工具家族 = {grasp, open, close, place, rotate}，指令 z = 任务子目标语言；
   对比 naive 硬切 / chord / eff_shift / energy / e2e-trained / TAPT 式训练对齐。
2. **调用忠实度评测（LIBERO-CF-Long 式）**：改目标对象/空间关系/截断指令的反事实任务，
   报告 Faithful Rate / Non-biased Rate——直接对标 TAPT 的指标体系。
3. **VLA 骨干结合**：用冻结 OpenVLA-OFT / Octo 的视觉-语言特征替换状态观测
   （至少 LIBERO 图像观测 CNN-DP 起步），验证「冻结 VLA 特征 + Chord 交接」路径。
4. **高层规划器接入**：VLM（或脚本化 oracle planner）输出调用序列 {(g_k, z_k)}，
   以场能量触发重规划；对比 TAPT 的「planner 包裹不训执行器不可靠」结论，
   证明 Chord 场让**未对齐的冻结执行器**也能被可靠调用（这是与 TAPT 的分水岭实验）。
5. **恢复评测**：注入扰动（掉落/碰撞）后场能量上升 → 触发重规划的成功率
   （对标 HELM 54.2% 恢复率）。
6. 关键对比增至 24-50 回合/任务；一致性蒸馏修复（c_skip/c_out 参数化）复证单步价值。

## 5. 对现有实验代码的改动建议

| 文件 | 改动 |
|---|---|
| code/swdp/nets.py | FiLM 条件支持连续嵌入：技能条件 = [one-hot(g); e(z)]（e(z) 由冻结 CLIP/编码器离线预计算） |
| code/swdp/chord_compose.py | `switch/chain/combine` 条件从 one-hot 改为 (g, z) 嵌入对；场能量输出已具备，补充「风险信号」接口 |
| code/swdp/planner.py（新增） | 高层规划器接口：VLM 调用 stub + 脚本化 oracle planner（Meta-World 用阶段图，LIBERO 用子目标语言），输出调用序列 |
| code/libero/ | 增加在线评测（BDDL 成功判定）与 CF-Long 反事实任务构造；图像观测 CNN-DP 变体 |
| code/metaworld/eval_compose.py | 增加「场能量触发重规划」消融与扰动恢复评测 |

改动原则：接口先行（条件嵌入 + planner 接口），评测逐步替换，不破坏已跑通的
Meta-World/LIBERO 离线结论。
