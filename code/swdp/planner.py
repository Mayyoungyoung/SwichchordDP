"""高层规划器接口(Plan-and-Compose 的高层侧)与脚本化 oracle planner。

对应 SWDP.md v3 §3.3:高层输出 TAPT 式调用序列 {(g_k, z_k)},低层 ChordCompose 执行交接。
场能量 ‖û‖²(由 chord_compose.switch 返回的 info["energy"])作为免训风险信号,
触发事件式重规划(对应 TAPT 的进度反馈与 HELM 的验证器,但零额外训练)。

真实 VLM 可替换 BasePlanner(接口同 ScriptedPlanner);本模块先提供脚本化 oracle
以便跑通闭环实验,再做 VLM 接入。
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch


@dataclass
class ToolCall:
    """TAPT 式工具调用 c = (g, z): 离散工具家族 g + 场景接地指令 z。"""
    g: str                                  # 工具家族: grasp/open/close/place/rotate/...
    z: Optional[str] = None                 # 场景接地指令(语言);None 退化为纯家族
    steps: int = 20                         # 有界执行窗口
    meta: dict = field(default_factory=dict)


def encode_call(call: ToolCall, g_vocab: dict, enc: Optional[Callable[[str], np.ndarray]],
                embed_dim: int = 64, seed: int = 0) -> np.ndarray:
    """调用嵌入 e(g,z) = [one-hot(g); enc(z)]。

    enc: 冻结文本编码器(CLIP/OpenVLA 文本塔),把 z 映射到 embed_dim 维;
    若 enc=None,用固定哈希嵌入(无语言信号,退化为 one-hot 语义)。
    返回 [n_g + embed_dim] 向量(与 nets.FiLM 的 skill_dim 输入兼容)。
    """
    n_g = len(g_vocab)
    onehot = np.zeros(n_g, dtype=np.float32)
    onehot[g_vocab[call.g]] = 1.0
    if call.z is not None and enc is not None:
        zemb = np.asarray(enc(call.z), dtype=np.float32).reshape(-1)[:embed_dim]
        zemb = np.pad(zemb, (0, max(0, embed_dim - len(zemb))))[:embed_dim]
    else:
        # 无编码器: 对 z 字符串做确定性哈希嵌入(保持可复现,无语言语义)
        h = abs(hash(call.z)) % (2 ** 31) if call.z else seed
        rng = np.random.default_rng(h)
        zemb = rng.standard_normal(embed_dim).astype(np.float32)
    zemb = zemb / (np.linalg.norm(zemb) + 1e-8)
    return np.concatenate([onehot, zemb]).astype(np.float32)


class BasePlanner:
    """高层规划器接口: 观测 -> 调用序列; 风险信号 -> 事件式重规划。"""

    def plan(self, obs) -> List[ToolCall]:
        raise NotImplementedError

    def replan(self, obs, risk: float, threshold: float) -> Optional[List[ToolCall]]:
        """risk = 最近一次交接的场能量 ‖û‖²。默认策略:超阈值返回 None(继续),
        由子类实现真实重规划。"""
        if risk > threshold:
            return self.plan(obs)
        return None


class ScriptedPlanner(BasePlanner):
    """脚本化 oracle planner: 按预定义阶段图/序列输出调用序列(不读观测,供闭环验证)。

    用于:
    - Meta-World: 技能序列已知(如 reach->grasp->lift->carry->place);
    - LIBERO: 按任务子目标语言构造 (g, z) 序列;
    - 恢复实验: 重规划 = 回到失败调用并追加恢复调用(如 place 失败 -> 重新 grasp)。
    """

    def __init__(self, sequence: List[ToolCall], g_vocab: dict,
                 recovery: Optional[List[ToolCall]] = None):
        self.sequence = sequence
        self.g_vocab = g_vocab
        self.recovery = recovery or []
        self._pos = 0

    def plan(self, obs=None) -> List[ToolCall]:
        return list(self.sequence)

    def replan(self, obs, risk: float, threshold: float):
        if risk > threshold and self.recovery:
            return list(self.recovery)
        return None

    # ---- 低层执行侧的辅助: 序列 -> 条件张量 ----
    def embed_sequence(self, calls: List[ToolCall],
                       enc: Optional[Callable[[str], np.ndarray]] = None,
                       device: str = "cuda") -> torch.Tensor:
        """把调用序列编码为条件张量 [K, n_g + embed_dim](供 chord_compose 使用)。"""
        embs = [encode_call(c, self.g_vocab, enc) for c in calls]
        return torch.from_numpy(np.stack(embs)).float().to(device)
