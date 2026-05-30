"""
learnable_finger.py — 学习型8维指纹（增量新增，原有文件零修改）
======================================================================
核心问题：
  现有指纹是随机初始化或规则编码（模态热编码）。
  稀疏路由的相似度匹配只能靠运气——指纹并不真正反映
  "这个单元擅长处理什么语义"。

解决方案：
  用梯度下降在推理反馈信号上更新8维指纹，
  使得"语义相似的输入→路由到相似指纹的单元"这一对应关系
  随使用次数增加而自动收敛。

无 PyTorch 的纯 numpy 实现：
  使用简化的对比学习（Contrastive Learning）：
    正样本：同一输入 x 经过某单元后输出高质量结果 → 拉近 x_finger 与 unit_finger
    负样本：x 经过该单元后输出低质量结果 → 推远 x_finger 与 unit_finger
  质量信号来源：输出范数/方差/余弦稳定性（无需标签）

更新规则（梯度模拟，纯 numpy）：
  Δfinger_unit = η × quality × (x_finger - finger_unit)
  finger_unit  = normalize(finger_unit + Δfinger_unit)

λ函数单元 + 同伦形变大模型系统 · 学习型指纹层
"""

from __future__ import annotations
import os, sys, json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_utils import LambdaUnit, FINGER_DIM
from sparse_router import extract_query_finger, cosine_sim

# ── 默认超参 ─────────────────────────────────────────────────────────────
DEFAULT_LR          = 0.05    # 指纹更新学习率
DEFAULT_MOMENTUM    = 0.9     # 动量系数（平滑更新方向）
DEFAULT_HISTORY_LEN = 50      # 质量信号历史窗口


# ══════════════════════════════════════════════════════════════════════
# 质量信号估算（无需标签）
# ══════════════════════════════════════════════════════════════════════
def estimate_output_quality(
    x_in:  np.ndarray,
    x_out: np.ndarray,
    t:     float,
) -> float:
    """
    估算单元推理质量 → [0, 1]，越高越好
    三路信号融合：
      1. 输出范数稳定性（非零且不爆炸）
      2. 输入→输出余弦相似度（信息保留程度）
      3. t 感知因子（高 t 时允许更低范数）

    注意：此函数不依赖任何标签或监督信号
    """
    out_norm = float(np.linalg.norm(x_out.astype(np.float32)))
    in_norm  = float(np.linalg.norm(x_in.astype(np.float32)))

    # 信号1：范数合理性（目标范数随 t 线性降低）
    target_norm = in_norm * (1.0 - t * 0.5)
    norm_ratio  = out_norm / (target_norm + 1e-8)
    norm_score  = 1.0 - min(1.0, abs(norm_ratio - 1.0))   # 越接近1越好

    # 信号2：余弦相似度（维度对齐后计算，处理 4096→512 的降维情况）
    in_f  = x_in.astype(np.float32).flatten()
    out_f = x_out.astype(np.float32).flatten()
    n     = min(len(in_f), len(out_f))
    in_t, out_t = in_f[:n], out_f[:n]
    denom = np.linalg.norm(in_t) * np.linalg.norm(out_t) + 1e-8
    cos_score = max(0.0, float(np.dot(in_t, out_t) / denom))

    # 信号3：无 NaN / Inf
    valid_score = 1.0 if np.all(np.isfinite(x_out)) else 0.0

    quality = 0.4 * norm_score + 0.4 * cos_score + 0.2 * valid_score
    return float(np.clip(quality, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════
# 单元指纹状态（每个 λ单元独立）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class FingerState:
    unit_id:   str
    finger:    np.ndarray                        # 当前 8维指纹（float32）
    momentum:  np.ndarray = field(default_factory=lambda: np.zeros(FINGER_DIM, dtype=np.float32))
    update_count: int     = 0
    quality_history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_LEN))

    def mean_quality(self) -> float:
        if not self.quality_history:
            return 0.5
        return float(sum(self.quality_history) / len(self.quality_history))


# ══════════════════════════════════════════════════════════════════════
# 学习型指纹管理器
# ══════════════════════════════════════════════════════════════════════
class LearnableFingerManager:
    """
    管理一组 λ单元的可学习指纹。
    在每次推理后，根据质量信号更新指纹，
    使稀疏路由越来越准确地匹配"擅长的输入"。

    用法（与 SparseRouter 结合）：
        manager = LearnableFingerManager(units, lr=0.05)
        # 推理后更新
        manager.update(unit, x_in, x_out, global_t)
        # 获取当前指纹（用于路由）
        finger = manager.get_finger(unit.unit_id)
        # 保存学到的指纹回 *.bin 文件
        manager.save_fingers()
    """

    def __init__(
        self,
        units:    list,           # LambdaUnit 列表
        lr:       float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
    ):
        self.lr       = lr
        self.momentum = momentum
        self._states: dict[str, FingerState] = {}

        for unit in units:
            f = unit.finger.astype(np.float32) if unit.finger is not None \
                else np.random.randn(FINGER_DIM).astype(np.float32)
            f = self._normalize(f)
            self._states[unit.unit_id] = FingerState(
                unit_id = unit.unit_id,
                finger  = f,
            )

        self._unit_map: dict[str, LambdaUnit] = {u.unit_id: u for u in units}
        print(f"  [LearnableFinger] 初始化 {len(units)} 个单元指纹  lr={lr}")

    # ── 指纹更新（每次推理后调用）──────────────────────────────────
    def update(
        self,
        unit:    LambdaUnit,
        x_in:   np.ndarray,
        x_out:  np.ndarray,
        global_t: float,
        query_finger: Optional[np.ndarray] = None,
    ) -> float:
        """
        根据推理质量更新单元指纹
        返回本次质量得分
        """
        state   = self._states.get(unit.unit_id)
        if state is None:
            return 0.5

        quality = estimate_output_quality(x_in, x_out, global_t)
        state.quality_history.append(quality)

        # 查询指纹（输入特征摘要）
        if query_finger is None:
            query_finger = extract_query_finger(x_in).astype(np.float32)
        else:
            query_finger = query_finger.astype(np.float32)

        # 对比学习方向：
        #   quality > 0.5 → 正样本：将 unit_finger 拉向 query_finger
        #   quality < 0.5 → 负样本：将 unit_finger 推离 query_finger
        direction = query_finger - state.finger
        signed_direction = direction * (quality - 0.5) * 2   # 映射到 [-1, 1]

        # 动量更新（平滑方向，防止抖动）
        state.momentum = (self.momentum * state.momentum
                          + (1 - self.momentum) * signed_direction)

        # 梯度步骤
        state.finger = state.finger + self.lr * state.momentum
        state.finger = self._normalize(state.finger)
        state.update_count += 1

        return quality

    # ── 批量更新（适合并联组整体更新）──────────────────────────────
    def batch_update(
        self,
        units:    list,
        x_in:    np.ndarray,
        outputs: list[np.ndarray],
        global_t: float,
    ) -> list[float]:
        """批量更新多个单元的指纹"""
        query_finger = extract_query_finger(x_in).astype(np.float32)
        return [
            self.update(unit, x_in, out, global_t, query_finger)
            for unit, out in zip(units, outputs)
        ]

    # ── 指纹查询 ────────────────────────────────────────────────────
    def get_finger(self, unit_id: str) -> np.ndarray:
        """获取当前学习到的指纹（float16，与 LambdaUnit.finger 格式一致）"""
        state = self._states.get(unit_id)
        if state is None:
            return np.zeros(FINGER_DIM, dtype=np.float16)
        return state.finger.astype(np.float16)

    def sync_to_units(self):
        """将学习后的指纹同步回对应的 LambdaUnit 实例（内存中）"""
        for uid, state in self._states.items():
            unit = self._unit_map.get(uid)
            if unit:
                unit.finger = state.finger.astype(np.float16)

    # ── 持久化 ──────────────────────────────────────────────────────
    def save_fingers(self, save_dir: Optional[str] = None):
        """
        将学习后的指纹写回对应的 *.bin 文件
        采用"只写指纹区"策略：定位区块4偏移，原位修改，不重写整个文件
        """
        saved = 0
        for uid, state in self._states.items():
            unit = self._unit_map.get(uid)
            if unit is None or not os.path.exists(unit.file_path):
                continue
            try:
                offset = self._calc_finger_offset(unit)
                with open(unit.file_path, "r+b") as f:
                    f.seek(offset)
                    f.write(state.finger.astype(np.float16).tobytes())
                unit.finger = state.finger.astype(np.float16)
                saved += 1
            except Exception as e:
                print(f"  [LearnableFinger] 写回失败 {uid}: {e}")

        print(f"  [LearnableFinger] 已将 {saved} 个单元指纹写回 *.bin 文件")

    def load_fingers(self):
        """从 *.bin 文件重新加载当前指纹（覆盖内存中的状态）"""
        for uid, state in self._states.items():
            unit = self._unit_map.get(uid)
            if unit is None:
                continue
            unit.load()   # 重新加载整个单元（简化实现）
            if unit.finger is not None:
                state.finger = unit.finger.astype(np.float32)

    def save_checkpoint(self, path: str):
        """将所有指纹状态保存为 JSON checkpoint（可读格式）"""
        ckpt = {
            uid: {
                "finger":       state.finger.tolist(),
                "update_count": state.update_count,
                "mean_quality": round(state.mean_quality(), 4),
            }
            for uid, state in self._states.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)
        print(f"  [LearnableFinger] checkpoint 已保存: {path}")

    def load_checkpoint(self, path: str):
        """从 JSON checkpoint 恢复指纹状态"""
        with open(path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        for uid, data in ckpt.items():
            if uid in self._states:
                self._states[uid].finger       = np.array(data["finger"], dtype=np.float32)
                self._states[uid].update_count = data.get("update_count", 0)
        print(f"  [LearnableFinger] checkpoint 已恢复: {path} "
              f"({len(ckpt)} 个单元)")

    # ── 统计摘要 ────────────────────────────────────────────────────
    def stats(self) -> dict:
        qualities = [s.mean_quality() for s in self._states.values()]
        updates   = [s.update_count   for s in self._states.values()]
        return {
            "units":          len(self._states),
            "total_updates":  sum(updates),
            "quality_mean":   round(sum(qualities) / max(1, len(qualities)), 4),
            "quality_min":    round(min(qualities), 4),
            "quality_max":    round(max(qualities), 4),
            "lr":             self.lr,
            "momentum":       self.momentum,
        }

    def print_fingers(self):
        """打印当前所有单元指纹状态"""
        print(f"\n  {'单元ID':20s}  {'更新次数':>6}  {'均值质量':>8}  指纹（8维）")
        print(f"  {'─'*20}  {'─'*6}  {'─'*8}  {'─'*32}")
        for uid, state in self._states.items():
            f_str = " ".join(f"{v:+.3f}" for v in state.finger)
            print(f"  {uid:20s}  {state.update_count:>6}  "
                  f"{state.mean_quality():>8.4f}  {f_str}")

    # ── 内部工具 ────────────────────────────────────────────────────
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v) + 1e-8
        return (v / norm).astype(np.float32)

    @staticmethod
    def _calc_finger_offset(unit: LambdaUnit) -> int:
        """计算区块4（指纹区）的字节起始偏移"""
        from base_utils import HEAD_LEN, FP16_SIZE, FINGER_DIM
        size_w = unit.dim_origin * unit.dim_skeleton
        weight_len = size_w * FP16_SIZE * 2   # W0 + W1
        bias_len   = unit.dim_skeleton * FP16_SIZE

        # 扫描控制规则区长度
        offset = HEAD_LEN
        with open(unit.file_path, "rb") as f:
            f.seek(HEAD_LEN)
            while True:
                b = f.read(1)
                offset += 1
                if not b or b == b"\n":
                    break

        return offset + weight_len + bias_len
