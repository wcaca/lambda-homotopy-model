"""
sparse_router.py — 8维指纹稀疏路由（增量新增，原有文件零修改）
核心命题：并联组内只激活指纹相似度最高的 top-k 单元，
         计算量从 O(N) 压到 O(k)，k << N。

路由策略：
  1. 输入向量提取"查询指纹"（8维），与组内各单元指纹做余弦相似度
  2. 取 top-k 激活（阈值门控 + 数量上限双重约束）
  3. 只对激活单元执行 forward，结果按相似度加权融合
  4. 未激活单元直接跳过，零计算开销

λ函数单元 + 同伦形变大模型系统 · 稀疏路由层
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional

# ── 全局路由常量 ────────────────────────────────────────────────────────
FINGER_DIM    = 8
DEFAULT_TOP_K = 2        # 默认最多激活单元数
DEFAULT_THRESHOLD = 0.1  # 余弦相似度阈值（低于此值不激活）


# ══════════════════════════════════════════════════════════════════════
# 指纹工具函数
# ══════════════════════════════════════════════════════════════════════
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度，数值稳定版"""
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    denom = (np.linalg.norm(a_f) * np.linalg.norm(b_f)) + 1e-8
    return float(np.dot(a_f, b_f) / denom)


def extract_query_finger(x: np.ndarray, finger_dim: int = FINGER_DIM) -> np.ndarray:
    """
    从输入向量提取查询指纹（8维）
    策略：取前 finger_dim 个均方根分段特征，归一化
    可替换为：模态指纹直传、学习型指纹提取器等
    """
    x_f   = x.astype(np.float32).flatten()
    seg   = max(1, len(x_f) // finger_dim)
    segs  = [x_f[i*seg:(i+1)*seg] for i in range(finger_dim)]
    # 每段取 RMS（均方根）
    rms   = np.array([np.sqrt(np.mean(s**2)) if len(s) > 0 else 0.0 for s in segs],
                     dtype=np.float32)
    norm  = np.linalg.norm(rms) + 1e-8
    return (rms / norm).astype(np.float16)


# ══════════════════════════════════════════════════════════════════════
# 稀疏路由器核心
# ══════════════════════════════════════════════════════════════════════
class SparseRouter:
    """
    8维指纹稀疏路由器
    管理一组 LambdaUnit（或 ModalLambdaUnit），执行 top-k 稀疏激活。

    用法：
        router = SparseRouter(units, top_k=2, threshold=0.1)
        output = router.route_forward(x, global_t)
    """
    def __init__(
        self,
        units:      list,           # LambdaUnit 列表（含 .finger 属性）
        top_k:      int   = DEFAULT_TOP_K,
        threshold:  float = DEFAULT_THRESHOLD,
        merge_rule: str   = "weighted_sum",  # weighted_sum | sum | mean
    ):
        self.units      = units
        self.top_k      = max(1, min(top_k, len(units)))
        self.threshold  = threshold
        self.merge_rule = merge_rule

        # 预缓存所有单元指纹矩阵，避免重复转换
        self._finger_matrix: Optional[np.ndarray] = None
        self._build_finger_matrix()

        # 统计信息
        self.total_calls      = 0
        self.total_activated  = 0
        self.total_skipped    = 0

    def _build_finger_matrix(self):
        """预构建指纹矩阵（N × 8），加速批量相似度计算"""
        fingers = []
        for u in self.units:
            if u.finger is not None:
                fingers.append(u.finger.astype(np.float32))
            else:
                fingers.append(np.zeros(FINGER_DIM, dtype=np.float32))
        self._finger_matrix = np.stack(fingers, axis=0)   # shape: (N, 8)

    def _compute_similarities(self, query_finger: np.ndarray) -> np.ndarray:
        """批量计算 query 与所有单元指纹的余弦相似度 → shape: (N,)"""
        q = query_finger.astype(np.float32).flatten()[:FINGER_DIM]
        # 广播余弦相似度
        norms  = np.linalg.norm(self._finger_matrix, axis=1) + 1e-8
        q_norm = np.linalg.norm(q) + 1e-8
        sims   = self._finger_matrix.dot(q) / (norms * q_norm)
        return sims.astype(np.float32)

    def _select_top_k(self, sims: np.ndarray) -> tuple[list[int], np.ndarray]:
        """
        top-k 激活选择
        双重过滤：相似度 ≥ threshold  AND  数量 ≤ top_k
        :return: (激活索引列表, 对应相似度权重)
        """
        # 阈值过滤
        candidates = np.where(sims >= self.threshold)[0]

        if len(candidates) == 0:
            # 所有单元都低于阈值 → 兜底激活相似度最高的 1 个
            candidates = np.array([int(np.argmax(sims))])

        # 在候选中取 top_k
        if len(candidates) > self.top_k:
            top_idx  = np.argsort(sims[candidates])[-self.top_k:]
            selected = candidates[top_idx]
        else:
            selected = candidates

        weights = sims[selected]
        # softmax 归一化权重（让权重之和=1，数值稳定）
        w_exp   = np.exp(weights - weights.max())
        weights = w_exp / (w_exp.sum() + 1e-8)
        return selected.tolist(), weights

    def route_forward(
        self,
        x:        np.ndarray,
        global_t: float,
        query_finger: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        稀疏路由前向推理
        :param x:             输入向量
        :param global_t:      全局形变参数 t
        :param query_finger:  可选外部指纹（如模态指纹直传），None 则从 x 提取
        :return:              激活单元加权输出（O(k) 复杂度）
        """
        self.total_calls += 1

        # 步骤1：提取查询指纹
        if query_finger is None:
            query_finger = extract_query_finger(x)

        # 步骤2：批量相似度计算
        sims = self._compute_similarities(query_finger)

        # 步骤3：top-k 激活选择
        selected_idx, weights = self._select_top_k(sims)
        activated  = len(selected_idx)
        skipped    = len(self.units) - activated

        self.total_activated += activated
        self.total_skipped   += skipped

        # 步骤4：只对激活单元执行 forward
        outputs = []
        for i, idx in enumerate(selected_idx):
            out = self.units[idx].forward(x, global_t)
            outputs.append((out.astype(np.float32), float(weights[i])))

        # 步骤5：结果融合
        return self._merge(outputs, x)

    def _merge(self, outputs: list[tuple[np.ndarray, float]], x: np.ndarray) -> np.ndarray:
        """加权融合激活单元的输出"""
        if not outputs:
            return x.astype(np.float16)

        if self.merge_rule == "weighted_sum":
            result = sum(out * w for out, w in outputs)
        elif self.merge_rule == "sum":
            result = sum(out for out, _ in outputs)
        elif self.merge_rule == "mean":
            result = np.mean([out for out, _ in outputs], axis=0)
        else:
            result = sum(out * w for out, w in outputs)

        # L2 归一化，与全激活路径保持量级一致
        norm = np.linalg.norm(result) + 1e-8
        return (result / norm).astype(np.float16)

    def stats(self) -> dict:
        """返回路由统计信息"""
        if self.total_calls == 0:
            return {"calls": 0, "avg_activated": 0, "avg_skipped": 0,
                    "activation_rate": 0.0}
        return {
            "calls":           self.total_calls,
            "avg_activated":   round(self.total_activated / self.total_calls, 2),
            "avg_skipped":     round(self.total_skipped   / self.total_calls, 2),
            "activation_rate": round(self.total_activated /
                                     (self.total_activated + self.total_skipped + 1e-8), 4),
            "top_k":           self.top_k,
            "threshold":       self.threshold,
            "total_units":     len(self.units),
        }

    def reset_stats(self):
        self.total_calls = self.total_activated = self.total_skipped = 0

    def __repr__(self) -> str:
        return (f"<SparseRouter units={len(self.units)} "
                f"top_k={self.top_k} threshold={self.threshold}>")
