"""
wavelet_basis.py — 用小波变换基构造 W0/W1
======================================================================
核心思想：
  Haar 小波和 DCT 基是自然信号（图像/音频）最重要的稀疏化基。
  用小波基列作为 W0，可以让 λ单元自然实现"频率分析"：

  W0：完整的小波基（保留全频率，从低频到高频）
  W1：只保留低频分量的截断小波基（高频归零 → 平滑/降噪效果）

  当 t 从 0→1：
    t=0  全频率响应（保留高频细节）
    t=1  只保留低频（平滑/去噪）
    中间  连续渐进低通滤波

物理意义：
  - W0·x 等价于"全波段特征提取"
  - W1·x 等价于"低通滤波后特征提取"
  - t 形变 = 低通截止频率的连续调节

零依赖：纯 numpy，不需要 PyWavelets。

λ函数单元 + 同伦形变大模型系统 · 小波基矩阵生成层
"""

from __future__ import annotations
import numpy as np
from typing import Literal


# ══════════════════════════════════════════════════════════════════════
# Haar 小波基（纯 numpy 实现）
# ══════════════════════════════════════════════════════════════════════
def haar_basis_1d(n: int) -> np.ndarray:
    """
    生成 n×n 的归一化 Haar 小波变换矩阵 H，使得 H @ x = Haar变换系数。
    n 必须是 2 的幂次。
    返回 (n, n) float32 矩阵，行正交归一化。
    """
    if n == 1:
        return np.ones((1, 1), dtype=np.float32)

    H = np.zeros((n, n), dtype=np.float32)
    # 尺度函数（低频）
    H[0, :] = 1.0 / np.sqrt(n)

    level = 1
    row   = 1
    step  = n
    while step > 1:
        step //= 2
        for pos in range(0, n, step * 2):
            H[row, pos:pos+step]        =  1.0 / np.sqrt(step * 2)
            H[row, pos+step:pos+step*2] = -1.0 / np.sqrt(step * 2)
            row += 1
            if row >= n:
                break
        level += 1
    return H


def dct_basis_1d(n: int) -> np.ndarray:
    """
    生成 n×n 的 DCT-II 变换矩阵（归一化）。
    不需要任何外部库。
    """
    k = np.arange(n, dtype=np.float32)
    D = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n)).astype(np.float32)
    # 归一化
    D[:, 0]  *= np.sqrt(1.0 / n)
    D[:, 1:] *= np.sqrt(2.0 / n)
    return D.T   # (n, n)，行为基向量


# ══════════════════════════════════════════════════════════════════════
# 小波基矩阵构造器
# ══════════════════════════════════════════════════════════════════════
class WaveletBasisBuilder:
    """
    构造小波/DCT 基的 W0/W1。

    W0：全频率基（Haar 或 DCT，完整保留所有频率分量）
    W1：低频截断基（只保留低频 low_k 列，高频列用近零噪声填充）

    用法：
        builder = WaveletBasisBuilder(origin_dim=256, compact_dim=64)
        W0, W1 = builder.build_weights()
    """

    def __init__(
        self,
        origin_dim:  int,
        compact_dim: int,
        basis_type:  Literal["haar", "dct"] = "dct",
        low_k:       int  = None,   # W1 保留的低频分量数（默认 compact_dim//4）
        seed:        int  = 42,
    ):
        self.origin_dim  = origin_dim
        self.compact_dim = compact_dim
        self.basis_type  = basis_type
        self.low_k       = low_k or max(4, compact_dim // 4)
        self.seed        = seed

    def _make_basis_1d(self, n: int) -> np.ndarray:
        """生成 n×n 正交变换矩阵"""
        # 向下取最近的 2 的幂次（Haar 限制）
        if self.basis_type == "haar":
            p = 1
            while p * 2 <= n:
                p *= 2
            B = haar_basis_1d(p)
            if p < n:
                # 填充至 n×n：用单位阵填充剩余行
                extra = np.eye(n - p, n, k=p, dtype=np.float32)
                B_big = np.zeros((n, n), dtype=np.float32)
                B_big[:p, :p] = B
                B_big[p:, p:] = np.eye(n - p, dtype=np.float32)
                B = B_big
            return B
        else:   # dct
            return dct_basis_1d(n)

    def build_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (W0, W1)，均为 (origin_dim, compact_dim) FP16。

        构造逻辑：
          1. 生成 origin_dim×origin_dim 的完整小波/DCT 基矩阵 B
          2. W0 = B 前 compact_dim 列（全频率投影）
          3. W1 = 前 low_k 列保留，其余列用极小噪声填充（低通）
        """
        rng = np.random.default_rng(self.seed)

        # 完整基矩阵（正交）
        B = self._make_basis_1d(self.origin_dim)   # (origin_dim, origin_dim)

        # W0：前 compact_dim 列（低频+高频混合）
        k0 = min(self.compact_dim, self.origin_dim)
        W0 = B[:, :k0].copy().astype(np.float32)
        if k0 < self.compact_dim:
            pad = rng.standard_normal((self.origin_dim, self.compact_dim - k0)).astype(np.float32)
            pad *= 1e-4
            W0 = np.hstack([W0, pad])

        # W1：只保留低频 low_k 列，高频部分近零
        k1     = min(self.low_k, self.origin_dim)
        W1_low = B[:, :k1].copy().astype(np.float32)
        noise  = rng.standard_normal((self.origin_dim, self.compact_dim - k1)).astype(np.float32)
        noise *= 1e-4
        W1 = np.hstack([W1_low, noise])

        return W0.astype(np.float16), W1.astype(np.float16)

    def lowpass_response(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        直接计算 t 形变下的频率响应（调试用）
        W(t) = (1-t)·W0 + t·W1，返回前向输出的频率分布（各分量幅值）
        """
        W0, W1 = self.build_weights()
        W_t    = (1 - t) * W0.astype(np.float32) + t * W1.astype(np.float32)
        out    = x.astype(np.float32) @ W_t
        return np.abs(out)

    def energy_ratio(self, x: np.ndarray, t: float) -> float:
        """
        返回 t 下输出能量 / t=0 下输出能量（应随 t 增大而降低，反映高频抑制）
        """
        resp_0 = self.lowpass_response(x, 0.0)
        resp_t = self.lowpass_response(x, t)
        return float(np.sum(resp_t ** 2) / (np.sum(resp_0 ** 2) + 1e-8))
