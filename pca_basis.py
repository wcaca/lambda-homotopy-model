"""
pca_basis.py — 用 PCA 构造有意义的 W0/W1
======================================================================
核心思想：
  W0 = 原始高维保留矩阵（前 k 个主成分方向，保留最大方差）
  W1 = 极致压缩矩阵（前 compact_k 个主成分，丢弃次要方差）

  当 t 从 0→1，W(t) = (1-t)·W0 + t·W1：
    t=0  保留所有主成分信息（原始精度）
    t=1  只保留最主要成分（极致压缩）
    中间  连续平滑降维

物理意义：
  - 比随机矩阵：压缩后的信息损失可量化（重建误差）
  - W0·x 保留原始数据 origin_dim 维最重要的 compact_dim 方向
  - W1·x 只保留最显著的 compact_k 方向（compact_k << compact_dim）
  - t 形变 = "压缩强度"的连续调节旋钮

λ函数单元 + 同伦形变大模型系统 · PCA 基矩阵生成层
"""

from __future__ import annotations
import os, sys
import numpy as np
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


# ══════════════════════════════════════════════════════════════════════
# PCA 基矩阵构造器
# ══════════════════════════════════════════════════════════════════════
class PCABasisBuilder:
    """
    从输入数据（或协方差矩阵）构造 PCA 投影矩阵作为 W0/W1。

    用法A（有数据）：
        builder = PCABasisBuilder(origin_dim=4096, compact_dim=512)
        builder.fit(X)           # X: (n_samples, origin_dim)
        W0, W1 = builder.build_weights()

    用法B（无数据，构造模拟数据）：
        builder = PCABasisBuilder(origin_dim=128, compact_dim=32)
        W0, W1 = builder.build_from_synthetic(n_samples=1000)
    """

    def __init__(
        self,
        origin_dim:  int,
        compact_dim: int,
        compact_k:   Optional[int] = None,  # W1 保留的主成分数（默认 compact_dim//4）
        seed:        int = 42,
    ):
        self.origin_dim  = origin_dim
        self.compact_dim = compact_dim
        self.compact_k   = compact_k or max(8, compact_dim // 4)
        self.seed        = seed
        self._components: Optional[np.ndarray] = None  # (origin_dim, origin_dim) 主成分矩阵
        self._explained_var: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCABasisBuilder":
        """
        对数据矩阵 X(n_samples, origin_dim) 做 PCA，计算主成分方向。
        纯 numpy SVD，无需 sklearn。
        """
        assert X.shape[1] == self.origin_dim, \
            f"数据列数 {X.shape[1]} != origin_dim {self.origin_dim}"

        X_f = X.astype(np.float32)
        X_centered = X_f - X_f.mean(axis=0, keepdims=True)

        # SVD（经济模式）
        n = X_centered.shape[0]
        k = min(n, self.origin_dim, self.compact_dim + 16)   # 只算需要的奇异值
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Vt 的行 = 主成分方向（降序方差）
        self._components   = Vt.T                           # (origin_dim, n_components)
        self._explained_var = s[:min(len(s), self.compact_dim + 16)] ** 2 / (n - 1)
        return self

    def fit_from_covariance(self, cov: np.ndarray) -> "PCABasisBuilder":
        """从协方差矩阵直接做特征分解（适合 origin_dim 较小时）"""
        evals, evecs = np.linalg.eigh(cov.astype(np.float32))
        idx = np.argsort(evals)[::-1]
        self._components   = evecs[:, idx]
        self._explained_var = evals[idx]
        return self

    def build_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (W0, W1)，均为 (origin_dim, compact_dim) FP16 矩阵。
        W0: 前 compact_dim 个主成分（保留最大方差）
        W1: 前 compact_k  个主成分填充至 compact_dim（其余列接近零）
        """
        assert self._components is not None, "请先调用 fit() 或 fit_from_covariance()"

        # W0：前 compact_dim 个主成分方向（行归一化）
        k0 = min(self.compact_dim, self._components.shape[1])
        W0 = self._components[:, :k0].copy()
        if k0 < self.compact_dim:
            pad = np.zeros((self.origin_dim, self.compact_dim - k0), dtype=np.float32)
            W0  = np.hstack([W0, pad])

        # W1：只保留前 compact_k 个主成分，其余列用衰减噪声填充（接近零）
        k1     = min(self.compact_k, self._components.shape[1])
        W1_top = self._components[:, :k1].copy()
        rng    = np.random.default_rng(self.seed)
        noise  = rng.standard_normal((self.origin_dim, self.compact_dim - k1)).astype(np.float32)
        noise *= 1e-4   # 极小噪声，保证 W1 列满秩，避免退化
        W1 = np.hstack([W1_top, noise])

        # 列归一化（使每列范数≈1）
        W0 = W0 / (np.linalg.norm(W0, axis=0, keepdims=True) + 1e-8)
        W1 = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-8)

        return W0.astype(np.float16), W1.astype(np.float16)

    def build_from_synthetic(
        self,
        n_samples:       int = 500,
        n_signal_dims:   int = 64,   # 真实信号维度（低秩结构）
        noise_scale:     float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        无真实数据时，构造低秩结构合成数据后做 PCA。
        合成数据 = 低维信号（结构方差）+ 高斯噪声（均匀方差）
        目的：确保 W0/W1 的主成分有实际的"信号方向"
        """
        rng = np.random.default_rng(self.seed)

        # 低维信号基（n_signal_dims 个有意义方向）
        signal_basis  = rng.standard_normal((self.origin_dim, n_signal_dims)).astype(np.float32)
        signal_basis /= np.linalg.norm(signal_basis, axis=0, keepdims=True) + 1e-8
        signal_coeff  = rng.standard_normal((n_samples, n_signal_dims)).astype(np.float32)
        signal_coeff *= np.linspace(3.0, 0.5, n_signal_dims)  # 方差递减

        # 合成数据 = 结构信号 + 噪声
        X_signal = signal_coeff @ signal_basis.T
        X_noise  = rng.standard_normal((n_samples, self.origin_dim)).astype(np.float32) * noise_scale
        X = X_signal + X_noise

        self.fit(X)
        return self.build_weights()

    def reconstruction_error(self, X_test: np.ndarray, W: np.ndarray) -> float:
        """计算用 W 投影后的重建误差（越小=保留信息越多）"""
        X_f      = X_test.astype(np.float32)
        W_f      = W.astype(np.float32)
        proj     = X_f @ W_f               # (n, compact_dim)
        recon    = proj @ W_f.T            # (n, origin_dim)
        err      = np.linalg.norm(X_f - recon) / (np.linalg.norm(X_f) + 1e-8)
        return float(err)

    def explained_variance_ratio(self) -> np.ndarray:
        """返回各主成分解释方差比例"""
        if self._explained_var is None:
            return np.array([])
        total = self._explained_var.sum() + 1e-8
        return (self._explained_var / total).astype(np.float32)
