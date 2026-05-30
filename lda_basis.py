"""
lda_basis.py — 用 LDA 判别基构造 W1（有监督压缩方向）
======================================================================
核心思想：
  LDA（线性判别分析）找到"类间方差最大、类内方差最小"的投影方向。
  对比 PCA（最大总方差方向），LDA 方向对分类任务更有意义。

  W0：PCA 方向（无监督，保留最大方差）
  W1：LDA 方向（有监督，保留最大判别信息）

  当 t 从 0→1，W(t) = (1-t)·W0 + t·W1：
    t=0  无监督 PCA 表示（保留原始信号方差）
    t=1  判别性 LDA 表示（最大化类间区分度）
    中间  连续平滑从通用特征空间向判别特征空间过渡

物理意义：
  - 信号分类场景：t 越大，单元越"专注于区分类别"
  - 适合：音频类型识别、图像类别路由、文本意图分类

无监督降级：
  若无标签数据，W1 退回到稀疏/正交方向（等效于小波低频截断）。

λ函数单元 + 同伦形变大模型系统 · LDA 判别基矩阵生成层
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# LDA 基矩阵构造器
# ══════════════════════════════════════════════════════════════════════
class LDABasisBuilder:
    """
    构造 LDA 判别方向作为 W1，PCA 方向作为 W0。

    用法A（有标签数据）：
        builder = LDABasisBuilder(origin_dim=128, compact_dim=32)
        builder.fit(X, y)          # X: (n, origin_dim), y: (n,) 类别标签
        W0, W1 = builder.build_weights()

    用法B（无标签，构造合成多类数据）：
        W0, W1 = builder.build_from_synthetic(n_classes=4, n_per_class=100)
    """

    def __init__(
        self,
        origin_dim:  int,
        compact_dim: int,
        seed:        int = 42,
    ):
        self.origin_dim  = origin_dim
        self.compact_dim = compact_dim
        self.seed        = seed
        self._lda_dirs:  Optional[np.ndarray] = None
        self._pca_dirs:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDABasisBuilder":
        """
        计算 LDA 投影方向和 PCA 投影方向。
        X: (n_samples, origin_dim), y: (n_samples,) int 类别标签
        """
        X_f   = X.astype(np.float32)
        y_i   = y.astype(np.int32)
        classes = np.unique(y_i)
        n, d  = X_f.shape

        # ── PCA 方向（W0 基础）──────────────────────────────────────
        Xc = X_f - X_f.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self._pca_dirs = Vt.T.astype(np.float32)   # (d, n_components)

        # ── LDA 方向（W1 基础）──────────────────────────────────────
        # 类内散度矩阵 Sw
        mu_global = X_f.mean(axis=0)
        Sw = np.zeros((d, d), dtype=np.float32)
        Sb = np.zeros((d, d), dtype=np.float32)

        for c in classes:
            Xc_cls   = X_f[y_i == c]
            mu_c     = Xc_cls.mean(axis=0)
            diff_w   = Xc_cls - mu_c
            Sw      += diff_w.T @ diff_w

            n_c      = len(Xc_cls)
            diff_b   = (mu_c - mu_global).reshape(-1, 1)
            Sb      += n_c * (diff_b @ diff_b.T)

        # 广义特征分解 Sb·v = λ·Sw·v
        # 数值稳定：Sw 加正则化
        Sw += np.eye(d, dtype=np.float32) * 1e-4

        # 求解：转化为标准特征问题 Sw^{-1} Sb
        try:
            Sw_inv  = np.linalg.inv(Sw)
            M       = Sw_inv @ Sb
            evals, evecs = np.linalg.eigh(M)
            # 取最大特征值对应方向（类间方差最大）
            idx     = np.argsort(evals)[::-1]
            self._lda_dirs = evecs[:, idx].astype(np.float32)
        except np.linalg.LinAlgError:
            # 降级到 PCA
            print("  [LDA] Sw 奇异，降级为 PCA 方向")
            self._lda_dirs = self._pca_dirs.copy()

        return self

    def build_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (W0, W1)：
          W0: PCA 方向（前 compact_dim 个，列归一化）
          W1: LDA 方向（前 compact_dim 个，列归一化）
        """
        assert self._pca_dirs is not None, "请先调用 fit()"
        rng = np.random.default_rng(self.seed)

        def _take(dirs, k, dim):
            k_real = min(k, dirs.shape[1])
            W = dirs[:, :k_real].copy()
            if k_real < dim:
                noise = rng.standard_normal((dirs.shape[0], dim - k_real)).astype(np.float32)
                noise *= 1e-4
                W = np.hstack([W, noise])
            else:
                W = W[:, :dim]
            # 列归一化
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
            return W.astype(np.float16)

        W0 = _take(self._pca_dirs, self.compact_dim, self.compact_dim)
        W1 = _take(self._lda_dirs, self.compact_dim, self.compact_dim)
        return W0, W1

    def build_from_synthetic(
        self,
        n_classes:    int   = 4,
        n_per_class:  int   = 100,
        signal_scale: float = 2.0,
        noise_scale:  float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        构造合成多类数据后做 LDA。
        每类围绕一个随机中心点，高斯分布。
        """
        rng  = np.random.default_rng(self.seed)
        X_list, y_list = [], []
        # 类别中心（彼此有较大间距）
        centers = rng.standard_normal((n_classes, self.origin_dim)).astype(np.float32)
        centers *= signal_scale
        for c, mu in enumerate(centers):
            Xi = rng.standard_normal((n_per_class, self.origin_dim)).astype(np.float32)
            Xi = Xi * noise_scale + mu
            X_list.append(Xi)
            y_list.append(np.full(n_per_class, c, dtype=np.int32))
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        self.fit(X, y)
        return self.build_weights()

    def class_separability(self, X: np.ndarray, y: np.ndarray,
                            W: np.ndarray) -> float:
        """
        计算投影后的类间/类内距离比（越大=判别性越强）
        """
        X_proj = X.astype(np.float32) @ W.astype(np.float32)
        classes = np.unique(y.astype(np.int32))
        mu_all  = X_proj.mean(axis=0)
        sb, sw  = 0.0, 0.0
        for c in classes:
            Xc = X_proj[y == c]
            mu_c = Xc.mean(axis=0)
            sb  += len(Xc) * float(np.sum((mu_c - mu_all) ** 2))
            sw  += float(np.sum((Xc - mu_c) ** 2))
        return sb / (sw + 1e-8)
