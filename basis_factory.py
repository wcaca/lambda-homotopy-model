"""
basis_factory.py — 统一 W0/W1 生成入口（增量新增，原有文件零修改）
======================================================================
一行代码为任意 λ单元写入有意义的 W0/W1，替换随机初始化。

支持三种基：
  pca      — 主成分基（通用降维，无监督）
  wavelet  — 小波/DCT 基（信号滤波，纯数学）
  lda      — 判别基（有监督分类，需要标签）
  random   — 原始随机基（保留兼容，不推荐）

用法：
  # 直接生成并写入 *.bin 文件
  from basis_factory import BasisFactory
  factory = BasisFactory(origin_dim=4096, compact_dim=512)
  factory.apply_to_unit("./model_root/attn_group/lambda_001_0.bin", basis="pca")

  # 批量更新整个组
  factory.apply_to_group("./model_root/attn_group", basis="wavelet")

  # 仅获取矩阵，不写文件
  W0, W1 = factory.build(basis="lda")

λ函数单元 + 同伦形变大模型系统 · 基矩阵统一工厂
"""

from __future__ import annotations
import os, sys, struct
import numpy as np
from typing import Literal, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pca_basis     import PCABasisBuilder
from wavelet_basis import WaveletBasisBuilder
from lda_basis     import LDABasisBuilder
from base_utils    import ENDIAN, HEAD_LEN, FP16_SIZE, FINGER_DIM, LambdaUnit

BasisType = Literal["pca", "wavelet", "lda", "random"]


class BasisFactory:
    """
    统一的 W0/W1 基矩阵工厂。
    """

    def __init__(
        self,
        origin_dim:  int = 4096,
        compact_dim: int = 512,
        seed:        int = 42,
        # PCA 参数
        pca_n_samples:    int   = 500,
        pca_n_signal_dims: int  = 64,
        # 小波参数
        wavelet_type:     str   = "dct",
        wavelet_low_k:    Optional[int] = None,
        # LDA 参数
        lda_n_classes:    int   = 4,
        lda_n_per_class:  int   = 80,
    ):
        self.origin_dim  = origin_dim
        self.compact_dim = compact_dim
        self.seed        = seed
        self._pca_cfg    = dict(n_samples=pca_n_samples, n_signal_dims=pca_n_signal_dims)
        self._wav_cfg    = dict(basis_type=wavelet_type, low_k=wavelet_low_k)
        self._lda_cfg    = dict(n_classes=lda_n_classes, n_per_class=lda_n_per_class)
        # 缓存（同配置多次调用时复用）
        self._cache: dict[str, tuple] = {}

    def build(self, basis: BasisType = "pca") -> tuple[np.ndarray, np.ndarray]:
        """生成 (W0, W1) 矩阵，均为 (origin_dim, compact_dim) FP16"""
        if basis in self._cache:
            return self._cache[basis]

        if basis == "pca":
            b = PCABasisBuilder(self.origin_dim, self.compact_dim, seed=self.seed)
            W0, W1 = b.build_from_synthetic(**self._pca_cfg)

        elif basis == "wavelet":
            b = WaveletBasisBuilder(
                self.origin_dim, self.compact_dim,
                seed=self.seed, **self._wav_cfg,
            )
            W0, W1 = b.build_weights()

        elif basis == "lda":
            b = LDABasisBuilder(self.origin_dim, self.compact_dim, seed=self.seed)
            W0, W1 = b.build_from_synthetic(**self._lda_cfg)

        elif basis == "random":
            rng = np.random.default_rng(self.seed)
            W0  = rng.standard_normal((self.origin_dim, self.compact_dim)).astype(np.float16)
            W1  = rng.standard_normal((self.origin_dim, self.compact_dim)).astype(np.float16) * 0.4
        else:
            raise ValueError(f"未知 basis: {basis}，可选: pca/wavelet/lda/random")

        self._cache[basis] = (W0, W1)
        return W0, W1

    # ── 写入单个 *.bin 文件 ────────────────────────────────────────────
    def apply_to_unit(
        self,
        bin_path: str,
        basis:    BasisType = "pca",
        dry_run:  bool      = False,
    ) -> dict:
        """
        将有意义的 W0/W1 写入指定 *.bin 文件（原位替换权重区）。
        其余区块（头部/规则/指纹/接口）完全不变。
        返回写入统计。
        """
        unit = LambdaUnit(bin_path)
        unit.load()

        # 如果工厂维度和单元维度不一致，临时构造对应维度的矩阵
        if unit.dim_origin != self.origin_dim or unit.dim_skeleton != self.compact_dim:
            tmp = BasisFactory(
                origin_dim  = unit.dim_origin,
                compact_dim = unit.dim_skeleton,
                seed        = self.seed,
                **{k: v for k, v in {
                    "pca_n_samples": self._pca_cfg.get("n_samples", 200),
                    "pca_n_signal_dims": min(self._pca_cfg.get("n_signal_dims", 32),
                                             unit.dim_skeleton // 2),
                    "wavelet_low_k": self._wav_cfg.get("low_k"),
                    "lda_n_classes": self._lda_cfg.get("n_classes", 4),
                    "lda_n_per_class": self._lda_cfg.get("n_per_class", 50),
                }.items() if not (k.startswith("wavelet") and k != "wavelet_low_k")}
            )
            W0, W1 = tmp.build(basis)
        else:
            W0, W1 = self.build(basis)

        if dry_run:
            return {"unit_id": unit.unit_id, "basis": basis,
                    "dry_run": True, "W0_norm": float(np.linalg.norm(W0))}

        # 重建整个文件（最安全的方式，避免偏移计算复杂性）
        self._rewrite_unit_weights(bin_path, unit, W0, W1)

        return {
            "unit_id":    unit.unit_id,
            "basis":      basis,
            "W0_norm":    round(float(np.linalg.norm(W0)), 4),
            "W1_norm":    round(float(np.linalg.norm(W1)), 4),
            "dim":        f"{unit.dim_origin}→{unit.dim_skeleton}",
        }

    # ── 批量更新整个组目录 ────────────────────────────────────────────
    def apply_to_group(
        self,
        group_dir: str,
        basis:     BasisType = "pca",
        dry_run:   bool      = False,
    ) -> list[dict]:
        """批量将有意义的 W0/W1 写入组目录下所有 *.bin 文件"""
        results = []
        for fname in sorted(os.listdir(group_dir)):
            if not fname.endswith(".bin"):
                continue
            path = os.path.join(group_dir, fname)
            try:
                stat = self.apply_to_unit(path, basis=basis, dry_run=dry_run)
                results.append(stat)
                tag = "[dry]" if dry_run else "✓"
                print(f"  {tag} {fname}  basis={basis}  "
                      f"W0_norm={stat['W0_norm']:.4f}  W1_norm={stat['W1_norm']:.4f}"
                      if not dry_run else f"  [dry] {fname}")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")
        return results

    # ── 内部：重写 *.bin 权重区 ───────────────────────────────────────
    @staticmethod
    def _rewrite_unit_weights(
        path:  str,
        unit:  LambdaUnit,
        W0:    np.ndarray,
        W1:    np.ndarray,
    ):
        """读取原文件，仅替换权重区（区块3），写回完整文件"""
        import json
        with open(path, "rb") as f:
            raw = f.read()

        # 重建完整文件
        out = b""

        # 区块1：头部（前 HEAD_LEN 字节不变）
        out += raw[:HEAD_LEN]

        # 区块2：控制规则区（找到第一个 \n）
        pos = HEAD_LEN
        rule_end = raw.index(b"\n", pos) + 1
        out += raw[pos:rule_end]
        pos = rule_end

        # 区块3：权重区（origin_dim × compact_dim × 2 × FP16_SIZE + bias）
        size_w   = unit.dim_origin * unit.dim_skeleton * FP16_SIZE
        bias_len = unit.dim_skeleton * FP16_SIZE
        old_weight_end = pos + size_w * 2 + bias_len   # W0+W1+bias

        # 写入新权重
        out += W0.astype(np.float16).flatten().tobytes()
        out += W1.astype(np.float16).flatten().tobytes()
        out += unit.bias.astype(np.float16).tobytes()

        # 区块4~末：指纹+接口+模态扩展（原样保留）
        out += raw[old_weight_end:]

        with open(path, "wb") as f:
            f.write(out)

    # ── 质量对比 ────────────────────────────────────────────────────────
    def compare_bases(
        self,
        test_dim: int    = 64,
        n_test:   int    = 50,
    ) -> dict:
        """
        在标准测试数据上对比四种基的性能：
        - W0 重建误差（越小=保留信息越多）
        - W0→W1 信息保留率（越低=W1 压缩越激进）
        - W0 和 W1 余弦相似度（越低=形变幅度越大）
        """
        rng  = np.random.default_rng(self.seed + 100)
        X    = rng.standard_normal((n_test, self.origin_dim)).astype(np.float32)

        report = {}
        for basis in ["random", "pca", "wavelet", "lda"]:
            try:
                W0, W1 = self.build(basis)
                W0f, W1f = W0.astype(np.float32), W1.astype(np.float32)

                # 重建误差
                proj0 = X @ W0f
                recon = proj0 @ W0f.T
                err0  = float(np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-8))

                proj1 = X @ W1f
                recon1 = proj1 @ W1f.T
                err1  = float(np.linalg.norm(X - recon1) / (np.linalg.norm(X) + 1e-8))

                # W0/W1 余弦相似度（列方向平均）
                n_cols = min(W0f.shape[1], 16)
                sims = [
                    float(np.dot(W0f[:, i], W1f[:, i]) /
                          (np.linalg.norm(W0f[:, i]) * np.linalg.norm(W1f[:, i]) + 1e-8))
                    for i in range(n_cols)
                ]
                mean_sim = float(np.mean(sims))

                report[basis] = {
                    "W0_recon_err":   round(err0, 4),
                    "W1_recon_err":   round(err1, 4),
                    "W0_W1_cos_sim":  round(mean_sim, 4),
                    "compress_delta": round(err1 - err0, 4),
                }
            except Exception as e:
                report[basis] = {"error": str(e)}
        return report
