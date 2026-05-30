"""
int8_quantizer.py — INT8 权重量化工具（增量新增，原有文件零修改）
======================================================================
功能：
  将 λ单元的 W0/W1 从 FP16（2字节/参数）量化为 INT8（1字节/参数），
  存储体积再减少 50%，同时维持同伦形变的数学正确性。

量化方案：对称线性量化（Symmetric Quantization）
  scale  = max(|W|) / 127
  W_int8 = round(W / scale).clip(-127, 127)
  W_fp32 = W_int8 * scale                    ← 反量化

每个矩阵独立计算 scale（per-tensor），不做 per-channel
（per-channel 精度更高但实现复杂，留作后续扩展）

存储格式（区块3 INT8 变体）：
  原有 FP16：W0(FP16) + W1(FP16) + Bias(FP16)
  INT8 格式：W0(INT8) + scale_W0(FP32) + W1(INT8) + scale_W1(FP32) + Bias(FP16)
  头部 Version 字段：2 → 3（标记为 INT8 格式，保持向后兼容）

压缩率：
  FP16 权重：origin_dim × skeleton_dim × 2字节 × 2矩阵
  INT8 权重：origin_dim × skeleton_dim × 1字节 × 2矩阵 + 8字节（scale）
  理论压缩率 ≈ 49%（接近 2× 体积缩减）

λ函数单元 + 同伦形变大模型系统 · INT8量化层
"""

from __future__ import annotations
import os, sys, struct
import numpy as np
from typing import Optional, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_utils import ENDIAN, HEAD_LEN, FP16_SIZE, FINGER_DIM

# ── INT8 格式版本号 ──────────────────────────────────────────────────────
VERSION_FP16 = 1
VERSION_INT8 = 3
INT8_SIZE    = 1
SCALE_SIZE   = 4    # FP32 scale，4字节


# ══════════════════════════════════════════════════════════════════════
# 量化 / 反量化核心
# ══════════════════════════════════════════════════════════════════════
def quantize_int8(
    w: np.ndarray,
    clip_ratio: float = 0.99,
) -> Tuple[np.ndarray, float]:
    """
    FP32/FP16 → INT8 对称量化
    :param clip_ratio: 用百分位数截断异常值（默认99分位，减少量化误差）
    :return: (w_int8, scale)
    """
    w_f   = w.astype(np.float32).flatten()
    # 用百分位数截断，避免极端值拉大 scale 导致精度损失
    abs_max = float(np.percentile(np.abs(w_f), clip_ratio * 100))
    abs_max = max(abs_max, 1e-8)   # 防止全零矩阵

    scale   = abs_max / 127.0
    w_int8  = np.clip(np.round(w_f / scale), -127, 127).astype(np.int8)
    return w_int8.reshape(w.shape), scale


def dequantize_int8(w_int8: np.ndarray, scale: float) -> np.ndarray:
    """INT8 → FP32 反量化"""
    return (w_int8.astype(np.float32) * scale)


def quantization_error(w_orig: np.ndarray, w_int8: np.ndarray, scale: float) -> dict:
    """量化误差统计"""
    w_orig_f  = w_orig.astype(np.float32).flatten()
    w_dequant = dequantize_int8(w_int8, scale).flatten()
    n = min(len(w_orig_f), len(w_dequant))
    diff      = w_orig_f[:n] - w_dequant[:n]
    rel_err   = float(np.linalg.norm(diff) / (np.linalg.norm(w_orig_f[:n]) + 1e-8))
    return {
        "rel_error":  round(rel_err, 6),
        "max_abs_err":round(float(np.max(np.abs(diff))), 6),
        "mean_abs_err":round(float(np.mean(np.abs(diff))), 6),
        "scale":      round(scale, 8),
    }


# ══════════════════════════════════════════════════════════════════════
# INT8 λ单元文件读写
# ══════════════════════════════════════════════════════════════════════
class INT8Quantizer:
    """
    λ单元 FP16 → INT8 量化器
    输入：FP16 格式的 *.bin 文件路径（或 LambdaUnit 实例）
    输出：INT8 格式的 *.bin 文件（version=3，区块3格式变更）
    """

    def quantize_file(
        self,
        src_path:   str,
        dst_path:   str,
        clip_ratio: float = 0.99,
    ) -> dict:
        """
        将单个 FP16 λ单元文件量化为 INT8 格式
        :return: 量化统计信息（压缩率、误差等）
        """
        # 1. 读取原始文件
        from base_utils import LambdaUnit
        unit = LambdaUnit(src_path)
        unit.load()

        # 2. 量化 W0 / W1
        w0_int8, scale_w0 = quantize_int8(unit.w0, clip_ratio)
        w1_int8, scale_w1 = quantize_int8(unit.w1, clip_ratio)

        err_w0 = quantization_error(unit.w0, w0_int8, scale_w0)
        err_w1 = quantization_error(unit.w1, w1_int8, scale_w1)

        # 3. 写出 INT8 格式文件
        self._write_int8_unit(dst_path, unit, w0_int8, scale_w0, w1_int8, scale_w1)

        # 4. 统计
        src_size = os.path.getsize(src_path)
        dst_size = os.path.getsize(dst_path)
        stats = {
            "src_path":     src_path,
            "dst_path":     dst_path,
            "src_size_kb":  round(src_size / 1024, 1),
            "dst_size_kb":  round(dst_size / 1024, 1),
            "compress_ratio": round(dst_size / max(1, src_size), 4),
            "size_reduction": f"{(1 - dst_size/src_size)*100:.1f}%",
            "w0_error":     err_w0,
            "w1_error":     err_w1,
        }
        return stats

    def quantize_group(
        self,
        group_dir:   str,
        output_dir:  str,
        clip_ratio:  float = 0.99,
        suffix:      str   = "_int8",
    ) -> list[dict]:
        """批量量化一个组目录下的所有 FP16 λ单元"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for fname in sorted(os.listdir(group_dir)):
            if not fname.endswith(".bin"):
                continue
            src = os.path.join(group_dir, fname)
            dst = os.path.join(output_dir, fname.replace(".bin", f"{suffix}.bin"))
            try:
                stat = self.quantize_file(src, dst, clip_ratio)
                results.append(stat)
                print(f"    ✓ {fname} → {stat['src_size_kb']}KB → "
                      f"{stat['dst_size_kb']}KB  "
                      f"压缩={stat['size_reduction']}  "
                      f"误差={stat['w0_error']['rel_error']:.4f}")
            except Exception as e:
                print(f"    ✗ {fname}: {e}")
        return results

    # ── 内部写出逻辑 ─────────────────────────────────────────────────
    @staticmethod
    def _write_int8_unit(
        path:      str,
        unit,                    # LambdaUnit 实例
        w0_int8:   np.ndarray,   # INT8
        scale_w0:  float,
        w1_int8:   np.ndarray,   # INT8
        scale_w1:  float,
    ):
        with open(path, "wb") as f:
            # ── 区块1：头部（version 改为 3=INT8）────────────────────
            uid_bytes = unit.unit_id.encode("utf-8").ljust(32, b"\x00")[:32]
            head = (
                uid_bytes
                + struct.pack(f"{ENDIAN}B", unit.func_type)
                + struct.pack(f"{ENDIAN}H", unit.dim_skeleton)
                + struct.pack(f"{ENDIAN}B", unit.dim_finger if unit.dim_finger else FINGER_DIM)
                + struct.pack(f"{ENDIAN}H", unit.dim_origin)
                + struct.pack(f"{ENDIAN}H", VERSION_INT8)   # ← 版本号=3
                + b"\x00" * 24
            )
            assert len(head) == HEAD_LEN
            f.write(head)

            # ── 区块2：控制规则区（不变）────────────────────────────
            rule = (
                f"DimRule=d(t)={unit.dim_origin}*(1-t)+{unit.dim_skeleton}*t;"
                f"WeightRule=W(t)=(1-t)*W0+t*W1;"
                f"Enable=1;"
                f"Quantization=INT8"
            )
            f.write(rule.encode("utf-8") + b"\n")

            # ── 区块3：INT8 权重区 ───────────────────────────────────
            # 格式：W0_int8 + scale_W0(FP32) + W1_int8 + scale_W1(FP32) + Bias(FP16)
            f.write(w0_int8.flatten().astype(np.int8).tobytes())
            f.write(struct.pack(f"{ENDIAN}f", scale_w0))    # FP32 scale
            f.write(w1_int8.flatten().astype(np.int8).tobytes())
            f.write(struct.pack(f"{ENDIAN}f", scale_w1))    # FP32 scale
            f.write(unit.bias.astype(np.float16).tobytes()) # Bias 保持 FP16

            # ── 区块4：指纹区（不变）────────────────────────────────
            f.write(unit.finger.astype(np.float16).tobytes())

            # ── 区块5：接口JSON（不变）──────────────────────────────
            import json
            port_cfg = json.dumps({
                "InPort": 1, "OutPort": 1, "MergeRule": "sum",
                "Quantization": "INT8",
            })
            f.write(port_cfg.encode("utf-8") + b"\x00")


# ══════════════════════════════════════════════════════════════════════
# INT8 格式同伦形变插值
# ══════════════════════════════════════════════════════════════════════
def int8_interpolate_and_forward(
    x:        np.ndarray,
    w0_int8:  np.ndarray,
    scale_w0: float,
    w1_int8:  np.ndarray,
    scale_w1: float,
    bias:     np.ndarray,
    t:        float,
) -> np.ndarray:
    """
    INT8 权重的同伦形变前向计算
    两种策略：
      A) 先反量化再插值（精度优先）：W(t) = (1-t)·dequant(W0) + t·dequant(W1)
      B) 先插值 scale 再反量化（速度优先）：
           scale_t = (1-t)·scale_w0 + t·scale_w1
           W_int8_t = round((1-t)·W0_int8 + t·W1_int8)
           W(t) = W_int8_t · scale_t

    默认使用策略 B（避免反量化到 FP32 的全量内存）
    """
    # 策略 B：INT8 空间插值
    w_interp = (1 - t) * w0_int8.astype(np.float32) + t * w1_int8.astype(np.float32)
    scale_t  = (1 - t) * scale_w0 + t * scale_w1
    w_fp32   = np.clip(np.round(w_interp), -127, 127).astype(np.int8).astype(np.float32) * scale_t

    x_f = x.astype(np.float32)
    if x_f.shape[-1] != w_fp32.shape[0]:
        if x_f.shape[-1] < w_fp32.shape[0]:
            pad  = np.zeros(w_fp32.shape[0] - x_f.shape[-1], dtype=np.float32)
            x_f  = np.concatenate([x_f, pad])
        else:
            x_f = x_f[:w_fp32.shape[0]]

    out = np.dot(x_f, w_fp32) + bias.astype(np.float32)
    return out.astype(np.float16)
