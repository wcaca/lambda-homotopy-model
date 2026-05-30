"""
int8_lambda_unit.py — INT8格式λ单元读写类（增量新增，原有文件零修改）
继承 LambdaUnit，覆盖 load() 支持 version=3（INT8），
forward() 使用 INT8 空间插值，与原有接口完全兼容。

λ函数单元 + 同伦形变大模型系统 · INT8单元层
"""

from __future__ import annotations
import os, sys, struct
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_utils import LambdaUnit, ENDIAN, HEAD_LEN, FP16_SIZE, FINGER_DIM
from int8_quantizer import (
    VERSION_INT8, INT8_SIZE, SCALE_SIZE,
    dequantize_int8, int8_interpolate_and_forward,
)


class INT8LambdaUnit(LambdaUnit):
    """
    INT8 格式λ单元
    load()    : 自动检测 version，INT8 路径读 int8 权重 + scale
    forward() : 优先用 INT8 空间插值（快、省内存），FP16 路径保留
    save()    : 写回 INT8 格式文件
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        # INT8 专属字段
        self.w0_int8:   np.ndarray = None
        self.w1_int8:   np.ndarray = None
        self.scale_w0:  float      = 1.0
        self.scale_w1:  float      = 1.0
        self.is_int8:   bool       = False

    def load(self):
        """加载 *.bin 文件，自动识别 FP16(v1) / INT8(v3) 两种格式"""
        with open(self.file_path, "rb") as f:
            # ── 区块1：头部 ──────────────────────────────────────────
            head = f.read(HEAD_LEN)
            self.unit_id      = struct.unpack_from(f"{ENDIAN}32s", head, 0)[0].decode("utf-8", errors="replace").strip("\x00")
            self.func_type    = struct.unpack_from(f"{ENDIAN}B",   head, 32)[0]
            self.dim_skeleton = struct.unpack_from(f"{ENDIAN}H",   head, 33)[0]
            self.dim_finger   = struct.unpack_from(f"{ENDIAN}B",   head, 35)[0]
            self.dim_origin   = struct.unpack_from(f"{ENDIAN}H",   head, 36)[0]
            self.version      = struct.unpack_from(f"{ENDIAN}H",   head, 38)[0]

            # ── 区块2：控制规则区 ────────────────────────────────────
            rule_buf = b""
            while True:
                b = f.read(1)
                if not b or b == b"\n":
                    break
                rule_buf += b
            self.rule_text = rule_buf.decode("utf-8", errors="replace")
            for seg in self.rule_text.split(";"):
                if seg.startswith("Enable="):
                    self.enable = int(seg.split("=")[1]) == 1

            size_w = self.dim_origin * self.dim_skeleton

            # ── 区块3：权重区（根据 version 选择格式）──────────────
            if self.version == VERSION_INT8:
                # INT8 格式：W0_int8 + scale_W0(FP32) + W1_int8 + scale_W1(FP32) + Bias(FP16)
                self.is_int8 = True
                w0_buf       = f.read(size_w * INT8_SIZE)
                self.scale_w0 = struct.unpack(f"{ENDIAN}f", f.read(SCALE_SIZE))[0]
                w1_buf        = f.read(size_w * INT8_SIZE)
                self.scale_w1 = struct.unpack(f"{ENDIAN}f", f.read(SCALE_SIZE))[0]
                bias_buf      = f.read(self.dim_skeleton * FP16_SIZE)

                self.w0_int8 = np.frombuffer(w0_buf, dtype=np.int8).reshape(self.dim_origin, self.dim_skeleton)
                self.w1_int8 = np.frombuffer(w1_buf, dtype=np.int8).reshape(self.dim_origin, self.dim_skeleton)
                self.bias    = np.frombuffer(bias_buf, dtype=np.float16)
                # 同时提供反量化版本（供需要 FP 权重的工具使用）
                self.w0 = dequantize_int8(self.w0_int8, self.scale_w0).reshape(
                    self.dim_origin, self.dim_skeleton).astype(np.float16)
                self.w1 = dequantize_int8(self.w1_int8, self.scale_w1).reshape(
                    self.dim_origin, self.dim_skeleton).astype(np.float16)
            else:
                # 原有 FP16 格式（version=1），走父类权重读取逻辑
                self.is_int8  = False
                w0_buf        = f.read(size_w * FP16_SIZE)
                w1_buf        = f.read(size_w * FP16_SIZE)
                bias_buf      = f.read(self.dim_skeleton * FP16_SIZE)
                self.w0       = np.frombuffer(w0_buf,   dtype=np.float16).reshape(self.dim_origin, self.dim_skeleton)
                self.w1       = np.frombuffer(w1_buf,   dtype=np.float16).reshape(self.dim_origin, self.dim_skeleton)
                self.bias     = np.frombuffer(bias_buf, dtype=np.float16)

            # ── 区块4：指纹区 ────────────────────────────────────────
            finger_buf  = f.read(FINGER_DIM * FP16_SIZE)
            self.finger = np.frombuffer(finger_buf, dtype=np.float16)

            # ── 区块5：接口JSON ──────────────────────────────────────
            import json
            json_buf = b""
            while True:
                b = f.read(1)
                if not b or b == b"\x00":
                    break
                json_buf += b
            self.port_config = json.loads(json_buf.decode("utf-8")) if json_buf else {}

    def forward(self, x: np.ndarray, global_t: float) -> np.ndarray:
        """
        前向计算：
          INT8 格式 → 使用 INT8 空间插值（省内存、快）
          FP16 格式 → 使用父类原有 FP16 插值
        """
        if not self.enable:
            return x

        if self.is_int8 and self.w0_int8 is not None:
            return int8_interpolate_and_forward(
                x, self.w0_int8, self.scale_w0,
                self.w1_int8,   self.scale_w1,
                self.bias, global_t,
            )
        else:
            # 降级到父类 FP16 前向
            return super().forward(x, global_t)

    def storage_stats(self) -> dict:
        """返回存储统计信息"""
        size_bytes = os.path.getsize(self.file_path)
        w_elem     = self.dim_origin * self.dim_skeleton
        if self.is_int8:
            weight_bytes = w_elem * INT8_SIZE * 2 + SCALE_SIZE * 2
        else:
            weight_bytes = w_elem * FP16_SIZE * 2
        return {
            "unit_id":     self.unit_id,
            "format":      "INT8" if self.is_int8 else "FP16",
            "file_kb":     round(size_bytes / 1024, 1),
            "weight_kb":   round(weight_bytes / 1024, 1),
            "version":     self.version,
        }
