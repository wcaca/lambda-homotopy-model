"""
base_utils.py — 公共基础工具类
功能：二进制读写、同伦形变计算、配置解析、向量运算
λ函数单元 + 同伦形变大模型系统 · 核心基础层
"""

import os
import struct
import json
import configparser
import numpy as np

# ===================== 全局常量（统一配置）=====================
ENDIAN = "<"        # 小端序，跨平台统一
FP16_SIZE = 2
HEAD_LEN = 64       # 头部元数据固定64字节
FINGER_DIM = 8      # 固定8维指纹


# ===================== 同伦形变计算函数 =====================
def calc_dim(t: float, origin_dim: int, compact_dim: int) -> int:
    """维度连续形变：d(t) = origin*(1-t) + compact*t"""
    return int(origin_dim * (1 - t) + compact_dim * t)


def interpolate_weight(t: float, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """权重连续插值：W(t) = (1-t)*W0 + t*W1"""
    return (1 - t) * w0.astype(np.float32) + t * w1.astype(np.float32)


# ===================== 二进制 λ单元读写 =====================
class LambdaUnit:
    """
    最小执行颗粒：λ函数单元
    对外仅暴露 forward(x, global_t)，内部封装权重、维度、变换逻辑
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.unit_id = ""
        self.func_type = 0          # 0=attn 1=ffn 2=norm 3=proj 4=route
        self.dim_skeleton = 0
        self.dim_finger = FINGER_DIM
        self.dim_origin = 0
        self.version = 1
        self.rule_text = ""
        self.w0: np.ndarray = None
        self.w1: np.ndarray = None
        self.bias: np.ndarray = None
        self.finger: np.ndarray = None
        self.port_config: dict = {}
        self.enable = True

    def load(self):
        """从 *.bin 文件加载单元数据"""
        with open(self.file_path, "rb") as f:
            # ── 区块1：头部 64 Byte ──────────────────────────────────
            head_data = f.read(HEAD_LEN)
            self.unit_id   = struct.unpack_from(f"{ENDIAN}32s", head_data, 0)[0].decode("utf-8").strip("\x00")
            self.func_type = struct.unpack_from(f"{ENDIAN}B",   head_data, 32)[0]
            self.dim_skeleton = struct.unpack_from(f"{ENDIAN}H", head_data, 33)[0]
            self.dim_finger   = struct.unpack_from(f"{ENDIAN}B", head_data, 35)[0]
            self.dim_origin   = struct.unpack_from(f"{ENDIAN}H", head_data, 36)[0]
            self.version      = struct.unpack_from(f"{ENDIAN}H", head_data, 38)[0]
            # 跳过 24 字节保留位（HEAD_LEN=64，已读40字节）

            # ── 区块2：控制规则区（读到 \n 结束）─────────────────────
            rule_buf = b""
            while True:
                b = f.read(1)
                if not b or b == b"\n":
                    break
                rule_buf += b
            self.rule_text = rule_buf.decode("utf-8")
            for line in self.rule_text.split(";"):
                if line.startswith("Enable="):
                    self.enable = int(line.split("=")[1]) == 1

            # ── 区块3：权重区 W0 + W1 + Bias（FP16）────────────────
            size_w = self.dim_origin * self.dim_skeleton
            w0_buf   = f.read(size_w * FP16_SIZE)
            w1_buf   = f.read(size_w * FP16_SIZE)
            bias_buf = f.read(self.dim_skeleton * FP16_SIZE)
            self.w0   = np.frombuffer(w0_buf,   dtype=np.float16).reshape(self.dim_origin, self.dim_skeleton)
            self.w1   = np.frombuffer(w1_buf,   dtype=np.float16).reshape(self.dim_origin, self.dim_skeleton)
            self.bias = np.frombuffer(bias_buf, dtype=np.float16)

            # ── 区块4：8维指纹（FP16，定长16Byte）──────────────────
            finger_buf = f.read(FINGER_DIM * FP16_SIZE)
            self.finger = np.frombuffer(finger_buf, dtype=np.float16)

            # ── 区块5：接口JSON（读到 \0 结束）──────────────────────
            json_buf = b""
            while True:
                b = f.read(1)
                if not b or b == b"\x00":
                    break
                json_buf += b
            self.port_config = json.loads(json_buf.decode("utf-8")) if json_buf else {}

    def save(self, file_path: str = None):
        """将单元数据写入 *.bin 文件"""
        path = file_path or self.file_path
        with open(path, "wb") as f:
            # ── 区块1：头部 64 Byte ──────────────────────────────────
            uid_bytes = self.unit_id.encode("utf-8").ljust(32, b"\x00")[:32]
            head = (
                uid_bytes
                + struct.pack(f"{ENDIAN}B",  self.func_type)
                + struct.pack(f"{ENDIAN}H",  self.dim_skeleton)
                + struct.pack(f"{ENDIAN}B",  self.dim_finger)
                + struct.pack(f"{ENDIAN}H",  self.dim_origin)
                + struct.pack(f"{ENDIAN}H",  self.version)
                + b"\x00" * 24   # 保留位
            )
            assert len(head) == HEAD_LEN, f"头部长度异常: {len(head)}"
            f.write(head)

            # ── 区块2：控制规则区 ────────────────────────────────────
            rule = (
                f"DimRule=d(t)={self.dim_origin}*(1-t)+{self.dim_skeleton}*t;"
                f"WeightRule=W(t)=(1-t)*W0+t*W1;"
                f"Enable={'1' if self.enable else '0'}"
            )
            f.write(rule.encode("utf-8") + b"\n")

            # ── 区块3：权重区 ────────────────────────────────────────
            f.write(self.w0.astype(np.float16).tobytes())
            f.write(self.w1.astype(np.float16).tobytes())
            f.write(self.bias.astype(np.float16).tobytes())

            # ── 区块4：指纹 ──────────────────────────────────────────
            f.write(self.finger.astype(np.float16).tobytes())

            # ── 区块5：接口JSON ──────────────────────────────────────
            jstr = json.dumps(self.port_config, ensure_ascii=False)
            f.write(jstr.encode("utf-8") + b"\x00")

    def forward(self, x: np.ndarray, global_t: float) -> np.ndarray:
        """
        单单元前向计算
        执行同伦形变 W(t) = (1-t)*W0 + t*W1，然后线性变换
        """
        if not self.enable:
            return x
        w_t  = interpolate_weight(global_t, self.w0, self.w1)
        b_t  = self.bias.astype(np.float32)
        x_f  = x.astype(np.float32)

        # 若输入维度与 origin_dim 不匹配，做截断或零填充（兼容不同压缩态输入）
        if x_f.shape[-1] != self.dim_origin:
            if x_f.shape[-1] < self.dim_origin:
                pad = np.zeros((*x_f.shape[:-1], self.dim_origin - x_f.shape[-1]), dtype=np.float32)
                x_f = np.concatenate([x_f, pad], axis=-1)
            else:
                x_f = x_f[..., :self.dim_origin]

        out = np.dot(x_f, w_t) + b_t
        return out


# ===================== 配置文件解析工具 =====================
def load_ini_config(file_path: str) -> configparser.ConfigParser:
    """加载 INI 格式配置文件"""
    cfg = configparser.ConfigParser()
    cfg.read(file_path, encoding="utf-8")
    return cfg


# ===================== 函数类型映射 =====================
FUNC_TYPE_NAMES = {
    0: "attention",
    1: "ffn",
    2: "normalization",
    3: "projection",
    4: "route",
}
