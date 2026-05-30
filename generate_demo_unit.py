"""
generate_demo_unit.py — 仿真λ单元生成工具
功能：批量生成所有组的仿真 *.bin 文件（开发调试用，无需真实大模型权重）
λ函数单元 + 同伦形变大模型系统 · 仿真数据层
用法：python generate_demo_unit.py
"""

import os
import json
import struct
import numpy as np

ENDIAN     = "<"
HEAD_LEN   = 64
FP16_SIZE  = 2
FINGER_DIM = 8

# 函数类型枚举
FUNC_TYPES = {
    "attention":     0,
    "ffn":           1,
    "normalization": 2,
    "projection":    3,
    "route":         4,
}


def create_lambda_unit(
    save_path: str,
    unit_id: str,
    func_type: int,
    origin_dim: int = 4096,
    compact_dim: int = 512,
    enable: bool = True,
    seed: int = None,
):
    """
    生成单个仿真 λ单元 *.bin 文件
    所有权重为随机 FP16 数据（纯逻辑验证，不依赖真实模型）
    """
    rng = np.random.RandomState(seed)

    with open(save_path, "wb") as f:
        # ── 区块1：头部 64 Byte ──────────────────────────────────
        uid_bytes = unit_id.encode("utf-8").ljust(32, b"\x00")[:32]
        head = (
            uid_bytes
            + struct.pack(f"{ENDIAN}B", func_type)
            + struct.pack(f"{ENDIAN}H", compact_dim)
            + struct.pack(f"{ENDIAN}B", FINGER_DIM)
            + struct.pack(f"{ENDIAN}H", origin_dim)
            + struct.pack(f"{ENDIAN}H", 1)          # version=1
            + b"\x00" * 24                           # 保留位
        )
        assert len(head) == HEAD_LEN, f"头部长度异常: {len(head)}"
        f.write(head)

        # ── 区块2：控制规则区 ────────────────────────────────────
        rule = (
            f"DimRule=d(t)={origin_dim}*(1-t)+{compact_dim}*t;"
            f"WeightRule=W(t)=(1-t)*W0+t*W1;"
            f"Enable={'1' if enable else '0'}"
        )
        f.write(rule.encode("utf-8") + b"\n")

        # ── 区块3：权重区 W0 + W1 + Bias（FP16）────────────────
        # 使用 Xavier 初始化比例，仿真时保持数值稳定
        scale = np.sqrt(2.0 / (origin_dim + compact_dim))
        w0   = (rng.randn(origin_dim, compact_dim) * scale).astype(np.float16)
        w1   = (rng.randn(origin_dim, compact_dim) * scale).astype(np.float16)
        bias = (rng.randn(compact_dim) * 0.01).astype(np.float16)
        f.write(w0.tobytes())
        f.write(w1.tobytes())
        f.write(bias.tobytes())

        # ── 区块4：8维指纹（归一化单位向量）────────────────────
        raw_finger = rng.randn(FINGER_DIM).astype(np.float16)
        norm = np.linalg.norm(raw_finger.astype(np.float32)) + 1e-8
        finger = (raw_finger.astype(np.float32) / norm).astype(np.float16)
        f.write(finger.tobytes())

        # ── 区块5：接口JSON ──────────────────────────────────────
        port_cfg = json.dumps({"InPort": 1, "OutPort": 1, "MergeRule": "sum"})
        f.write(port_cfg.encode("utf-8") + b"\x00")

    print(f"    生成: {os.path.basename(save_path)}  (type={func_type}, dim={origin_dim}→{compact_dim})")


def generate_all_demo_units(model_root: str, origin_dim: int = 4096, compact_dim: int = 512):
    """
    按规范生成四组仿真单元 + 各组 group.conf
    attn_group:  3个注意力单元（并联）
    ffn_group:   2个FFN单元（串联）
    norm_group:  1个归一化单元（串联）
    route_group: 1个路由单元（串联）
    """
    groups = [
        {
            "name":       "attn_group",
            "group_type": "parallel",
            "merge":      "sum",
            "func_type":  FUNC_TYPES["attention"],
            "units": [
                {"id": "attn_001", "suffix": "0"},
                {"id": "attn_002", "suffix": "0"},
                {"id": "attn_003", "suffix": "0"},
            ],
        },
        {
            "name":       "ffn_group",
            "group_type": "serial",
            "merge":      "sum",
            "func_type":  FUNC_TYPES["ffn"],
            "units": [
                {"id": "ffn_001", "suffix": "1"},
                {"id": "ffn_002", "suffix": "1"},
            ],
        },
        {
            "name":       "norm_group",
            "group_type": "serial",
            "merge":      "sum",
            "func_type":  FUNC_TYPES["normalization"],
            "units": [
                {"id": "norm_001", "suffix": "2"},
            ],
        },
        {
            "name":       "route_group",
            "group_type": "serial",
            "merge":      "sum",
            "func_type":  FUNC_TYPES["route"],
            "units": [
                {"id": "route_001", "suffix": "4"},
            ],
        },
    ]

    for group in groups:
        group_dir = os.path.join(model_root, group["name"])
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n  [{group['name']}]  模式={group['group_type']}")

        unit_files = []
        for i, u in enumerate(group["units"]):
            filename = f"lambda_{u['id']}_{u['suffix']}.bin"
            unit_files.append(filename)
            save_path = os.path.join(group_dir, filename)
            # 用固定 seed 保证可复现
            create_lambda_unit(
                save_path,
                unit_id=u["id"],
                func_type=group["func_type"],
                origin_dim=origin_dim,
                compact_dim=compact_dim,
                seed=hash(u["id"]) % (2**31),
            )

        # 写 group.conf
        conf_path = os.path.join(group_dir, "group.conf")
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(f"[GroupBase]\n")
            f.write(f"GroupName={group['name']}\n")
            f.write(f"UnitList={','.join(unit_files)}\n")
            f.write(f"GroupType={group['group_type']}\n")
            f.write(f"LocalTOffset=0.0\n\n")
            f.write(f"[MergeConfig]\n")
            f.write(f"DefaultMerge={group['merge']}\n")
        print(f"    group.conf 已写入")


if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    model_root  = os.path.join(script_dir, "model_root")

    print("═══ 仿真 λ单元批量生成 ═══")
    print(f"  输出根目录: {model_root}\n")
    generate_all_demo_units(model_root, origin_dim=4096, compact_dim=512)
    print("\n═══ 所有仿真单元生成完毕 ✓ ═══")
