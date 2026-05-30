"""
generate_modal_units.py — 多模态专用仿真λ单元生成工具（增量新增）
在原有 generate_demo_unit.py 基础上，追加区块6（模态扩展区）。
原有仿真单元格式完全兼容，本脚本仅生成新增的模态专属目录。
λ函数单元 + 同伦形变大模型系统 · 多模态单元生成层

用法：python generate_modal_units.py
"""

import os
import sys
import json
import struct
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

ENDIAN     = "<"
HEAD_LEN   = 64
FP16_SIZE  = 2
FINGER_DIM = 8

# 模态指纹槽位
_MODAL_FINGER = {"text": 0, "image": 1, "audio": 2}


def create_modal_lambda_unit(
    save_path:   str,
    unit_id:     str,
    func_type:   int,
    origin_dim:  int  = 4096,
    compact_dim: int  = 512,
    enable:      bool = True,
    seed:        int  = None,
    # 模态扩展区参数（区块6）
    input_modality:  list = None,
    output_modality: list = None,
    modal_merge:     str  = "sum",
    is_encoder:      int  = 0,
    is_decoder:      int  = 0,
):
    """
    生成含区块6（模态扩展区）的模态λ单元
    前5区块格式与原有完全一致，区块6追加在末尾
    """
    rng = np.random.RandomState(seed)

    with open(save_path, "wb") as f:
        # ── 区块1：头部 64 Byte ──────────────────────────────────────
        uid_bytes = unit_id.encode("utf-8").ljust(32, b"\x00")[:32]
        head = (
            uid_bytes
            + struct.pack(f"{ENDIAN}B", func_type)
            + struct.pack(f"{ENDIAN}H", compact_dim)
            + struct.pack(f"{ENDIAN}B", FINGER_DIM)
            + struct.pack(f"{ENDIAN}H", origin_dim)
            + struct.pack(f"{ENDIAN}H", 1)
            + b"\x00" * 24
        )
        assert len(head) == HEAD_LEN
        f.write(head)

        # ── 区块2：控制规则区 ────────────────────────────────────────
        rule = (
            f"DimRule=d(t)={origin_dim}*(1-t)+{compact_dim}*t;"
            f"WeightRule=W(t)=(1-t)*W0+t*W1;"
            f"Enable={'1' if enable else '0'}"
        )
        f.write(rule.encode("utf-8") + b"\n")

        # ── 区块3：权重区 W0 + W1 + Bias（FP16，Xavier初始化）────────
        scale = np.sqrt(2.0 / (origin_dim + compact_dim))
        w0   = (rng.randn(origin_dim, compact_dim) * scale).astype(np.float16)
        w1   = (rng.randn(origin_dim, compact_dim) * scale).astype(np.float16)
        bias = (rng.randn(compact_dim) * 0.01).astype(np.float16)
        f.write(w0.tobytes())
        f.write(w1.tobytes())
        f.write(bias.tobytes())

        # ── 区块4：8维模态指纹（模态感知路由）──────────────────────
        finger = np.zeros(FINGER_DIM, dtype=np.float32)
        # 根据主要输入模态设置指纹槽位
        if input_modality:
            for m in input_modality:
                slot = _MODAL_FINGER.get(m, 0)
                finger[slot] = 1.0 / len(input_modality)
        else:
            finger[0] = 1.0   # 默认 text
        f.write(finger.astype(np.float16).tobytes())

        # ── 区块5：接口JSON ──────────────────────────────────────────
        merge_rule = modal_merge if is_encoder or is_decoder else "sum"
        port_cfg = json.dumps({
            "InPort":    1,
            "OutPort":   1,
            "MergeRule": merge_rule,
        })
        f.write(port_cfg.encode("utf-8") + b"\x00")

        # ── 区块6：模态扩展区（增量新增）─────────────────────────────
        modal_cfg = {
            "InputModality":  input_modality  or ["text"],
            "OutputModality": output_modality or ["text"],
            "ModalityMerge":  modal_merge,
            "IsModalEncoder": is_encoder,
            "IsModalDecoder": is_decoder,
        }
        f.write(json.dumps(modal_cfg, ensure_ascii=False).encode("utf-8") + b"\x00")

    in_m  = ",".join(input_modality  or ["text"])
    out_m = ",".join(output_modality or ["text"])
    tag   = " [编码器]" if is_encoder else (" [解码器]" if is_decoder else "")
    print(f"    生成: {os.path.basename(save_path)}  ({in_m}→{out_m}){tag}")


def write_group_conf(
    group_dir:   str,
    group_name:  str,
    unit_files:  list,
    group_type:  str  = "serial",
    merge:       str  = "sum",
    modalities:  list = None,
):
    """写入 group.conf（含增量 SupportModality 字段）"""
    modal_str = ",".join(modalities or ["text"])
    with open(os.path.join(group_dir, "group.conf"), "w", encoding="utf-8") as f:
        f.write(f"[GroupBase]\n")
        f.write(f"GroupName={group_name}\n")
        f.write(f"UnitList={','.join(unit_files)}\n")
        f.write(f"GroupType={group_type}\n")
        f.write(f"LocalTOffset=0.0\n")
        f.write(f"; ── 增量模态配置 ──\n")
        f.write(f"SupportModality={modal_str}\n")
        f.write(f"ModalFusion=enable\n\n")
        f.write(f"[MergeConfig]\n")
        f.write(f"DefaultMerge={merge}\n")
    print(f"    group.conf 已写入 (SupportModality={modal_str})")


def generate_modal_groups(model_root: str, origin_dim: int = 4096, compact_dim: int = 512):
    """生成所有多模态专属组（仅新增目录，不修改原有组）"""

    groups = [
        # ── 文本编码组 ────────────────────────────────────────────
        {
            "dir":        "encoder_text",
            "group_type": "serial",
            "merge":      "sum",
            "modalities": ["text"],
            "units": [
                {"id": "enc_txt_001", "ft": 3, "in_m": ["text"], "out_m": ["text"],
                 "is_enc": 1, "is_dec": 0, "sfx": "3"},
            ],
        },
        # ── 图像编码组 ────────────────────────────────────────────
        {
            "dir":        "encoder_image",
            "group_type": "serial",
            "merge":      "sum",
            "modalities": ["image"],
            "units": [
                {"id": "enc_img_001", "ft": 3, "in_m": ["image"], "out_m": ["text"],
                 "is_enc": 1, "is_dec": 0, "sfx": "3"},
            ],
        },
        # ── 音频编码组 ────────────────────────────────────────────
        {
            "dir":        "encoder_audio",
            "group_type": "serial",
            "merge":      "sum",
            "modalities": ["audio"],
            "units": [
                {"id": "enc_aud_001", "ft": 3, "in_m": ["audio"], "out_m": ["text"],
                 "is_enc": 1, "is_dec": 0, "sfx": "3"},
            ],
        },
        # ── 多模态融合并联组 ──────────────────────────────────────
        {
            "dir":        "modal_fusion_group",
            "group_type": "parallel",
            "merge":      "weight",
            "modalities": ["text", "image", "audio"],
            "units": [
                {"id": "fuse_001", "ft": 4, "in_m": ["text", "image", "audio"],
                 "out_m": ["text"], "is_enc": 0, "is_dec": 0, "sfx": "4"},
                {"id": "fuse_002", "ft": 4, "in_m": ["text", "image", "audio"],
                 "out_m": ["text"], "is_enc": 0, "is_dec": 0, "sfx": "4"},
            ],
        },
        # ── 文本解码组 ────────────────────────────────────────────
        {
            "dir":        "decoder_text",
            "group_type": "serial",
            "merge":      "sum",
            "modalities": ["text"],
            "units": [
                {"id": "dec_txt_001", "ft": 3, "in_m": ["text"], "out_m": ["text"],
                 "is_enc": 0, "is_dec": 1, "sfx": "3"},
            ],
        },
        # ── 图像解码组 ────────────────────────────────────────────
        {
            "dir":        "decoder_image",
            "group_type": "serial",
            "merge":      "sum",
            "modalities": ["image"],
            "units": [
                {"id": "dec_img_001", "ft": 3, "in_m": ["text"], "out_m": ["image"],
                 "is_enc": 0, "is_dec": 1, "sfx": "3"},
            ],
        },
        # ── 音频解码组 ────────────────────────────────────────────
        {
            "dir":        "decoder_audio",
            "group_type": "serial",
            "merge":      "sum",
            "modalities": ["audio"],
            "units": [
                {"id": "dec_aud_001", "ft": 3, "in_m": ["text"], "out_m": ["audio"],
                 "is_enc": 0, "is_dec": 1, "sfx": "3"},
            ],
        },
    ]

    for g in groups:
        gdir = os.path.join(model_root, g["dir"])
        os.makedirs(gdir, exist_ok=True)
        print(f"\n  [{g['dir']}]  模式={g['group_type']}")

        unit_files = []
        for u in g["units"]:
            fname = f"lambda_{u['id']}_{u['sfx']}.bin"
            unit_files.append(fname)
            create_modal_lambda_unit(
                save_path       = os.path.join(gdir, fname),
                unit_id         = u["id"],
                func_type       = u["ft"],
                origin_dim      = origin_dim,
                compact_dim     = compact_dim,
                seed            = hash(u["id"]) % (2**31),
                input_modality  = u["in_m"],
                output_modality = u["out_m"],
                modal_merge     = g["merge"],
                is_encoder      = u["is_enc"],
                is_decoder      = u["is_dec"],
            )

        write_group_conf(
            group_dir  = gdir,
            group_name = g["dir"],
            unit_files = unit_files,
            group_type = g["group_type"],
            merge      = g["merge"],
            modalities = g["modalities"],
        )


if __name__ == "__main__":
    model_root = os.path.join(script_dir, "model_root")
    print("═══ 多模态专用仿真λ单元批量生成 ═══")
    print(f"  输出根目录: {model_root}\n")
    generate_modal_groups(model_root)
    print("\n═══ 多模态单元生成完毕 ✓ ═══")
