"""
modal_lambda_unit.py — 模态λ单元扩展类（增量新增，原有 base_utils.py 不修改）
继承 LambdaUnit，增量读取【模态扩展区】（区块6），完全复用原有形变/前向逻辑。
向后兼容：无区块6的旧单元自动降级为纯文本单模态，不会报错。
λ函数单元 + 同伦形变大模型系统 · 多模态单元层
"""

from __future__ import annotations
import json
import struct
import os
import numpy as np
from base_utils import LambdaUnit, ENDIAN, HEAD_LEN, FP16_SIZE, FINGER_DIM


# 区块6 默认值（旧单元无此区块时使用）
_DEFAULT_MODAL_CFG = {
    "InputModality":  ["text"],
    "OutputModality": ["text"],
    "ModalityMerge":  "sum",
    "IsModalEncoder": 0,
    "IsModalDecoder": 0,
}


class ModalLambdaUnit(LambdaUnit):
    """
    模态扩展λ单元
    在原有 5 区块基础上，可选读取区块6（模态扩展区）。
    区块6 追加在原有接口JSON(\0) 之后，格式：UTF-8 JSON + \0
    """
    def __init__(self, file_path: str):
        super().__init__(file_path)
        # 增量模态字段
        self.input_modality:  list[str] = ["text"]
        self.output_modality: list[str] = ["text"]
        self.modal_merge:     str       = "sum"
        self.is_encoder:      int       = 0
        self.is_decoder:      int       = 0
        self._has_modal_ext:  bool      = False

    # ── 加载：先跑父类完整加载，再增量读区块6 ────────────────────────
    def load(self):
        """加载单元文件（兼容有/无区块6两种格式）"""
        super().load()
        self._load_modal_ext()

    def _load_modal_ext(self):
        """
        增量读取区块6（模态扩展区）
        定位策略：计算前5区块精确字节偏移，直接 seek 定位。
        若文件在区块5结尾后无额外字节，判定为旧格式，静默降级。
        """
        try:
            offset = self._calc_block6_offset()
            file_size = os.path.getsize(self.file_path)
            if offset >= file_size:
                # 旧格式，无区块6
                return

            with open(self.file_path, "rb") as f:
                f.seek(offset)
                buf = b""
                while True:
                    b = f.read(1)
                    if not b or b == b"\x00":
                        break
                    buf += b

            if not buf:
                return

            cfg = json.loads(buf.decode("utf-8"))
            self.input_modality  = cfg.get("InputModality",  _DEFAULT_MODAL_CFG["InputModality"])
            self.output_modality = cfg.get("OutputModality", _DEFAULT_MODAL_CFG["OutputModality"])
            self.modal_merge     = cfg.get("ModalityMerge",  _DEFAULT_MODAL_CFG["ModalityMerge"])
            self.is_encoder      = cfg.get("IsModalEncoder", 0)
            self.is_decoder      = cfg.get("IsModalDecoder", 0)
            self._has_modal_ext  = True

        except Exception as e:
            # 任何解析异常均降级为默认值，不中断主流程
            pass

    def _calc_block6_offset(self) -> int:
        """
        精确计算区块6起始字节偏移（基于文件格式规范）
        规则：HEAD_LEN + 控制规则区(至\n) + 权重区(定长FP16) + 指纹区(定长) + 接口JSON(至\0) + 1(\0本身)
        """
        size_w     = self.dim_origin * self.dim_skeleton
        weight_len = size_w * FP16_SIZE * 2          # W0 + W1
        bias_len   = self.dim_skeleton * FP16_SIZE
        finger_len = FINGER_DIM * FP16_SIZE

        # 重新扫描文件，精确定位控制规则区和接口JSON的长度
        offset = 0
        with open(self.file_path, "rb") as f:
            # 跳过头部
            f.seek(HEAD_LEN)
            offset = HEAD_LEN

            # 控制规则区（读到 \n）
            while True:
                b = f.read(1)
                offset += 1
                if not b or b == b"\n":
                    break

            # 权重区（定长）
            skip = weight_len + bias_len + finger_len
            f.seek(skip, 1)
            offset += skip

            # 接口JSON（读到 \0）
            while True:
                b = f.read(1)
                offset += 1
                if not b or b == b"\x00":
                    break

        return offset

    # ── 保存：调用父类后追加区块6 ────────────────────────────────────
    def save(self, file_path: str = None):
        """
        保存单元文件
        调用父类写出前5区块，然后追加区块6（若已有模态配置）
        """
        super().save(file_path)
        path = file_path or self.file_path
        if self._has_modal_ext or any([
            self.input_modality != ["text"],
            self.output_modality != ["text"],
            self.is_encoder, self.is_decoder
        ]):
            modal_cfg = {
                "InputModality":  self.input_modality,
                "OutputModality": self.output_modality,
                "ModalityMerge":  self.modal_merge,
                "IsModalEncoder": self.is_encoder,
                "IsModalDecoder": self.is_decoder,
            }
            with open(path, "ab") as f:
                f.write(json.dumps(modal_cfg, ensure_ascii=False).encode("utf-8") + b"\x00")
            self._has_modal_ext = True

    # ── 模态感知的前向（复用父类 forward，加模态元信息透传）────────────
    def modal_forward(self, x: np.ndarray, global_t: float,
                      in_modal: str = "text") -> tuple[np.ndarray, str]:
        """
        模态感知前向：执行原有形变+变换，同时返回输出模态标记
        :return: (输出向量, 输出模态类型)
        """
        if in_modal not in self.input_modality:
            # 模态不匹配：透传向量，不进行变换（路由跳过）
            return x, in_modal

        out_vec = self.forward(x, global_t)   # 复用原有父类前向

        # 输出模态：优先匹配同模态，否则取第一个输出模态
        out_modal = in_modal if in_modal in self.output_modality else self.output_modality[0]
        return out_vec, out_modal

    def __repr__(self) -> str:
        return (f"<ModalLambdaUnit id={self.unit_id} "
                f"in={self.input_modality} out={self.output_modality} "
                f"enc={self.is_encoder} dec={self.is_decoder}>")
