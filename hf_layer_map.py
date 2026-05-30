"""
hf_layer_map.py — HuggingFace 模型层名 → λ单元类型映射表
======================================================================
职责：
  1. 识别 HuggingFace state_dict 中各权重张量属于哪种 λ函数类型
  2. 将权重键名归组（按 Transformer 层编号），决定写入哪个 group 目录
  3. 支持主流开源模型架构：GPT-2、LLaMA/LLaMA-2/3、Qwen、Mistral、
     Falcon、Bloom、OPT、ChatGLM（键名模式注册表，可持续扩展）

λ函数类型枚举（与 base_utils.py 保持一致）：
  0 = attention   自注意力层（q/k/v/o 投影）
  1 = ffn         前馈网络层（gate/up/down 投影）
  2 = normalization 归一化层（layernorm/rmsnorm）
  3 = projection  输入/输出嵌入、lm_head 等投影
  4 = route       路由层（MoE gate、位置编码等）

λ函数单元 + 同伦形变大模型系统 · HuggingFace 适配层
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

# ── λ函数类型常量 ─────────────────────────────────────────────────────
FUNC_ATTENTION      = 0
FUNC_FFN            = 1
FUNC_NORM           = 2
FUNC_PROJECTION     = 3
FUNC_ROUTE          = 4
FUNC_UNKNOWN        = -1

FUNC_TYPE_NAMES = {
    FUNC_ATTENTION:  "attention",
    FUNC_FFN:        "ffn",
    FUNC_NORM:       "norm",
    FUNC_PROJECTION: "projection",
    FUNC_ROUTE:      "route",
    FUNC_UNKNOWN:    "unknown",
}

# ── 层编号提取正则（覆盖主流模型的层索引格式）────────────────────────
_LAYER_IDX_PATTERNS = [
    r"\.layers?\.(\d+)\.",          # .layer.0. / .layers.0.
    r"\.h\.(\d+)\.",                # GPT-2: .h.0.
    r"\.transformer\.h\.(\d+)\.",   # GPT-2 full path
    r"\.blocks?\.(\d+)\.",          # Falcon: .blocks.0.
]


def extract_layer_idx(key: str) -> Optional[int]:
    """从权重键名提取 Transformer 层编号，无法识别则返回 None"""
    for pat in _LAYER_IDX_PATTERNS:
        m = re.search(pat, key)
        if m:
            return int(m.group(1))
    return None


# ══════════════════════════════════════════════════════════════════════
# 架构注册表：每个架构定义一组键名模式 → λ类型的规则
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ArchRule:
    """单条匹配规则：若键名包含任意一个 patterns 子串，则判定为 func_type"""
    patterns:  list[str]
    func_type: int
    group_hint: str = ""   # 建议所在组名（空=自动推断）


@dataclass
class ArchConfig:
    """一种模型架构的完整映射规则集合"""
    name:        str
    rules:       list[ArchRule]
    skip_keys:   list[str] = field(default_factory=list)  # 跳过这些键（不拆分）
    embed_keys:  list[str] = field(default_factory=list)  # 嵌入层键名前缀


# ── GPT-2 ──────────────────────────────────────────────────────────
GPT2_CONFIG = ArchConfig(
    name="gpt2",
    rules=[
        ArchRule(["attn.c_attn", "attn.c_proj"],    FUNC_ATTENTION, "attn_group"),
        ArchRule(["mlp.c_fc",  "mlp.c_proj"],        FUNC_FFN,       "ffn_group"),
        ArchRule(["ln_1", "ln_2", "ln_f"],           FUNC_NORM,      "norm_group"),
    ],
    skip_keys=["attn.bias", "attn.masked_bias"],
    embed_keys=["wte", "wpe", "lm_head"],
)

# ── LLaMA / LLaMA-2 / LLaMA-3 / Mistral（同架构）────────────────
LLAMA_CONFIG = ArchConfig(
    name="llama",
    rules=[
        ArchRule(["self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.v_proj", "self_attn.o_proj"],  FUNC_ATTENTION, "attn_group"),
        ArchRule(["mlp.gate_proj", "mlp.up_proj",
                  "mlp.down_proj"],                          FUNC_FFN,       "ffn_group"),
        ArchRule(["input_layernorm", "post_attention_layernorm",
                  "norm"],                                   FUNC_NORM,      "norm_group"),
        ArchRule(["self_attn.rotary_emb"],                   FUNC_ROUTE,     "route_group"),
    ],
    skip_keys=["rotary_emb.inv_freq"],
    embed_keys=["embed_tokens", "lm_head"],
)

# ── Qwen / Qwen-2 ─────────────────────────────────────────────────
QWEN_CONFIG = ArchConfig(
    name="qwen",
    rules=[
        ArchRule(["attn.c_attn", "attn.c_proj",
                  "self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.v_proj", "self_attn.o_proj"],  FUNC_ATTENTION, "attn_group"),
        ArchRule(["mlp.w1", "mlp.w2", "mlp.c_proj",
                  "mlp.gate_proj", "mlp.up_proj",
                  "mlp.down_proj"],                          FUNC_FFN,       "ffn_group"),
        ArchRule(["ln_1", "ln_2", "ln_f",
                  "input_layernorm", "post_attention_layernorm",
                  "norm"],                                   FUNC_NORM,      "norm_group"),
        ArchRule(["rotary_emb"],                             FUNC_ROUTE,     "route_group"),
    ],
    skip_keys=["rotary_emb.inv_freq"],
    embed_keys=["embed_tokens", "wte", "lm_head"],
)

# ── Falcon ────────────────────────────────────────────────────────
FALCON_CONFIG = ArchConfig(
    name="falcon",
    rules=[
        ArchRule(["self_attention.query_key_value",
                  "self_attention.dense"],                   FUNC_ATTENTION, "attn_group"),
        ArchRule(["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"], FUNC_FFN,     "ffn_group"),
        ArchRule(["input_layernorm", "ln_f",
                  "post_attention_layernorm"],               FUNC_NORM,      "norm_group"),
    ],
    skip_keys=[],
    embed_keys=["word_embeddings", "lm_head"],
)

# ── Bloom / OPT（相似架构）────────────────────────────────────────
BLOOM_CONFIG = ArchConfig(
    name="bloom",
    rules=[
        ArchRule(["self_attention.query_key_value",
                  "self_attention.dense"],                   FUNC_ATTENTION, "attn_group"),
        ArchRule(["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"], FUNC_FFN,     "ffn_group"),
        ArchRule(["input_layernorm", "post_attention_layernorm",
                  "ln_f"],                                   FUNC_NORM,      "norm_group"),
    ],
    skip_keys=[],
    embed_keys=["word_embeddings", "word_embeddings_layernorm", "lm_head"],
)

# ── ChatGLM-2/3 ───────────────────────────────────────────────────
CHATGLM_CONFIG = ArchConfig(
    name="chatglm",
    rules=[
        ArchRule(["self_attention.query_key_value",
                  "self_attention.dense"],                   FUNC_ATTENTION, "attn_group"),
        ArchRule(["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"], FUNC_FFN,     "ffn_group"),
        ArchRule(["input_layernorm", "post_attention_layernorm",
                  "final_layernorm"],                        FUNC_NORM,      "norm_group"),
        ArchRule(["rotary_pos_emb"],                         FUNC_ROUTE,     "route_group"),
    ],
    skip_keys=[],
    embed_keys=["embedding.word_embeddings", "output_layer"],
)

# ── 全局注册表（自动识别时按此顺序尝试匹配）──────────────────────────
ARCH_REGISTRY: dict[str, ArchConfig] = {
    "gpt2":    GPT2_CONFIG,
    "llama":   LLAMA_CONFIG,
    "qwen":    QWEN_CONFIG,
    "falcon":  FALCON_CONFIG,
    "bloom":   BLOOM_CONFIG,
    "chatglm": CHATGLM_CONFIG,
}


# ══════════════════════════════════════════════════════════════════════
# 核心接口：权重键名分类
# ══════════════════════════════════════════════════════════════════════
@dataclass
class WeightRecord:
    """单个权重张量的完整分类结果"""
    key:        str           # 原始 state_dict 键名
    layer_idx:  Optional[int] # Transformer 层编号（None=全局层）
    func_type:  int           # λ函数类型
    group_name: str           # 目标组目录名
    unit_id:    str           # 生成的 unit_id
    is_embed:   bool = False  # 是否为嵌入/输出投影层
    skip:       bool = False  # 是否跳过（不拆分）


def classify_key(key: str, arch_cfg: ArchConfig) -> WeightRecord:
    """
    对单个权重键名进行分类，返回 WeightRecord
    """
    # 1. 检查是否需要跳过
    for skip_pat in arch_cfg.skip_keys:
        if skip_pat in key:
            return WeightRecord(key=key, layer_idx=None, func_type=FUNC_UNKNOWN,
                                group_name="", unit_id="", skip=True)

    # 2. 检查是否为嵌入/输出投影层（全局，不属于任何 Transformer 层）
    for embed_pat in arch_cfg.embed_keys:
        if embed_pat in key:
            unit_id = re.sub(r"[.\-/]", "_", key.split(".")[-2] + "_" + key.split(".")[-1])
            return WeightRecord(key=key, layer_idx=None, func_type=FUNC_PROJECTION,
                                group_name="projection_group", unit_id=unit_id,
                                is_embed=True)

    # 3. 提取层编号
    layer_idx = extract_layer_idx(key)

    # 4. 按规则匹配 λ类型
    func_type  = FUNC_UNKNOWN
    group_name = "unknown_group"
    for rule in arch_cfg.rules:
        if any(pat in key for pat in rule.patterns):
            func_type  = rule.func_type
            group_name = rule.group_hint or f"{FUNC_TYPE_NAMES[func_type]}_group"
            break

    # 5. 构建 unit_id（层编号 + 权重名后缀）
    suffix   = key.split(".")[-1]          # weight / bias
    basename = key.split(".")[-2]          # c_attn / q_proj / ...
    layer_str = f"L{layer_idx:03d}" if layer_idx is not None else "Lglb"
    unit_id   = f"{layer_str}_{basename}_{suffix}"

    return WeightRecord(
        key=key, layer_idx=layer_idx, func_type=func_type,
        group_name=group_name, unit_id=unit_id,
    )


def auto_detect_arch(state_dict_keys: list[str]) -> Optional[ArchConfig]:
    """
    从 state_dict 键名列表自动检测模型架构
    返回最匹配的 ArchConfig，无法识别则返回 None
    """
    key_str = " ".join(state_dict_keys[:50])  # 只看前50个键即可判断

    scores: dict[str, int] = {}
    for arch_name, arch_cfg in ARCH_REGISTRY.items():
        score = 0
        for rule in arch_cfg.rules:
            for pat in rule.patterns:
                if pat in key_str:
                    score += 1
        scores[arch_name] = score

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return None
    print(f"  [LayerMap] 自动检测架构: {best} (score={scores[best]})")
    return ARCH_REGISTRY[best]


def classify_state_dict(
    state_dict_keys: list[str],
    arch_cfg:        Optional[ArchConfig] = None,
) -> list[WeightRecord]:
    """
    对整个 state_dict 键名列表批量分类
    arch_cfg=None 时自动检测架构
    """
    if arch_cfg is None:
        arch_cfg = auto_detect_arch(state_dict_keys)
    if arch_cfg is None:
        raise ValueError("无法自动识别模型架构，请手动传入 arch_cfg")

    records = [classify_key(k, arch_cfg) for k in state_dict_keys]
    skipped = sum(1 for r in records if r.skip)
    unknown = sum(1 for r in records if r.func_type == FUNC_UNKNOWN and not r.skip)
    print(f"  [LayerMap] 分类完成: 总={len(records)} 跳过={skipped} 未识别={unknown}")
    return records
