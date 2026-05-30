"""
hf_splitter.py — HuggingFace 模型权重拆分主脚本
======================================================================
功能：
  将 HuggingFace transformers 模型（本地路径或 Hub 模型名）自动拆分为
  符合本架构规范的 λ单元 *.bin 文件集合，生成完整 model_root 目录树。

核心流程：
  1. 加载模型 state_dict（支持 safetensors / pytorch_model.bin / 分片）
  2. 自动检测架构（或手动指定），调用 hf_layer_map 分类所有权重
  3. 按层编号分组，构建 W0（原始权重）和 W1（SVD降维压缩权重）
  4. 写出 λ单元 *.bin 文件（完整6区块格式，含模态扩展区）
  5. 生成各组 group.conf 和全局 model_global.conf

无 GPU/PyTorch 的降级路径：
  若环境无 PyTorch，则使用 numpy 模拟（仅用于接口/格式验证）

λ函数单元 + 同伦形变大模型系统 · HuggingFace 拆分层

用法：
  # 拆分本地模型
  python hf_splitter.py --model_path ./my_llama --output ./split_model

  # 拆分 Hub 模型（需网络）
  python hf_splitter.py --model_name gpt2 --output ./split_gpt2

  # 仅用 numpy mock 验证流程（无需 PyTorch）
  python hf_splitter.py --mock --output ./mock_split
"""

from __future__ import annotations
import os, sys, argparse, json, struct, time
import numpy as np
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from hf_layer_map import (
    classify_state_dict, auto_detect_arch, WeightRecord,
    ARCH_REGISTRY, FUNC_TYPE_NAMES, FUNC_UNKNOWN,
)
from base_utils import ENDIAN, HEAD_LEN, FP16_SIZE, FINGER_DIM

# ── 全局常量 ────────────────────────────────────────────────────────────
DEFAULT_COMPACT_DIM = 512     # 默认骨架目标维度
DEFAULT_ORIGIN_DIM  = 4096    # 默认原始维度（与模型隐藏层对齐）
SVD_RANK            = 512     # SVD 截断秩（生成 W1 的降维目标）


# ══════════════════════════════════════════════════════════════════════
# W1 生成策略：SVD 截断降维
# ══════════════════════════════════════════════════════════════════════
def compute_w1_svd(w0: np.ndarray, compact_dim: int) -> np.ndarray:
    """
    用截断 SVD 生成 W1（压缩权重矩阵）
    W0: (origin_dim, ...) → 展平为 2D → SVD → 取前 compact_dim 奇异向量
    返回与 W0 同 shape 的 W1，但低秩近似（信息量压缩）
    """
    orig_shape = w0.shape
    w0_f = w0.astype(np.float32)

    # 强制转为 2D 矩阵
    if w0_f.ndim == 1:
        # 偏置向量：直接缩放到目标维度（不做 SVD）
        if len(w0_f) > compact_dim:
            return w0_f[:compact_dim].astype(np.float16)
        pad = np.zeros(compact_dim - len(w0_f), dtype=np.float32)
        return np.concatenate([w0_f, pad]).astype(np.float16)

    rows, cols = w0_f.shape[0], int(np.prod(w0_f.shape[1:]))
    mat = w0_f.reshape(rows, cols)

    # SVD（经济模式，节省内存）
    try:
        rank = min(compact_dim, rows, cols)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        # 截断到 rank
        U_r  = U[:, :rank]
        S_r  = S[:rank]
        Vt_r = Vt[:rank, :]
        # 重构低秩近似
        w1_mat = (U_r * S_r) @ Vt_r
    except np.linalg.LinAlgError:
        # SVD 失败降级：直接用 W0 的缩放版本
        w1_mat = mat * (compact_dim / max(rows, cols))

    # 还原 shape，对齐至标准 (origin_dim, compact_dim)
    origin_dim = orig_shape[0]
    if w1_mat.shape[0] != origin_dim:
        if w1_mat.shape[0] < origin_dim:
            pad = np.zeros((origin_dim - w1_mat.shape[0], w1_mat.shape[1]), dtype=np.float32)
            w1_mat = np.vstack([w1_mat, pad])
        else:
            w1_mat = w1_mat[:origin_dim]

    if w1_mat.shape[1] < compact_dim:
        pad = np.zeros((w1_mat.shape[0], compact_dim - w1_mat.shape[1]), dtype=np.float32)
        w1_mat = np.hstack([w1_mat, pad])
    else:
        w1_mat = w1_mat[:, :compact_dim]

    return w1_mat.astype(np.float16)


def align_w0(w: np.ndarray, origin_dim: int, compact_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    将原始权重 tensor 对齐至标准 (origin_dim, compact_dim) 的 W0，
    同时生成对应 W1（SVD 压缩）
    """
    w_f = w.astype(np.float32)

    # 偏置向量：1D → 对齐至 compact_dim
    if w_f.ndim == 1:
        if len(w_f) >= compact_dim:
            w0_aligned = w_f[:compact_dim]
        else:
            w0_aligned = np.concatenate([w_f, np.zeros(compact_dim - len(w_f))])
        w1_aligned = w0_aligned * 0.5   # 偏置 W1：缩放版本
        return w0_aligned.astype(np.float16), w1_aligned.astype(np.float16)

    # 2D 权重矩阵
    rows = w_f.shape[0]
    cols = int(np.prod(w_f.shape[1:]))
    mat  = w_f.reshape(rows, cols)

    # 行对齐至 origin_dim
    if rows < origin_dim:
        pad_r = np.zeros((origin_dim - rows, cols), dtype=np.float32)
        mat   = np.vstack([mat, pad_r])
    else:
        mat = mat[:origin_dim]

    # 列对齐至 compact_dim（W0 侧用截断/填充，保留原始信息）
    if cols < compact_dim:
        pad_c = np.zeros((origin_dim, compact_dim - cols), dtype=np.float32)
        w0_aligned = np.hstack([mat, pad_c])
    else:
        w0_aligned = mat[:, :compact_dim]

    # W1：SVD 压缩
    w1_aligned = compute_w1_svd(mat, compact_dim)

    return w0_aligned.astype(np.float16), w1_aligned.astype(np.float16)


# ══════════════════════════════════════════════════════════════════════
# λ单元写出（复用 base_utils 格式规范）
# ══════════════════════════════════════════════════════════════════════
def write_lambda_unit(
    save_path:   str,
    unit_id:     str,
    func_type:   int,
    w0:          np.ndarray,   # (origin_dim, compact_dim) FP16
    w1:          np.ndarray,   # (origin_dim, compact_dim) FP16
    bias:        np.ndarray,   # (compact_dim,) FP16
    finger:      np.ndarray,   # (8,) FP16
    origin_dim:  int,
    compact_dim: int,
    source_key:  str = "",     # 原始 HF 权重键名（存入模态扩展区备注）
):
    """写出完整6区块 λ单元 *.bin 文件"""
    with open(save_path, "wb") as f:
        # ── 区块1：头部 ──────────────────────────────────────────────
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
            f"Enable=1"
        )
        f.write(rule.encode("utf-8") + b"\n")

        # ── 区块3：权重区（FP16）────────────────────────────────────
        # 确保 shape 和 dtype 严格正确
        w0_write = w0.astype(np.float16).reshape(origin_dim, compact_dim)
        w1_write = w1.astype(np.float16).reshape(origin_dim, compact_dim)
        b_write  = bias.astype(np.float16).flatten()[:compact_dim]
        if len(b_write) < compact_dim:
            b_write = np.concatenate([b_write, np.zeros(compact_dim - len(b_write), dtype=np.float16)])

        f.write(w0_write.tobytes())
        f.write(w1_write.tobytes())
        f.write(b_write.tobytes())

        # ── 区块4：指纹 ──────────────────────────────────────────────
        f.write(finger.astype(np.float16).tobytes())

        # ── 区块5：接口JSON ──────────────────────────────────────────
        port_cfg = json.dumps({"InPort": 1, "OutPort": 1, "MergeRule": "sum"})
        f.write(port_cfg.encode("utf-8") + b"\x00")

        # ── 区块6：模态扩展区（含原始键名溯源）──────────────────────
        modal_cfg = json.dumps({
            "InputModality":  ["text"],
            "OutputModality": ["text"],
            "ModalityMerge":  "sum",
            "IsModalEncoder": 0,
            "IsModalDecoder": 0,
            "SourceKey":      source_key,     # 溯源：原始 HF 权重键名
            "SplitTool":      "hf_splitter",
        }, ensure_ascii=False)
        f.write(modal_cfg.encode("utf-8") + b"\x00")


# ══════════════════════════════════════════════════════════════════════
# 指纹生成（基于权重特征，比仿真随机更有意义）
# ══════════════════════════════════════════════════════════════════════
def compute_weight_finger(w0: np.ndarray) -> np.ndarray:
    """
    从 W0 权重矩阵提取 8 维语义指纹
    策略：取8个等分块的 Frobenius 范数 → 归一化
    比随机指纹更能反映权重分布特征，用于稀疏路由相似度匹配
    """
    w_f   = w0.astype(np.float32).flatten()
    seg   = max(1, len(w_f) // FINGER_DIM)
    norms = np.array([
        np.linalg.norm(w_f[i*seg:(i+1)*seg])
        for i in range(FINGER_DIM)
    ], dtype=np.float32)
    total = np.linalg.norm(norms) + 1e-8
    return (norms / total).astype(np.float16)


# ══════════════════════════════════════════════════════════════════════
# 主拆分器
# ══════════════════════════════════════════════════════════════════════
class HFSplitter:
    """
    HuggingFace 模型权重拆分器
    """
    def __init__(
        self,
        output_dir:  str,
        origin_dim:  int = DEFAULT_ORIGIN_DIM,
        compact_dim: int = DEFAULT_COMPACT_DIM,
        arch_name:   str = "",    # 强制指定架构，空=自动检测
        dry_run:     bool = False, # True=只分析不写文件
    ):
        self.output_dir  = output_dir
        self.origin_dim  = origin_dim
        self.compact_dim = compact_dim
        self.arch_name   = arch_name
        self.dry_run     = dry_run
        self._stats      = {"total": 0, "written": 0, "skipped": 0, "errors": 0}

    # ── 加载模型 state_dict ────────────────────────────────────────
    def load_state_dict(self, model_path: str) -> dict:
        """
        支持多种格式：
          - safetensors（优先，更快更安全）
          - pytorch_model.bin（单文件）
          - pytorch_model-xxxxx-of-xxxxx.bin（分片）
          - HuggingFace Hub 模型名（需网络+transformers）
        """
        path = Path(model_path)

        # safetensors 格式
        sf_files = list(path.glob("*.safetensors")) if path.is_dir() else (
            [path] if path.suffix == ".safetensors" else []
        )
        if sf_files:
            return self._load_safetensors(sf_files)

        # pytorch bin 格式
        bin_files = list(path.glob("pytorch_model*.bin")) if path.is_dir() else (
            [path] if path.suffix == ".bin" else []
        )
        if bin_files:
            return self._load_pytorch_bins(bin_files)

        raise FileNotFoundError(
            f"在 {model_path} 未找到 safetensors 或 pytorch_model*.bin 文件"
        )

    def _load_safetensors(self, files: list) -> dict:
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("请安装 safetensors: pip install safetensors")
        state_dict = {}
        for f in sorted(files):
            print(f"  加载 safetensors: {f.name}")
            with safe_open(str(f), framework="numpy") as st:
                for key in st.keys():
                    state_dict[key] = st.get_tensor(key)
        return state_dict

    def _load_pytorch_bins(self, files: list) -> dict:
        try:
            import torch
        except ImportError:
            raise ImportError("请安装 PyTorch: pip install torch")
        state_dict = {}
        for f in sorted(files):
            print(f"  加载 pytorch bin: {f.name}")
            sd = torch.load(str(f), map_location="cpu", weights_only=True)
            for k, v in sd.items():
                state_dict[k] = v.numpy() if hasattr(v, "numpy") else np.array(v)
        return state_dict

    # ── 核心拆分逻辑 ──────────────────────────────────────────────
    def split(self, state_dict: dict, arch_cfg=None) -> dict:
        """
        执行拆分：state_dict → model_root 目录树
        返回按 group_name 分组的 WeightRecord 汇总
        """
        keys = list(state_dict.keys())
        print(f"\n  共 {len(keys)} 个权重张量，开始分类...")

        # 架构检测
        if arch_cfg is None:
            if self.arch_name and self.arch_name in ARCH_REGISTRY:
                arch_cfg = ARCH_REGISTRY[self.arch_name]
                print(f"  [Splitter] 使用指定架构: {self.arch_name}")
            else:
                arch_cfg = auto_detect_arch(keys)
                if arch_cfg is None:
                    raise ValueError("无法自动识别架构，请用 --arch 参数指定")

        records = classify_state_dict(keys, arch_cfg)

        # 按 group 分组
        groups: dict[str, list[WeightRecord]] = {}
        for rec in records:
            if rec.skip or rec.func_type == FUNC_UNKNOWN:
                self._stats["skipped"] += 1
                continue
            groups.setdefault(rec.group_name, []).append(rec)

        if not self.dry_run:
            os.makedirs(self.output_dir, exist_ok=True)

        # 按 group 写出单元
        group_unit_files: dict[str, list[str]] = {}
        for gname, recs in sorted(groups.items()):
            unit_files = self._write_group(gname, recs, state_dict)
            group_unit_files[gname] = unit_files

        # 写出配置文件
        if not self.dry_run:
            self._write_all_group_confs(group_unit_files, groups)
            self._write_global_conf(group_unit_files)

        print(f"\n  拆分完成: 写出={self._stats['written']}  "
              f"跳过={self._stats['skipped']}  错误={self._stats['errors']}")
        return group_unit_files

    def _write_group(
        self, group_name: str, records: list[WeightRecord], state_dict: dict
    ) -> list[str]:
        """写出一个组的所有λ单元，返回写出的文件名列表"""
        gdir = os.path.join(self.output_dir, group_name)
        if not self.dry_run:
            os.makedirs(gdir, exist_ok=True)

        print(f"\n  [{group_name}]  {len(records)} 个权重张量")
        unit_files = []

        for rec in records:
            try:
                tensor = state_dict[rec.key]
                w_np   = tensor if isinstance(tensor, np.ndarray) else np.array(tensor)

                # 归一化层 weight 是 1D 向量（hidden_dim,），特殊处理
                # 把它当 bias 存储：W0/W1 退化为对角/缩放矩阵
                is_norm_weight = (
                    rec.func_type == 2   # FUNC_NORM
                    and w_np.ndim == 1
                )
                if is_norm_weight:
                    # 将 norm weight 向量对齐至 compact_dim 作为 bias
                    nw = w_np.astype(np.float32).flatten()
                    if len(nw) >= self.compact_dim:
                        bias_vec = nw[:self.compact_dim].astype(np.float16)
                    else:
                        bias_vec = np.concatenate(
                            [nw, np.ones(self.compact_dim - len(nw), dtype=np.float32)]
                        ).astype(np.float16)
                    # W0 = 单位矩阵（scale=1 不变换），W1 = 缩放版（压缩表示）
                    w0 = np.eye(self.origin_dim, self.compact_dim, dtype=np.float16)
                    w1 = (np.eye(self.origin_dim, self.compact_dim, dtype=np.float32)
                          * 0.5).astype(np.float16)
                    finger = compute_weight_finger(w0)

                    fname = f"lambda_{rec.unit_id}_{rec.func_type}.bin"
                    fpath = os.path.join(gdir, fname)
                    if not self.dry_run:
                        write_lambda_unit(
                            save_path=fpath, unit_id=rec.unit_id[:31],
                            func_type=rec.func_type, w0=w0, w1=w1,
                            bias=bias_vec, finger=finger,
                            origin_dim=self.origin_dim, compact_dim=self.compact_dim,
                            source_key=rec.key,
                        )
                        self._stats["written"] += 1
                        self._stats["total"]   += 1
                        unit_files.append(fname)
                        print(f"    ✓ {fname}  [norm-weight] src={rec.key[:60]}")
                    else:
                        unit_files.append(fname)
                    continue   # 跳过后续通用逻辑

                # 对齐并生成 W0/W1
                w0, w1 = align_w0(w_np, self.origin_dim, self.compact_dim)

                # 偏置（默认零向量，若当前张量是 bias 则用实际值）
                is_bias = rec.key.endswith(".bias") or rec.key.endswith("_bias")
                if is_bias and w_np.ndim == 1:
                    bias_vec = w0.flatten()[:self.compact_dim]
                    if len(bias_vec) < self.compact_dim:
                        bias_vec = np.concatenate([
                            bias_vec,
                            np.zeros(self.compact_dim - len(bias_vec), dtype=np.float16)
                        ])
                    # bias 本身作为 bias 区，W0/W1 退化为单位映射
                    w0 = np.eye(self.origin_dim, self.compact_dim, dtype=np.float16)
                    w1 = np.eye(self.origin_dim, self.compact_dim, dtype=np.float16) * 0.5
                else:
                    bias_vec = np.zeros(self.compact_dim, dtype=np.float16)

                # 权重指纹（从 W0 提取）
                finger = compute_weight_finger(w0)

                fname = f"lambda_{rec.unit_id}_{rec.func_type}.bin"
                fpath = os.path.join(gdir, fname)

                if not self.dry_run:
                    write_lambda_unit(
                        save_path   = fpath,
                        unit_id     = rec.unit_id[:31],  # 截断至 31 字符（头部限制）
                        func_type   = rec.func_type,
                        w0          = w0,
                        w1          = w1,
                        bias        = bias_vec,
                        finger      = finger,
                        origin_dim  = self.origin_dim,
                        compact_dim = self.compact_dim,
                        source_key  = rec.key,
                    )
                    self._stats["written"] += 1
                    self._stats["total"]   += 1
                    unit_files.append(fname)
                    print(f"    ✓ {fname}  src={rec.key[:60]}")
                else:
                    print(f"    [dry-run] {fname}  src={rec.key}")
                    unit_files.append(fname)

            except Exception as e:
                self._stats["errors"] += 1
                print(f"    ✗ 错误 [{rec.key}]: {e}")

        return unit_files

    def _write_all_group_confs(
        self,
        group_unit_files: dict[str, list[str]],
        groups: dict[str, list[WeightRecord]],
    ):
        """为每个组写出 group.conf"""
        for gname, fnames in group_unit_files.items():
            if not fnames:
                continue
            recs = groups.get(gname, [])
            # 判断并联还是串联：attn_group 多头用并联，其余串联
            gtype = "parallel" if "attn" in gname else "serial"
            modality = "text"

            conf = (
                f"[GroupBase]\n"
                f"GroupName={gname}\n"
                f"UnitList={','.join(fnames)}\n"
                f"GroupType={gtype}\n"
                f"LocalTOffset=0.0\n"
                f"SupportModality={modality}\n"
                f"ModalFusion=enable\n\n"
                f"[MergeConfig]\n"
                f"DefaultMerge=sum\n"
            )
            conf_path = os.path.join(self.output_dir, gname, "group.conf")
            with open(conf_path, "w", encoding="utf-8") as f:
                f.write(conf)

    def _write_global_conf(self, group_unit_files: dict[str, list[str]]):
        """生成全局 model_global.conf"""
        # 按照 norm→attn→ffn→route→projection 顺序构建拓扑链路
        order = ["norm_group", "attn_group", "ffn_group", "route_group", "projection_group"]
        present = [g for g in order if g in group_unit_files and group_unit_files[g]]
        extra   = [g for g in group_unit_files if g not in order and group_unit_files[g]]
        all_groups = present + extra

        topo_steps = []
        for g in all_groups:
            mode = "parallel" if "attn" in g else "serial"
            topo_steps.append(f"{mode}({g})")
        topo_chain = " -> ".join(topo_steps)

        conf = (
            f"[GlobalControl]\n"
            f"GlobalT=0.0\n"
            f"OriginDim={self.origin_dim}\n"
            f"CompactDim={self.compact_dim}\n\n"
            f"[GlobalRule]\n"
            f"DimFormula=d(t)={{OriginDim}}*(1-t)+{{CompactDim}}*t\n"
            f"WeightFormula=W(t)=(1-t)*W0+t*W1\n\n"
            f"[Topology]\n"
            f"TopologyChain={topo_chain}\n\n"
            f"[MultiModalGlobal]\n"
            f"ModalEnable=1\n"
            f"DefaultInputModal=text\n"
            f"DefaultOutputModal=text\n"
            f"CrossModalRoute=1\n\n"
            f"[MultiTConfig]\n"
            f"BlendMode=weighted\n\n"
            f"[System]\n"
            f"LazyLoad=1\n"
            f"ThreadNum=4\n"
        )
        conf_path = os.path.join(self.output_dir, "model_global.conf")
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(conf)
        print(f"\n  model_global.conf 已生成: {conf_path}")
        print(f"  拓扑链路: {topo_chain}")


# ══════════════════════════════════════════════════════════════════════
# Mock 模式：无需真实模型，用随机权重验证拆分流程
# ══════════════════════════════════════════════════════════════════════
def build_mock_state_dict(arch_name: str = "llama", n_layers: int = 4) -> dict:
    """
    构建模拟 LLaMA state_dict（随机权重，用于验证拆分流程）
    不需要 PyTorch，纯 numpy
    """
    np.random.seed(42)
    state_dict = {}
    hidden = 4096
    ffn_dim = 11008

    # 嵌入层
    state_dict["model.embed_tokens.weight"] = np.random.randn(32000, hidden).astype(np.float16)
    state_dict["lm_head.weight"]            = np.random.randn(32000, hidden).astype(np.float16)
    state_dict["model.norm.weight"]         = np.random.randn(hidden).astype(np.float16)

    for i in range(n_layers):
        base = f"model.layers.{i}"
        # 注意力层
        state_dict[f"{base}.self_attn.q_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float16)
        state_dict[f"{base}.self_attn.k_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float16)
        state_dict[f"{base}.self_attn.v_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float16)
        state_dict[f"{base}.self_attn.o_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float16)
        # FFN 层
        state_dict[f"{base}.mlp.gate_proj.weight"] = np.random.randn(ffn_dim, hidden).astype(np.float16)
        state_dict[f"{base}.mlp.up_proj.weight"]   = np.random.randn(ffn_dim, hidden).astype(np.float16)
        state_dict[f"{base}.mlp.down_proj.weight"] = np.random.randn(hidden, ffn_dim).astype(np.float16)
        # 归一化层
        state_dict[f"{base}.input_layernorm.weight"]           = np.random.randn(hidden).astype(np.float16)
        state_dict[f"{base}.post_attention_layernorm.weight"]  = np.random.randn(hidden).astype(np.float16)

    print(f"  [Mock] 构建 {arch_name} mock state_dict: {len(state_dict)} 个张量, {n_layers} 层")
    return state_dict


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="HuggingFace 模型权重拆分工具")
    parser.add_argument("--model_path", type=str, default="",
                        help="本地模型路径（含 safetensors 或 pytorch_model.bin）")
    parser.add_argument("--model_name", type=str, default="",
                        help="HuggingFace Hub 模型名（需网络+transformers）")
    parser.add_argument("--output",     type=str, default="./hf_split_output",
                        help="输出 model_root 目录")
    parser.add_argument("--arch",       type=str, default="",
                        help=f"强制指定架构: {list(ARCH_REGISTRY.keys())}")
    parser.add_argument("--origin_dim", type=int, default=DEFAULT_ORIGIN_DIM)
    parser.add_argument("--compact_dim",type=int, default=DEFAULT_COMPACT_DIM)
    parser.add_argument("--dry_run",    action="store_true",
                        help="只分析不写文件")
    parser.add_argument("--mock",       action="store_true",
                        help="使用模拟权重验证流程（无需真实模型）")
    parser.add_argument("--n_layers",   type=int, default=4,
                        help="mock 模式下的层数")
    args = parser.parse_args()

    splitter = HFSplitter(
        output_dir  = args.output,
        origin_dim  = args.origin_dim,
        compact_dim = args.compact_dim,
        arch_name   = args.arch,
        dry_run     = args.dry_run,
    )

    t0 = time.time()
    print(f"\n{'═'*60}")
    print(f"  HuggingFace 权重拆分工具")
    print(f"  输出目录: {args.output}")
    print(f"  origin_dim={args.origin_dim}  compact_dim={args.compact_dim}")
    print(f"{'═'*60}\n")

    if args.mock:
        state_dict = build_mock_state_dict(n_layers=args.n_layers)
    elif args.model_path:
        state_dict = splitter.load_state_dict(args.model_path)
    elif args.model_name:
        try:
            from transformers import AutoModel
            import torch
            print(f"  从 Hub 加载模型: {args.model_name}")
            model = AutoModel.from_pretrained(args.model_name)
            state_dict = {k: v.numpy() for k, v in model.state_dict().items()}
        except ImportError:
            print("  transformers 未安装，切换为 mock 模式")
            state_dict = build_mock_state_dict()
    else:
        print("  未指定输入，使用 mock 模式（--mock）")
        state_dict = build_mock_state_dict()

    splitter.split(state_dict)
    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.2f}s")
    print(f"  输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
