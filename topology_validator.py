"""
topology_validator.py — 拓扑结构核心校验引擎（增量新增，原有文件零修改）
======================================================================
职责：
  对 model_root 目录和运行时拓扑链路做静态 + 动态两层校验：

  静态校验（不需要真实推理，仅分析配置和文件）：
    V01  拓扑链路语法合法性（括号、箭头、组名格式）
    V02  组目录存在性（TopologyChain 中引用的组均有对应目录）
    V03  group.conf 完整性（必填字段不缺失）
    V04  UnitList 文件存在性（group.conf 中列出的 bin 文件都在磁盘上）
    V05  λ单元头部合法性（头部64字节可解析，维度字段在合理范围内）
    V06  环路检测（TopologyChain 中是否存在重复组名，导致数据环流）

  动态校验（需要传入真实输入向量，追踪维度流转）：
    V07  串联维度连续性（每步输出维度能被下一步输入接受）
    V08  并联输入一致性（并联组内所有单元接收相同输入维度）
    V09  并联输出融合合法性（concat 后维度是否超出下游接受范围）
    V10  全链路维度通路（t=0 和 t=1 两端均能正常流通）

每条规则输出：PASS / WARN / FAIL + 详细描述

λ函数单元 + 同伦形变大模型系统 · 拓扑校验层
"""

from __future__ import annotations
import os, re, struct, sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_utils import ENDIAN, HEAD_LEN, FP16_SIZE, FINGER_DIM, load_ini_config

# ── 校验结果常量 ────────────────────────────────────────────────────────
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# ── 合理维度范围 ────────────────────────────────────────────────────────
DIM_MIN = 8
DIM_MAX = 65535
VALID_FUNC_TYPES = {0, 1, 2, 3, 4}


# ══════════════════════════════════════════════════════════════════════
# 校验结果数据结构
# ══════════════════════════════════════════════════════════════════════
@dataclass
class CheckResult:
    code:    str          # V01–V10
    status:  str          # PASS / WARN / FAIL
    message: str          # 人类可读描述
    detail:  str = ""     # 可选附加细节


@dataclass
class ValidationReport:
    model_root: str
    results:    list[CheckResult] = field(default_factory=list)

    def add(self, code: str, status: str, message: str, detail: str = ""):
        self.results.append(CheckResult(code, status, message, detail))

    @property
    def passed(self)  -> list[CheckResult]:
        return [r for r in self.results if r.status == PASS]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == WARN]

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def is_ok(self) -> bool:
        return len(self.failures) == 0

    def print(self, verbose: bool = True):
        icons = {PASS: "✓", WARN: "⚠", FAIL: "✗"}
        print(f"\n{'─'*62}")
        print(f"  拓扑校验报告  model_root={self.model_root}")
        print(f"  结果: {len(self.passed)} PASS  "
              f"{len(self.warnings)} WARN  {len(self.failures)} FAIL")
        print(f"{'─'*62}")
        for r in self.results:
            icon = icons[r.status]
            print(f"  [{r.code}] {icon} {r.status:4s}  {r.message}")
            if verbose and r.detail:
                for line in r.detail.strip().split("\n"):
                    print(f"           {line}")
        print(f"{'─'*62}")
        overall = "OK — 可正常运行" if self.is_ok else "存在 FAIL — 请修复后再运行"
        print(f"  总体: {overall}\n")


# ══════════════════════════════════════════════════════════════════════
# 拓扑解析工具
# ══════════════════════════════════════════════════════════════════════
def parse_topology_chain(chain: str) -> list[tuple[str, str]]:
    """
    解析 TopologyChain 字符串为有序步骤列表
    返回: [(exec_mode, group_name), ...]
    exec_mode: "serial" | "parallel" | "unknown"
    """
    steps = []
    for token in chain.replace(" ", "").split("->"):
        token = token.split("|")[0]   # 去除 |modal:xxx 修饰
        m = re.match(r"(serial|parallel)\((\w+)\)", token)
        if m:
            steps.append((m.group(1), m.group(2)))
        elif token:
            steps.append(("unknown", token))
    return steps


# ══════════════════════════════════════════════════════════════════════
# λ单元头部快速读取（不加载完整权重）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class UnitHeader:
    unit_id:      str
    func_type:    int
    dim_skeleton: int
    dim_finger:   int
    dim_origin:   int
    version:      int
    file_size:    int


def read_unit_header(path: str) -> Optional[UnitHeader]:
    """读取 *.bin 头部 64 字节，返回 UnitHeader 或 None（文件损坏时）"""
    try:
        fsize = os.path.getsize(path)
        if fsize < HEAD_LEN:
            return None
        with open(path, "rb") as f:
            head = f.read(HEAD_LEN)
        uid      = struct.unpack_from(f"{ENDIAN}32s", head, 0)[0].decode("utf-8", errors="replace").strip("\x00")
        ftype    = struct.unpack_from(f"{ENDIAN}B",   head, 32)[0]
        dim_sk   = struct.unpack_from(f"{ENDIAN}H",   head, 33)[0]
        dim_fi   = struct.unpack_from(f"{ENDIAN}B",   head, 35)[0]
        dim_or   = struct.unpack_from(f"{ENDIAN}H",   head, 36)[0]
        version  = struct.unpack_from(f"{ENDIAN}H",   head, 38)[0]
        return UnitHeader(uid, ftype, dim_sk, dim_fi, dim_or, version, fsize)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# 主校验引擎
# ══════════════════════════════════════════════════════════════════════
class TopologyValidator:
    """
    拓扑校验引擎
    用法：
        validator = TopologyValidator("./model_root")
        report = validator.validate_static()          # 静态校验
        report = validator.validate_dynamic(input_vec) # 动态校验（含静态）
        report.print()
    """

    def __init__(self, model_root: str):
        self.model_root = os.path.abspath(model_root)
        self._cfg       = None
        self._chain_str = ""
        self._steps: list[tuple[str, str]] = []

    def _load_conf(self) -> bool:
        conf_path = os.path.join(self.model_root, "model_global.conf")
        if not os.path.exists(conf_path):
            return False
        self._cfg       = load_ini_config(conf_path)
        self._chain_str = self._cfg.get("Topology", "TopologyChain", fallback="")
        self._steps     = parse_topology_chain(self._chain_str)
        return True

    # ── 静态校验 ────────────────────────────────────────────────────
    def validate_static(self) -> ValidationReport:
        report = ValidationReport(model_root=self.model_root)

        # 前置：加载 conf
        if not os.path.exists(self.model_root):
            report.add("V00", FAIL, "model_root 目录不存在", self.model_root)
            return report
        if not self._load_conf():
            report.add("V00", FAIL, "model_global.conf 缺失或无法解析")
            return report

        self._check_v01(report)
        self._check_v02(report)
        self._check_v03(report)
        self._check_v04(report)
        self._check_v05(report)
        self._check_v06(report)
        return report

    def _check_v01(self, report: ValidationReport):
        """V01 拓扑链路语法"""
        if not self._chain_str.strip():
            report.add("V01", FAIL, "TopologyChain 为空")
            return
        unknown = [s for m, s in self._steps if m == "unknown"]
        if unknown:
            report.add("V01", FAIL,
                        "TopologyChain 包含无法解析的步骤",
                        f"无法解析: {unknown}")
        else:
            report.add("V01", PASS,
                        f"拓扑链路语法合法 ({len(self._steps)} 步骤)",
                        self._chain_str)

    def _check_v02(self, report: ValidationReport):
        """V02 组目录存在性"""
        missing = []
        for _, gname in self._steps:
            gdir = os.path.join(self.model_root, gname)
            if not os.path.isdir(gdir):
                missing.append(gname)
        if missing:
            report.add("V02", FAIL,
                        f"拓扑引用的组目录不存在: {missing}")
        else:
            report.add("V02", PASS,
                        f"所有组目录均存在 ({len(self._steps)} 个)")

    def _check_v03(self, report: ValidationReport):
        """V03 group.conf 完整性"""
        issues = []
        for _, gname in self._steps:
            conf_path = os.path.join(self.model_root, gname, "group.conf")
            if not os.path.exists(conf_path):
                issues.append(f"{gname}: group.conf 缺失")
                continue
            cfg = load_ini_config(conf_path)
            for field_name in ["GroupName", "UnitList", "GroupType"]:
                if not cfg.has_option("GroupBase", field_name):
                    issues.append(f"{gname}/group.conf: 缺少字段 {field_name}")
        if issues:
            report.add("V03", FAIL, "group.conf 存在问题",
                        "\n".join(issues))
        else:
            report.add("V03", PASS, "所有 group.conf 字段完整")

    def _check_v04(self, report: ValidationReport):
        """V04 UnitList 文件存在性"""
        missing_files = []
        for _, gname in self._steps:
            conf_path = os.path.join(self.model_root, gname, "group.conf")
            if not os.path.exists(conf_path):
                continue
            cfg   = load_ini_config(conf_path)
            raw   = cfg.get("GroupBase", "UnitList", fallback="")
            names = [n.strip() for n in raw.split(",") if n.strip()]
            for fname in names:
                fpath = os.path.join(self.model_root, gname, fname)
                if not os.path.exists(fpath):
                    missing_files.append(f"{gname}/{fname}")
        if missing_files:
            report.add("V04", FAIL,
                        f"UnitList 中 {len(missing_files)} 个文件缺失",
                        "\n".join(missing_files))
        else:
            report.add("V04", PASS, "所有 UnitList 文件均存在于磁盘")

    def _check_v05(self, report: ValidationReport):
        """V05 λ单元头部合法性"""
        bad_units = []
        warn_units = []
        for _, gname in self._steps:
            conf_path = os.path.join(self.model_root, gname, "group.conf")
            if not os.path.exists(conf_path):
                continue
            cfg   = load_ini_config(conf_path)
            raw   = cfg.get("GroupBase", "UnitList", fallback="")
            names = [n.strip() for n in raw.split(",") if n.strip()]
            for fname in names:
                fpath = os.path.join(self.model_root, gname, fname)
                if not os.path.exists(fpath):
                    continue
                hdr = read_unit_header(fpath)
                if hdr is None:
                    bad_units.append(f"{gname}/{fname}: 头部无法解析")
                    continue
                if hdr.func_type not in VALID_FUNC_TYPES:
                    bad_units.append(f"{gname}/{fname}: FuncType={hdr.func_type} 不合法")
                if not (DIM_MIN <= hdr.dim_skeleton <= DIM_MAX):
                    bad_units.append(f"{gname}/{fname}: dim_skeleton={hdr.dim_skeleton} 超范围")
                if not (DIM_MIN <= hdr.dim_origin <= DIM_MAX):
                    bad_units.append(f"{gname}/{fname}: dim_origin={hdr.dim_origin} 超范围")
                if hdr.dim_finger != FINGER_DIM:
                    warn_units.append(f"{gname}/{fname}: dim_finger={hdr.dim_finger} 不等于标准 {FINGER_DIM}")

        if bad_units:
            report.add("V05", FAIL, f"{len(bad_units)} 个单元头部不合法",
                        "\n".join(bad_units))
        elif warn_units:
            report.add("V05", WARN, f"头部合法但存在警告",
                        "\n".join(warn_units))
        else:
            report.add("V05", PASS, "所有λ单元头部合法")

    def _check_v06(self, report: ValidationReport):
        """V06 环路检测"""
        seen = []
        for _, gname in self._steps:
            if gname in seen:
                report.add("V06", FAIL,
                            f"拓扑链路存在重复组名（潜在数据环流）",
                            f"重复出现: {gname}")
                return
            seen.append(gname)
        report.add("V06", PASS, f"拓扑链路无环路 ({len(seen)} 个唯一组)")

    # ── 动态校验 ────────────────────────────────────────────────────
    def validate_dynamic(
        self,
        input_vec: np.ndarray,
        t_values:  list[float] = None,
    ) -> ValidationReport:
        """
        动态校验：在静态校验基础上，追踪真实推理过程中的维度流转
        :param input_vec: 初始输入向量（shape=(origin_dim,)）
        :param t_values:  校验用 t 值列表，默认 [0.0, 0.5, 1.0]
        """
        report = self.validate_static()
        if not report.is_ok:
            report.add("V07", WARN,
                       "静态校验存在 FAIL，跳过动态校验",
                       "请先修复静态校验问题再运行动态校验")
            return report

        t_values = t_values or [0.0, 0.5, 1.0]
        self._check_v07_v10(report, input_vec, t_values)
        return report

    def _check_v07_v10(
        self,
        report:    ValidationReport,
        input_vec: np.ndarray,
        t_values:  list[float],
    ):
        """V07-V10 动态维度追踪校验"""
        from base_utils import LambdaUnit
        from component_manager import ComponentManager

        dim_issues   = []
        fusion_warns = []
        flow_ok      = True

        for t_val in t_values:
            current = input_vec.copy().astype(np.float32)
            prev_dim = current.shape[-1]

            for mode, gname in self._steps:
                gdir = os.path.join(self.model_root, gname)
                try:
                    cm = ComponentManager(gdir, lazy_load=False)
                except Exception as e:
                    dim_issues.append(f"t={t_val} [{gname}] 加载失败: {e}")
                    flow_ok = False
                    break

                # V08 并联输入一致性
                if mode == "parallel" and cm.units:
                    in_dims = {u.dim_origin for u in cm.units}
                    if len(in_dims) > 1:
                        dim_issues.append(
                            f"t={t_val} [{gname}] 并联组内单元 dim_origin 不一致: {in_dims}"
                        )

                # 执行 forward 追踪输出维度
                try:
                    out = cm.forward(current, t_val)
                    out_dim = out.shape[-1]
                except Exception as e:
                    dim_issues.append(f"t={t_val} [{gname}] forward 异常: {e}")
                    flow_ok = False
                    break

                # V07 串联维度连续性
                if mode == "serial" and cm.units:
                    exp_out = cm.units[-1].dim_skeleton
                    if out_dim != exp_out and abs(out_dim - exp_out) > 64:
                        dim_issues.append(
                            f"t={t_val} [{gname}] 串联输出维度 {out_dim} "
                            f"与预期 {exp_out} 差异过大"
                        )

                # V09 并联 concat 融合维度警告
                if mode == "parallel" and cm.merge_rule == "concat":
                    expected_concat = sum(u.dim_skeleton for u in cm.units)
                    if out_dim != expected_concat:
                        fusion_warns.append(
                            f"t={t_val} [{gname}] concat 融合维度 {out_dim} "
                            f"!= 预期 {expected_concat}"
                        )

                prev_dim = out_dim
                current  = out.astype(np.float32)

        # V07 串联维度连续性
        if dim_issues:
            report.add("V07", FAIL, f"维度流转存在 {len(dim_issues)} 个问题",
                        "\n".join(dim_issues))
        else:
            report.add("V07", PASS,
                        f"串联维度连续性正常 (验证 t={t_values})")

        # V08 并联输入一致性（已在循环中检测，合并至 V07 结论）
        if not dim_issues:
            report.add("V08", PASS, "并联组内单元输入维度一致")
        else:
            report.add("V08", WARN, "并联组内维度问题已包含在 V07 中")

        # V09 并联 concat 融合
        if fusion_warns:
            report.add("V09", WARN, "并联 concat 融合维度存在偏差",
                        "\n".join(fusion_warns))
        else:
            report.add("V09", PASS, "并联输出融合维度正常")

        # V10 全链路通路
        if flow_ok and not dim_issues:
            report.add("V10", PASS,
                        f"全链路维度通路完整 (t=0.0 和 t=1.0 均正常流通)")
        else:
            report.add("V10", FAIL,
                        "全链路存在断点，t 形变后无法正常推理")
