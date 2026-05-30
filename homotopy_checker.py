"""
homotopy_checker.py — 同伦等价性检测（增量新增，原有文件零修改）
======================================================================
核心问题：
  对于同一输入 x，当 t 从 0 连续变化至 1，输出路径是否保持"同伦连续"？
  即：是否存在硬截断、拓扑撕裂、语义跳变？

检测维度：
  H01  连续性检测：相邻 t 步长输出的余弦相似度（骤降=撕裂）
  H02  单调性检测：输出范数是否随 t 单调变化（非单调=可能存在奇点）
  H03  边界完整性：t=0 和 t=1 两端输出与中间任意点的相似度下界
  H04  线性插值误差：f(lerp(t)) 是否约等于 lerp(f(t)) (同伦等价性核心)
  H05  语义稳定性：8维指纹在 t 变化时的余弦相似度分布（路由稳定性）

输出：每条检测结果 + 量化指标 + 通过阈值

λ函数单元 + 同伦形变大模型系统 · 同伦等价性检测层
"""

from __future__ import annotations
import os, sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from topology_validator import CheckResult, ValidationReport, PASS, WARN, FAIL


# ── 检测阈值（可调参） ──────────────────────────────────────────────────
CONTINUITY_WARN_THRESHOLD  = 0.80   # 相邻步余弦相似度低于此值 → WARN
CONTINUITY_FAIL_THRESHOLD  = 0.50   # 低于此值 → FAIL（发生撕裂）
BOUNDARY_MIN_SIMILARITY    = 0.10   # t=0/1 与中间点最小相似度
LINEARITY_WARN_THRESHOLD   = 0.30   # 线性插值误差相对比例 WARN
LINEARITY_FAIL_THRESHOLD   = 0.70   # 线性插值误差相对比例 FAIL


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    n   = min(len(a_f), len(b_f))
    a_f, b_f = a_f[:n], b_f[:n]
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-8
    return float(np.dot(a_f, b_f) / denom)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """相对 L2 误差：||a - b|| / (||a|| + 1e-8)"""
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    n   = min(len(a_f), len(b_f))
    return float(np.linalg.norm(a_f[:n] - b_f[:n]) / (np.linalg.norm(a_f[:n]) + 1e-8))


# ══════════════════════════════════════════════════════════════════════
# 同伦检测器
# ══════════════════════════════════════════════════════════════════════
class HomotopyChecker:
    """
    同伦等价性检测器
    接受一个可调用的 forward(x, t) 函数，对给定输入做全面同伦检测。

    用法：
        from model_scheduler import ModelScheduler
        scheduler = ModelScheduler("./model_root")
        checker = HomotopyChecker(scheduler.forward, scheduler.set_global_t)

        report = checker.check(input_vec, n_steps=20)
        report.print()
    """

    def __init__(
        self,
        forward_fn,      # callable(x: np.ndarray) -> np.ndarray
        set_t_fn,        # callable(t: float) -> None
        name: str = "model",
    ):
        self.forward_fn = forward_fn
        self.set_t_fn   = set_t_fn
        self.name       = name

    def _run_t_sweep(
        self,
        x:       np.ndarray,
        t_vals:  list[float],
    ) -> list[np.ndarray]:
        """在给定 t 序列上收集输出向量"""
        outputs = []
        for t in t_vals:
            self.set_t_fn(t)
            out = self.forward_fn(x.copy())
            outputs.append(out.astype(np.float32))
        return outputs

    def check(
        self,
        input_vec: np.ndarray,
        n_steps:   int   = 20,
        t_min:     float = 0.0,
        t_max:     float = 1.0,
    ) -> ValidationReport:
        """
        执行全套同伦检测
        :param input_vec: 输入向量
        :param n_steps:   t 均匀采样步数（越多越精细，越慢）
        """
        report = ValidationReport(model_root=self.name)
        t_vals  = list(np.linspace(t_min, t_max, n_steps))
        outputs = self._run_t_sweep(input_vec, t_vals)

        self._check_h01(report, t_vals, outputs)
        self._check_h02(report, t_vals, outputs)
        self._check_h03(report, t_vals, outputs)
        self._check_h04(report, input_vec, t_vals)
        self._check_h05(report, input_vec, t_vals)

        # 恢复 t=0
        self.set_t_fn(0.0)
        return report

    # ── H01 连续性检测 ──────────────────────────────────────────────
    def _check_h01(
        self,
        report:  ValidationReport,
        t_vals:  list[float],
        outputs: list[np.ndarray],
    ):
        """相邻 t 步输出的余弦相似度，骤降=形变不连续"""
        sims = [
            cosine_sim(outputs[i], outputs[i+1])
            for i in range(len(outputs) - 1)
        ]
        min_sim  = min(sims)
        min_idx  = sims.index(min_sim)
        min_t    = t_vals[min_idx]
        mean_sim = sum(sims) / len(sims)

        detail = (f"相邻步最小余弦相似度={min_sim:.4f} "
                  f"(发生在 t≈{min_t:.3f})  均值={mean_sim:.4f}")

        if min_sim < CONTINUITY_FAIL_THRESHOLD:
            report.add("H01", FAIL,
                        f"形变路径存在不连续跳变（相似度={min_sim:.4f}）",
                        detail)
        elif min_sim < CONTINUITY_WARN_THRESHOLD:
            report.add("H01", WARN,
                        f"形变路径存在轻微不连续（相似度={min_sim:.4f}）",
                        detail)
        else:
            report.add("H01", PASS,
                        f"形变路径连续（最小相似度={min_sim:.4f}）",
                        detail)

    # ── H02 单调性检测 ──────────────────────────────────────────────
    def _check_h02(
        self,
        report:  ValidationReport,
        t_vals:  list[float],
        outputs: list[np.ndarray],
    ):
        """输出范数随 t 变化是否单调（不要求严格单调，但不应剧烈震荡）"""
        norms = [np.linalg.norm(o) for o in outputs]

        # 计算符号翻转次数（范数增减方向改变）
        diffs  = [norms[i+1] - norms[i] for i in range(len(norms)-1)]
        flips  = sum(
            1 for i in range(len(diffs)-1)
            if diffs[i] * diffs[i+1] < -1e-4   # 方向发生实质性翻转
        )
        flip_rate = flips / max(1, len(diffs))

        norm_range = max(norms) - min(norms)
        detail = (f"范数区间=[{min(norms):.4f}, {max(norms):.4f}]  "
                  f"方向翻转次数={flips}/{len(diffs)}  翻转率={flip_rate:.2%}")

        if flip_rate > 0.30:
            report.add("H02", FAIL,
                        f"输出范数震荡严重（翻转率={flip_rate:.2%}）",
                        detail)
        elif flip_rate > 0.10:
            report.add("H02", WARN,
                        f"输出范数存在轻微震荡（翻转率={flip_rate:.2%}）",
                        detail)
        else:
            report.add("H02", PASS,
                        f"输出范数变化平滑（翻转率={flip_rate:.2%}）",
                        detail)

    # ── H03 边界完整性 ──────────────────────────────────────────────
    def _check_h03(
        self,
        report:  ValidationReport,
        t_vals:  list[float],
        outputs: list[np.ndarray],
    ):
        """t=0 和 t=1 两端输出，与路径中点的相似度不应过低"""
        out_0   = outputs[0]
        out_1   = outputs[-1]
        out_mid = outputs[len(outputs) // 2]

        sim_0_mid = cosine_sim(out_0, out_mid)
        sim_1_mid = cosine_sim(out_1, out_mid)
        sim_0_1   = cosine_sim(out_0, out_1)

        detail = (f"cos(t=0, mid)={sim_0_mid:.4f}  "
                  f"cos(t=1, mid)={sim_1_mid:.4f}  "
                  f"cos(t=0, t=1)={sim_0_1:.4f}")

        if min(sim_0_mid, sim_1_mid) < BOUNDARY_MIN_SIMILARITY:
            report.add("H03", WARN,
                        "边界与中点语义差距较大（可能压缩过激）",
                        detail)
        else:
            report.add("H03", PASS,
                        "边界与路径中点语义相关性正常",
                        detail)

    # ── H04 线性插值误差（同伦等价性核心）──────────────────────────
    def _check_h04(
        self,
        report:    ValidationReport,
        input_vec: np.ndarray,
        t_vals:    list[float],
    ):
        """
        核心检测：
          理想同伦：f(lerp(t, x0, x1)) ≈ lerp(t, f(x0), f(x1))
          实际检测：对单个输入，验证 f 在 t 轴上的线性近似误差

          做法：
            - 取 t=0 输出 y0，t=1 输出 y1
            - 对中间任意 t，计算线性插值预测 y_pred = (1-t)*y0 + t*y1
            - 与实际 y_actual 对比，误差越小 = 形变越接近线性同伦
        """
        self.set_t_fn(0.0)
        y0 = self.forward_fn(input_vec.copy()).astype(np.float32)
        self.set_t_fn(1.0)
        y1 = self.forward_fn(input_vec.copy()).astype(np.float32)

        # 对齐维度
        n = min(len(y0.flatten()), len(y1.flatten()))
        y0_f = y0.flatten()[:n]
        y1_f = y1.flatten()[:n]

        errors = []
        mid_t_vals = [t for t in t_vals if 0.05 < t < 0.95]
        for t in mid_t_vals[:8]:   # 最多验证8个中间点，控制耗时
            y_pred = (1 - t) * y0_f + t * y1_f
            self.set_t_fn(t)
            y_actual = self.forward_fn(input_vec.copy()).astype(np.float32).flatten()[:n]
            err = relative_error(y_pred, y_actual)
            errors.append((t, err))

        if not errors:
            report.add("H04", WARN, "中间 t 点不足，无法评估线性插值误差")
            return

        mean_err = sum(e for _, e in errors) / len(errors)
        max_err  = max(e for _, e in errors)
        max_t    = max(errors, key=lambda x: x[1])[0]
        detail   = (f"平均相对误差={mean_err:.4f}  "
                    f"最大误差={max_err:.4f}@t={max_t:.2f}  "
                    f"(误差越小=越接近线性同伦)")

        if mean_err > LINEARITY_FAIL_THRESHOLD:
            report.add("H04", FAIL,
                        f"形变路径严重非线性（误差={mean_err:.4f}）",
                        detail)
        elif mean_err > LINEARITY_WARN_THRESHOLD:
            report.add("H04", WARN,
                        f"形变路径存在非线性（误差={mean_err:.4f}）",
                        detail)
        else:
            report.add("H04", PASS,
                        f"形变路径近似线性同伦（误差={mean_err:.4f}）",
                        detail)

    # ── H05 语义稳定性（指纹视角）──────────────────────────────────
    def _check_h05(
        self,
        report:    ValidationReport,
        input_vec: np.ndarray,
        t_vals:    list[float],
    ):
        """
        检测：输出向量的 8 维指纹摘要在 t 变化时的稳定性
        （用输出向量的分段 RMS 模拟"语义指纹"）
        """
        from sparse_router import extract_query_finger

        fingers = []
        for t in t_vals[::4]:    # 每隔4步采一次，节省计算
            self.set_t_fn(t)
            out    = self.forward_fn(input_vec.copy()).astype(np.float32)
            finger = extract_query_finger(out)
            fingers.append(finger.astype(np.float32))

        if len(fingers) < 2:
            report.add("H05", WARN, "采样点不足，跳过语义稳定性检测")
            return

        sims = [cosine_sim(fingers[i], fingers[i+1])
                for i in range(len(fingers)-1)]
        min_sim  = min(sims)
        mean_sim = sum(sims) / len(sims)
        detail   = (f"指纹余弦相似度: 最小={min_sim:.4f}  均值={mean_sim:.4f}  "
                    f"采样点={len(fingers)}")

        if min_sim < 0.5:
            report.add("H05", WARN,
                        f"形变过程中语义路由指纹变化较大（最小相似度={min_sim:.4f}）",
                        detail)
        else:
            report.add("H05", PASS,
                        f"语义路由指纹在 t 变化时保持稳定（最小相似度={min_sim:.4f}）",
                        detail)
