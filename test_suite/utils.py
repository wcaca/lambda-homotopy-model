"""
test_suite/utils.py — 测试工具公共层
报告格式、断言封装、基准 I/O
"""
from __future__ import annotations
import os, sys, time, traceback
import numpy as np

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)

# ── 报告颜色（终端） ─────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

PASS_TAG = f"{GREEN}通过{RESET}"
FAIL_TAG = f"{RED}失败{RESET}"
SKIP_TAG = f"{YELLOW}跳过{RESET}"


class TestResult:
    def __init__(self, name: str):
        self.name    = name
        self.status  = "通过"   # 通过 / 失败 / 跳过
        self.metrics: dict = {}
        self.note:    str  = ""
        self.elapsed: float = 0.0

    def passed(self, **metrics):
        self.status  = "通过"
        self.metrics = metrics
        return self

    def failed(self, reason: str, **metrics):
        self.status  = "失败"
        self.note    = reason
        self.metrics = metrics
        return self

    def skipped(self, reason: str):
        self.status = "跳过"
        self.note   = reason
        return self

    def print(self):
        tag = {"通过": PASS_TAG, "失败": FAIL_TAG, "跳过": SKIP_TAG}[self.status]
        metric_str = "  ".join(f"{k}={v}" for k, v in self.metrics.items())
        note_str   = f"  ↳ {self.note}" if self.note else ""
        print(f"  【{self.name}】 状态: {tag} | {metric_str}{note_str}  ({self.elapsed:.2f}s)")


class TestSuite:
    def __init__(self, name: str):
        self.name    = name
        self.results: list[TestResult] = []

    def run_case(self, name: str, fn) -> TestResult:
        r = TestResult(name)
        t0 = time.perf_counter()
        try:
            fn(r)
        except Exception as e:
            r.failed(f"异常: {e}\n{traceback.format_exc(limit=3)}")
        r.elapsed = time.perf_counter() - t0
        self.results.append(r)
        r.print()
        return r

    def summary(self) -> dict:
        total  = len(self.results)
        passed = sum(1 for r in self.results if r.status == "通过")
        failed = sum(1 for r in self.results if r.status == "失败")
        skip   = sum(1 for r in self.results if r.status == "跳过")
        return {"suite": self.name, "total": total,
                "passed": passed, "failed": failed, "skipped": skip}

    def print_summary(self):
        s = self.summary()
        bar = "─" * 60
        tag = PASS_TAG if s["failed"] == 0 else FAIL_TAG
        print(f"\n{bar}")
        print(f"  [{self.name}] 总计: {s['total']}  "
              f"{GREEN}通过: {s['passed']}{RESET}  "
              f"{RED}失败: {s['failed']}{RESET}  "
              f"{YELLOW}跳过: {s['skipped']}{RESET}  → {tag}")
        print(f"{bar}\n")
        return s["failed"] == 0


# ── 公共断言 ─────────────────────────────────────────────────────────────
def assert_shape(arr: np.ndarray, expected_dim: int, label="输出"):
    assert arr.shape[-1] == expected_dim, \
        f"{label} 维度 {arr.shape[-1]} != 期望 {expected_dim}"

def assert_finite(arr: np.ndarray, label="输出"):
    assert np.all(np.isfinite(arr.astype(np.float32))), \
        f"{label} 含 NaN/Inf"

def assert_changing(vals: list, label="输出"):
    """断言列表中的值不全相同（有变化）"""
    norms = [float(np.linalg.norm(np.array(v).flatten())) for v in vals]
    assert max(norms) - min(norms) > 1e-6, \
        f"{label} 在不同 t 下范数不变（{norms}），形变未生效"

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32).flatten()
    bf = b.astype(np.float32).flatten()
    n  = min(len(af), len(bf))
    d  = np.linalg.norm(af[:n]) * np.linalg.norm(bf[:n]) + 1e-8
    return float(np.dot(af[:n], bf[:n]) / d)

# ── 固定测试向量 ─────────────────────────────────────────────────────────
def fixed_input(dim: int = 4096, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)

MODEL_ROOT = os.path.join(script_dir, "model_root")
