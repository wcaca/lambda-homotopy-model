"""
dynamic_t_scheduler.py — 动态t调度器（增量新增，原有文件零修改）
======================================================================
架构位置：
  继承 MultiTScheduler（三级t）
  在每次 forward() 前，由 HardwareMonitor 采集压力 →
  交给 TPolicy 计算目标 t → 自动调用 set_global_t() 更新

三层调度关系：
  HardwareMonitor  →  压力信号 [0,1]
        ↓
  TPolicy.suggest()  →  目标 global_t
        ↓
  MultiTScheduler.set_global_t()  →  三级t引擎重新分发

特性：
  - 完全向后兼容：不使用动态调度时，行为与 MultiTScheduler 完全一致
  - 可随时暂停/恢复自动调度（pause/resume）
  - 支持手动 override：临时锁定 t 值，不受自动调度影响
  - 调度日志：记录每次 t 变化的原因（压力值、策略、变化量）

λ函数单元 + 同伦形变大模型系统 · 动态t调度层

用法：
    python dynamic_t_scheduler.py
"""

from __future__ import annotations
import os, sys, time
import numpy as np
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from multi_t_scheduler import MultiTScheduler
from hardware_monitor import HardwareMonitor, HardwareSnapshot
from t_policy import TPolicy, AdaptivePolicy, make_policy


class DynamicTScheduler(MultiTScheduler):
    """
    动态 t 调度器
    ─────────────────────────────────────────────────────────────────
    在 MultiTScheduler 基础上增量添加：
      1. HardwareMonitor：每次 forward 后更新压力快照
      2. TPolicy：根据压力建议新的 global_t
      3. 自动调度开关（auto_dispatch）
      4. 手动 override 锁定
      5. 调度历史日志
    ─────────────────────────────────────────────────────────────────
    继承链：
      ModelScheduler → MultiTScheduler → DynamicTScheduler
    """

    def __init__(
        self,
        model_root:      str,
        policy:          TPolicy        = None,
        monitor:         HardwareMonitor = None,
        auto_dispatch:   bool           = True,
        dispatch_every:  int            = 1,     # 每隔多少次 forward 调度一次
        t_init:          float          = 0.0,
    ):
        super().__init__(model_root)
        self.policy         = policy or AdaptivePolicy()
        self.monitor        = monitor or HardwareMonitor()
        self.auto_dispatch  = auto_dispatch
        self.dispatch_every = max(1, dispatch_every)
        self._call_count    = 0
        self._override_t: Optional[float] = None   # 手动锁定值（None=不锁定）
        self._log: list[dict] = []                  # 调度日志

        self.set_global_t(t_init)
        print(f"  [DynamicT] 调度器就绪: policy={self.policy}  "
              f"auto={auto_dispatch}  every={dispatch_every}")

    # ── 手动控制 ────────────────────────────────────────────────────
    def pause(self):
        """暂停自动调度（t 保持当前值不变）"""
        self.auto_dispatch = False
        print(f"  [DynamicT] 自动调度已暂停  current_t={self.global_t:.3f}")

    def resume(self):
        """恢复自动调度"""
        self.auto_dispatch = True
        self.policy.reset()
        print(f"  [DynamicT] 自动调度已恢复  current_t={self.global_t:.3f}")

    def lock_t(self, t: float):
        """锁定 t 值，忽略自动调度（直到 unlock_t()）"""
        self._override_t = max(0.0, min(1.0, t))
        super().set_global_t(self._override_t)
        print(f"  [DynamicT] t 已锁定为 {self._override_t:.3f}")

    def unlock_t(self):
        """解除锁定，恢复自动调度"""
        self._override_t = None
        print(f"  [DynamicT] t 锁定已解除")

    def set_policy(self, policy: TPolicy):
        """运行时切换调度策略"""
        self.policy = policy
        policy.reset()
        print(f"  [DynamicT] 策略切换为: {policy}")

    # ── 核心 forward（在父类基础上包裹调度逻辑）────────────────────
    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        动态 t 前向推理
        流程：
          1. 计时开始
          2. 调用父类 forward（三级t注入推理）
          3. 计时结束 → 注入延迟到 monitor
          4. 满足调度周期时：采样压力 → 策略建议 → 更新 global_t
        """
        t_start = time.perf_counter()
        output  = super().forward(input_vec)        # 父类完整推理（三级t）
        latency = (time.perf_counter() - t_start) * 1000  # ms

        self._call_count += 1
        self.monitor.inject_latency(latency)

        # 满足调度周期且未锁定
        if (self.auto_dispatch
                and self._override_t is None
                and self._call_count % self.dispatch_every == 0):
            self._dispatch()

        return output

    def _dispatch(self):
        """执行一次 t 调度：采样 → 策略建议 → 更新"""
        snap       = self.monitor.sample()
        pressure   = snap.composite_pressure
        old_t      = self.global_t
        new_t      = self.policy.suggest(pressure, old_t)
        delta      = new_t - old_t

        if abs(delta) > 1e-6:
            super().set_global_t(new_t)   # 调用 MultiTScheduler.set_global_t，同步三级引擎

        # 记录日志
        entry = {
            "call":       self._call_count,
            "time":       snap.timestamp,
            "pressure":   round(pressure, 4),
            "cpu%":       round(snap.cpu_percent, 1),
            "mem%":       round(snap.mem_percent, 1),
            "latency_ms": round(snap.latency_ms, 2),
            "old_t":      round(old_t, 4),
            "new_t":      round(new_t, 4),
            "delta_t":    round(delta, 4),
            "policy":     self.policy.name,
        }
        self._log.append(entry)

    # ── 日志与统计 ──────────────────────────────────────────────────
    def dispatch_log(self, last_n: int = 10) -> list[dict]:
        """返回最近 n 条调度日志"""
        return self._log[-last_n:]

    def dispatch_stats(self) -> dict:
        """返回调度统计摘要"""
        if not self._log:
            return {"dispatches": 0, "policy": self.policy.name}
        ts = [e["new_t"]    for e in self._log]
        ps = [e["pressure"] for e in self._log]
        ds = [e["delta_t"]  for e in self._log]
        return {
            "dispatches":    len(self._log),
            "policy":        self.policy.name,
            "total_calls":   self._call_count,
            "t_min":         round(min(ts),  4),
            "t_max":         round(max(ts),  4),
            "t_mean":        round(sum(ts) / len(ts), 4),
            "pressure_mean": round(sum(ps) / len(ps), 4),
            "delta_mean":    round(sum(abs(d) for d in ds) / len(ds), 6),
            "auto":          self.auto_dispatch,
            "override_t":    self._override_t,
        }

    def print_log(self, last_n: int = 10):
        """格式化打印最近调度日志"""
        logs = self.dispatch_log(last_n)
        if not logs:
            print("  [DynamicT] 暂无调度记录")
            return
        print(f"\n  {'#':>4}  {'压力':>6}  {'CPU%':>5}  "
              f"{'延迟ms':>7}  {'old_t':>6}  {'new_t':>6}  {'Δt':>7}  策略")
        print(f"  {'─'*4}  {'─'*6}  {'─'*5}  "
              f"{'─'*7}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*10}")
        for e in logs:
            arrow = "↑" if e["delta_t"] > 0.001 else ("↓" if e["delta_t"] < -0.001 else "─")
            print(f"  {e['call']:>4}  {e['pressure']:>6.4f}  {e['cpu%']:>5.1f}  "
                  f"{e['latency_ms']:>7.1f}  {e['old_t']:>6.4f}  "
                  f"{e['new_t']:>6.4f}  {e['delta_t']:>+7.4f}{arrow}  {e['policy']}")


# ════════════════════════════════════════════════════════════════════════
# 主程序（冒烟测试）
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MODEL_ROOT = os.path.join(script_dir, "model_root")
    SEP = "═" * 56

    print(f"\n{SEP}")
    print("  DynamicTScheduler 冒烟测试")
    print(SEP)

    scheduler = DynamicTScheduler(
        model_root     = MODEL_ROOT,
        policy         = make_policy("adaptive", target_pressure=0.5),
        auto_dispatch  = True,
        dispatch_every = 1,
        t_init         = 0.0,
    )

    np.random.seed(0)
    x = np.random.randn(4096).astype(np.float32)

    print("\n  连续推理 8 次，观察 t 自动调整：")
    for i in range(8):
        out = scheduler.forward(x)
        print(f"    [{i+1}] t={scheduler.global_t:.4f}  "
              f"norm={np.linalg.norm(out):.4f}")

    scheduler.print_log()
    print(f"\n  统计: {scheduler.dispatch_stats()}")
    print(f"\n{SEP}")
    print("  ✓ 冒烟测试通过")
    print(SEP)
