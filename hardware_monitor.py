"""
hardware_monitor.py — 硬件负载实时采集层（增量新增，原有文件零修改）
======================================================================
职责：
  采集 CPU、内存、推理延迟三路信号，归一化为 [0,1] 压力值，
  供动态 t 调度策略使用。

压力值定义（统一 [0,1]，0=空闲，1=满负荷）：
  cpu_pressure    : CPU 使用率 / 100
  mem_pressure    : 内存使用率 / 100
  latency_pressure: 实测推理延迟 / 目标延迟阈值，超过则截断为 1.0

无 psutil 时自动降级：返回估算值，不中断主流程。

λ函数单元 + 同伦形变大模型系统 · 硬件监控层
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

# ── 可选依赖 ────────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


# ══════════════════════════════════════════════════════════════════════
# 硬件快照数据结构
# ══════════════════════════════════════════════════════════════════════
@dataclass
class HardwareSnapshot:
    """单次采样结果"""
    timestamp:        float = 0.0
    cpu_percent:      float = 0.0   # CPU 使用率 0~100
    mem_percent:      float = 0.0   # 内存使用率 0~100
    mem_available_mb: float = 0.0   # 可用内存 MB
    latency_ms:       float = 0.0   # 最近一次推理耗时 ms（由外部注入）

    # 归一化压力值 [0, 1]
    cpu_pressure:     float = 0.0
    mem_pressure:     float = 0.0
    latency_pressure: float = 0.0

    @property
    def composite_pressure(self) -> float:
        """综合压力（加权平均）：CPU 40% + 内存 35% + 延迟 25%"""
        return (self.cpu_pressure * 0.40
                + self.mem_pressure * 0.35
                + self.latency_pressure * 0.25)


# ══════════════════════════════════════════════════════════════════════
# 硬件监控器
# ══════════════════════════════════════════════════════════════════════
class HardwareMonitor:
    """
    硬件负载实时监控器
    可选后台线程定时采样，也可按需单次采样。

    用法（按需采样）：
        mon = HardwareMonitor()
        snap = mon.sample()
        print(snap.composite_pressure)

    用法（后台持续采样）：
        mon = HardwareMonitor(background=True, interval=1.0)
        mon.start()
        snap = mon.latest()
        mon.stop()
    """

    # 目标阈值（超过则压力=1.0）
    CPU_HIGH_THRESHOLD     = 85.0    # CPU%
    MEM_HIGH_THRESHOLD     = 85.0    # 内存%
    LATENCY_TARGET_MS      = 200.0   # 推理目标延迟 ms

    def __init__(
        self,
        background:  bool  = False,
        interval:    float = 1.0,      # 后台采样间隔（秒）
        history_len: int   = 30,       # 保留历史快照数
    ):
        self.background  = background
        self.interval    = interval
        self._history: deque[HardwareSnapshot] = deque(maxlen=history_len)
        self._latest: Optional[HardwareSnapshot] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 外部注入的推理延迟（由调度器在每次 forward 后更新）
        self._last_latency_ms: float = 0.0

        if not _PSUTIL_OK:
            print("  [HardwareMonitor] psutil 未安装，使用估算模式 "
                  "(pip install psutil 可启用精确采样)")

    # ── 采样 ────────────────────────────────────────────────────────
    def sample(self) -> HardwareSnapshot:
        """单次采样，返回当前硬件快照"""
        snap = HardwareSnapshot(timestamp=time.time())

        if _PSUTIL_OK:
            snap.cpu_percent      = psutil.cpu_percent(interval=0.1)
            mem                   = psutil.virtual_memory()
            snap.mem_percent      = mem.percent
            snap.mem_available_mb = mem.available / (1024 * 1024)
        else:
            # 降级：基于时间戳的伪随机估算（可复现，不依赖随机种子）
            import math
            t = time.time()
            snap.cpu_percent      = 30.0 + 20.0 * abs(math.sin(t * 0.3))
            snap.mem_percent      = 45.0 + 15.0 * abs(math.sin(t * 0.17))
            snap.mem_available_mb = 4096.0 * (1 - snap.mem_percent / 100)

        snap.latency_ms = self._last_latency_ms

        # 归一化为压力值
        snap.cpu_pressure     = min(1.0, snap.cpu_percent     / self.CPU_HIGH_THRESHOLD)
        snap.mem_pressure     = min(1.0, snap.mem_percent     / self.MEM_HIGH_THRESHOLD)
        snap.latency_pressure = min(1.0, snap.latency_ms      / max(1.0, self.LATENCY_TARGET_MS))

        with self._lock:
            self._latest = snap
            self._history.append(snap)

        return snap

    def latest(self) -> HardwareSnapshot:
        """返回最近一次快照（无快照则立即采样）"""
        with self._lock:
            if self._latest is not None:
                return self._latest
        return self.sample()

    def inject_latency(self, latency_ms: float):
        """由调度器注入推理延迟（每次 forward 结束后调用）"""
        self._last_latency_ms = latency_ms

    # ── 平滑压力（历史均值，防抖）──────────────────────────────────
    def smoothed_pressure(self, window: int = 5) -> float:
        """
        取最近 window 次快照的综合压力均值，平滑瞬时抖动
        """
        with self._lock:
            snaps = list(self._history)[-window:]
        if not snaps:
            return self.sample().composite_pressure
        return sum(s.composite_pressure for s in snaps) / len(snaps)

    # ── 后台线程 ────────────────────────────────────────────────────
    def start(self):
        """启动后台定时采样线程"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"  [HardwareMonitor] 后台采样启动 (interval={self.interval}s)")

    def stop(self):
        """停止后台采样线程"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("  [HardwareMonitor] 后台采样已停止")

    def _loop(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    # ── 统计摘要 ────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self._lock:
            snaps = list(self._history)
        if not snaps:
            return {"samples": 0}
        pressures = [s.composite_pressure for s in snaps]
        return {
            "samples":       len(snaps),
            "cpu_avg":       round(sum(s.cpu_percent  for s in snaps) / len(snaps), 1),
            "mem_avg":       round(sum(s.mem_percent  for s in snaps) / len(snaps), 1),
            "latency_avg_ms":round(sum(s.latency_ms   for s in snaps) / len(snaps), 1),
            "pressure_min":  round(min(pressures), 4),
            "pressure_max":  round(max(pressures), 4),
            "pressure_avg":  round(sum(pressures) / len(pressures), 4),
            "psutil_available": _PSUTIL_OK,
        }

    def __repr__(self) -> str:
        snap = self.latest()
        return (f"<HardwareMonitor cpu={snap.cpu_percent:.1f}% "
                f"mem={snap.mem_percent:.1f}% "
                f"pressure={snap.composite_pressure:.3f}>")
