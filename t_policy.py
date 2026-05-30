"""
t_policy.py — 动态t调度策略库（增量新增，原有文件零修改）
======================================================================
职责：
  接收硬件压力信号，输出目标 t 值建议。
  调度器只关心"应该把 t 调到多少"，不关心硬件采集细节。

内置三种策略：
  ConservativePolicy  保守策略：负载高时缓慢提升 t，优先稳定性
  AggressivePolicy    激进策略：负载一高立刻升 t，优先轻量推理
  AdaptivePolicy      自适应策略（推荐）：PID 风格连续调节，带惯性防抖

策略接口（所有策略实现此接口）：
  policy.suggest(pressure: float, current_t: float) -> float

λ函数单元 + 同伦形变大模型系统 · 动态t策略层
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ══════════════════════════════════════════════════════════════════════
# 策略协议（鸭子类型接口约定）
# ══════════════════════════════════════════════════════════════════════
@runtime_checkable
class TPolicy(Protocol):
    """所有策略必须实现的最小接口"""
    def suggest(self, pressure: float, current_t: float) -> float:
        """
        根据当前综合压力和当前 t，建议下一个目标 t 值
        :param pressure:  归一化综合压力 [0,1]
        :param current_t: 当前全局 t [0,1]
        :return:          建议的目标 t [0,1]
        """
        ...

    def reset(self): ...


# ══════════════════════════════════════════════════════════════════════
# 策略1：保守策略
# ══════════════════════════════════════════════════════════════════════
class ConservativePolicy:
    """
    保守策略（适合生产环境、低配设备）
    ─────────────────────────────────────────────────────────────────
    逻辑：
      - 压力低于 low_threshold  → t 缓慢下降（恢复精度）
      - 压力在中间区            → t 保持不变
      - 压力高于 high_threshold → t 缓慢上升（减少计算量）
    速率：每次调用最大变化 step_size，防止剧烈跳变
    """

    def __init__(
        self,
        low_threshold:  float = 0.40,   # 低于此压力 → 降 t
        high_threshold: float = 0.70,   # 高于此压力 → 升 t
        step_size:      float = 0.02,   # 每次最大 t 变化量
        t_min:          float = 0.0,
        t_max:          float = 0.9,    # 保守策略上限 0.9，避免过度压缩
    ):
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.step_size      = step_size
        self.t_min          = t_min
        self.t_max          = t_max
        self.name           = "conservative"

    def suggest(self, pressure: float, current_t: float) -> float:
        if pressure < self.low_threshold:
            target = current_t - self.step_size      # 负载低，降 t 恢复精度
        elif pressure > self.high_threshold:
            target = current_t + self.step_size      # 负载高，升 t 减轻计算
        else:
            target = current_t                        # 中间区，保持不变
        return max(self.t_min, min(self.t_max, target))

    def reset(self): pass

    def __repr__(self):
        return (f"<ConservativePolicy low={self.low_threshold} "
                f"high={self.high_threshold} step={self.step_size}>")


# ══════════════════════════════════════════════════════════════════════
# 策略2：激进策略
# ══════════════════════════════════════════════════════════════════════
class AggressivePolicy:
    """
    激进策略（适合对延迟极敏感、允许精度波动的场景）
    ─────────────────────────────────────────────────────────────────
    逻辑：
      target_t = pressure_to_t_curve(pressure)
      即：压力直接映射为 t，不做惯性平滑
      映射曲线：sigmoid 形，低压力→低 t，高压力→高 t
    """

    def __init__(
        self,
        t_min:      float = 0.0,
        t_max:      float = 1.0,
        steepness:  float = 8.0,    # sigmoid 陡峭程度（越大越突变）
        midpoint:   float = 0.55,   # sigmoid 中点（压力=midpoint 时 t≈0.5）
    ):
        self.t_min     = t_min
        self.t_max     = t_max
        self.steepness = steepness
        self.midpoint  = midpoint
        self.name      = "aggressive"

    def suggest(self, pressure: float, current_t: float) -> float:
        # sigmoid 映射：pressure → [0,1] → 线性缩放至 [t_min, t_max]
        sig    = 1.0 / (1.0 + math.exp(-self.steepness * (pressure - self.midpoint)))
        target = self.t_min + (self.t_max - self.t_min) * sig
        return max(self.t_min, min(self.t_max, target))

    def reset(self): pass

    def __repr__(self):
        return (f"<AggressivePolicy t=[{self.t_min},{self.t_max}] "
                f"steepness={self.steepness} mid={self.midpoint}>")


# ══════════════════════════════════════════════════════════════════════
# 策略3：自适应策略（推荐，PID 风格）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class AdaptivePolicyState:
    """自适应策略内部状态"""
    integral:      float = 0.0   # 积分项（历史偏差累积）
    prev_error:    float = 0.0   # 上次偏差（用于微分项）
    prev_t:        float = 0.0   # 上次建议 t
    prev_time:     float = field(default_factory=time.time)


class AdaptivePolicy:
    """
    自适应策略（推荐，PID 控制风格）
    ─────────────────────────────────────────────────────────────────
    控制目标：将综合压力稳定在 target_pressure 附近
    误差 e = pressure - target_pressure
    输出 Δt = Kp * e + Ki * ∫e·dt + Kd * de/dt

    特性：
      - 带惯性：t 变化平滑，不会因瞬时抖动剧烈跳变
      - 带积分：能消除稳态偏差，长期维持目标压力
      - 带微分：提前感知压力变化趋势，提前调节
      - 死区：压力偏差 < dead_zone 时不调整（防止微小抖动频繁触发）
      - t 变化速率限制（max_delta_per_call）：每次最多变化 0.05
    """

    def __init__(
        self,
        target_pressure: float = 0.55,   # 目标压力（t 围绕此值稳定）
        Kp:              float = 0.15,   # 比例增益
        Ki:              float = 0.03,   # 积分增益
        Kd:              float = 0.05,   # 微分增益
        dead_zone:       float = 0.05,   # 死区（压力偏差小于此值不调整）
        max_delta:       float = 0.05,   # 每次最大 t 变化量
        t_min:           float = 0.0,
        t_max:           float = 1.0,
        integral_limit:  float = 2.0,    # 积分项截断（防止积分饱和）
    ):
        self.target_pressure = target_pressure
        self.Kp              = Kp
        self.Ki              = Ki
        self.Kd              = Kd
        self.dead_zone       = dead_zone
        self.max_delta       = max_delta
        self.t_min           = t_min
        self.t_max           = t_max
        self.integral_limit  = integral_limit
        self.name            = "adaptive"
        self._state          = AdaptivePolicyState(prev_t=0.0, prev_time=time.time())

    def suggest(self, pressure: float, current_t: float) -> float:
        now   = time.time()
        dt    = max(0.001, now - self._state.prev_time)  # 时间步长（s）
        error = pressure - self.target_pressure           # 当前偏差

        # 死区：偏差过小则不调整
        if abs(error) < self.dead_zone:
            self._state.prev_time = now
            self._state.prev_t    = current_t
            return current_t

        # PID 三项
        p_term = self.Kp * error

        self._state.integral += error * dt
        self._state.integral  = max(-self.integral_limit,
                                    min(self.integral_limit, self._state.integral))
        i_term = self.Ki * self._state.integral

        d_error = (error - self._state.prev_error) / dt
        d_term  = self.Kd * d_error

        delta_t = p_term + i_term + d_term

        # 速率限制
        delta_t = max(-self.max_delta, min(self.max_delta, delta_t))

        target = current_t + delta_t
        target = max(self.t_min, min(self.t_max, target))

        # 更新状态
        self._state.prev_error = error
        self._state.prev_time  = now
        self._state.prev_t     = target

        return target

    def reset(self):
        self._state = AdaptivePolicyState(prev_t=0.0, prev_time=time.time())

    def pid_state(self) -> dict:
        return {
            "integral":   round(self._state.integral, 4),
            "prev_error": round(self._state.prev_error, 4),
            "target_p":   self.target_pressure,
            "Kp_Ki_Kd":   (self.Kp, self.Ki, self.Kd),
        }

    def __repr__(self):
        return (f"<AdaptivePolicy target_p={self.target_pressure} "
                f"Kp={self.Kp} Ki={self.Ki} Kd={self.Kd}>")


# ══════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════
POLICY_REGISTRY = {
    "conservative": ConservativePolicy,
    "aggressive":   AggressivePolicy,
    "adaptive":     AdaptivePolicy,
}

def make_policy(name: str = "adaptive", **kwargs) -> TPolicy:
    """
    策略工厂
    :param name: "conservative" | "aggressive" | "adaptive"
    :param kwargs: 传入对应策略的构造参数
    """
    if name not in POLICY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(POLICY_REGISTRY.keys())}")
    return POLICY_REGISTRY[name](**kwargs)
