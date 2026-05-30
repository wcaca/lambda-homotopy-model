"""
multi_t.py — 多级t参数体系核心（增量新增，原有文件零修改）
======================================================================
三级 t 参数结构：

  Level-0  global_t    ∈ [0,1]  全局形变（原有唯一控制接口，完全保留）
  Level-1  group_t     ∈ [0,1]  组级形变（每个组件组独立压缩率）
  Level-2  unit_t      ∈ [0,1]  单元级形变（每个λ单元独立形变）

最终作用于每个单元的实效 t 值（effective_t）：

  effective_t = clamp( blend(global_t, group_t, unit_t, mode) , 0, 1 )

融合模式 (blend_mode)：
  "override"  : effective_t = unit_t（单元级完全覆盖）
  "additive"  : effective_t = global_t + group_offset + unit_offset
  "multiply"  : effective_t = global_t * group_scale * unit_scale
  "weighted"  : effective_t = w0*global_t + w1*group_t + w2*unit_t（默认）

设计原则：
  - global_t 始终有效，是最粗粒度的统一管控
  - group_t 覆盖同组所有单元，可实现「注意力层更激进压缩」等策略
  - unit_t 精细调控单个λ单元，实现专家单元保留 / 普通单元压缩
  - 任何一级不设置则自动继承上级值（缺省透传）

λ函数单元 + 同伦形变大模型系统 · 多级t参数层
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── 全局常量 ────────────────────────────────────────────────────────────
T_MIN = 0.0
T_MAX = 1.0
_UNSET = float("nan")   # 标记"未设置"，区别于显式 0.0


def _is_set(v: float) -> bool:
    return not (v != v)   # NaN 检测


def clamp(v: float) -> float:
    return max(T_MIN, min(T_MAX, v))


# ══════════════════════════════════════════════════════════════════════
# 单元级 t 描述符
# ══════════════════════════════════════════════════════════════════════
@dataclass
class UnitTConfig:
    """
    单元级 t 配置
    unit_id : 对应 LambdaUnit.unit_id
    t       : 显式 unit_t（NaN = 未设置，继承 group_t）
    scale   : multiply 模式下的缩放因子
    offset  : additive 模式下的偏移量
    frozen  : True = 完全冻结此单元（effective_t 固定为 freeze_at，忽略上级）
    freeze_at: 冻结时的固定 t 值（默认 0.0，保持全量形态）
    """
    unit_id:   str
    t:         float = _UNSET
    scale:     float = 1.0
    offset:    float = 0.0
    frozen:    bool  = False
    freeze_at: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# 组级 t 描述符
# ══════════════════════════════════════════════════════════════════════
@dataclass
class GroupTConfig:
    """
    组级 t 配置
    group_name  : 对应 ComponentManager.group_name
    t           : 显式 group_t（NaN = 继承 global_t）
    scale       : multiply 模式下的缩放因子
    offset      : additive 模式下的偏移量（即原有 LocalTOffset 的扩展）
    blend_mode  : 本组的融合模式（覆盖全局 blend_mode）
    unit_configs: 本组内各单元的精细配置（可选，不填则组内所有单元继承 group_t）
    """
    group_name:   str
    t:            float = _UNSET
    scale:        float = 1.0
    offset:       float = 0.0
    blend_mode:   str   = "weighted"
    unit_configs: dict[str, UnitTConfig] = field(default_factory=dict)

    def add_unit(self, cfg: UnitTConfig):
        self.unit_configs[cfg.unit_id] = cfg

    def get_unit_cfg(self, unit_id: str) -> Optional[UnitTConfig]:
        return self.unit_configs.get(unit_id)


# ══════════════════════════════════════════════════════════════════════
# 多级 t 计算引擎
# ══════════════════════════════════════════════════════════════════════
class MultiTEngine:
    """
    三级 t 参数融合计算引擎
    负责根据 global_t、GroupTConfig、UnitTConfig 计算每个单元的 effective_t
    """
    # 默认融合权重（weighted 模式）
    DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)   # (global, group, unit)

    def __init__(
        self,
        global_t:     float = 0.0,
        blend_mode:   str   = "weighted",
        blend_weights: tuple = None,
    ):
        self.global_t     = clamp(global_t)
        self.blend_mode   = blend_mode
        self.blend_weights = blend_weights or self.DEFAULT_WEIGHTS
        self._group_cfgs: dict[str, GroupTConfig] = {}

        # 统计：记录每次计算的 effective_t 分布
        self._history: list[dict] = []

    # ── 配置注册 ────────────────────────────────────────────────────────
    def set_global_t(self, t: float):
        """设置 Level-0 全局 t（原有唯一接口，完全保留）"""
        self.global_t = clamp(t)

    def set_group(self, cfg: GroupTConfig):
        """注册 Level-1 组级 t 配置"""
        self._group_cfgs[cfg.group_name] = cfg

    def set_unit(self, group_name: str, cfg: UnitTConfig):
        """注册 Level-2 单元级 t 配置"""
        if group_name not in self._group_cfgs:
            self._group_cfgs[group_name] = GroupTConfig(group_name=group_name)
        self._group_cfgs[group_name].add_unit(cfg)

    def remove_group(self, group_name: str):
        self._group_cfgs.pop(group_name, None)

    # ── 核心计算：effective_t ────────────────────────────────────────────
    def effective_t(self, group_name: str, unit_id: str = "") -> float:
        """
        计算指定组/单元的最终实效 t 值
        优先级：unit_frozen > unit_t > group_t > global_t
        """
        g_cfg = self._group_cfgs.get(group_name)
        u_cfg = g_cfg.get_unit_cfg(unit_id) if (g_cfg and unit_id) else None

        # ── 冻结单元：直接返回固定值 ──────────────────────────────────
        if u_cfg and u_cfg.frozen:
            return clamp(u_cfg.freeze_at)

        # ── 确定各级 t 值 ─────────────────────────────────────────────
        g_t = (clamp(g_cfg.t) if (g_cfg and _is_set(g_cfg.t))
               else self.global_t)
        u_t = (clamp(u_cfg.t) if (u_cfg and _is_set(u_cfg.t))
               else g_t)

        # ── 根据融合模式计算 effective_t ─────────────────────────────
        mode = (g_cfg.blend_mode if g_cfg else self.blend_mode)
        eff  = self._blend(self.global_t, g_t, u_t, g_cfg, u_cfg, mode)

        result = clamp(eff)
        self._history.append({
            "group": group_name, "unit": unit_id,
            "global_t": self.global_t, "group_t": g_t, "unit_t": u_t,
            "effective_t": result, "mode": mode,
        })
        return result

    def _blend(
        self,
        gt: float, grp_t: float, ut: float,
        g_cfg: Optional[GroupTConfig],
        u_cfg: Optional[UnitTConfig],
        mode: str,
    ) -> float:
        """融合三级 t 值"""
        if mode == "override":
            # unit_t 完全覆盖（若无 unit_t 则用 group_t）
            return ut

        elif mode == "additive":
            g_offset = g_cfg.offset if g_cfg else 0.0
            u_offset = u_cfg.offset if u_cfg else 0.0
            return gt + g_offset + u_offset

        elif mode == "multiply":
            g_scale = g_cfg.scale if g_cfg else 1.0
            u_scale = u_cfg.scale if u_cfg else 1.0
            return gt * g_scale * u_scale

        elif mode == "weighted":
            w0, w1, w2 = self.blend_weights
            total = w0 + w1 + w2 + 1e-8
            return (w0 * gt + w1 * grp_t + w2 * ut) / total

        else:
            return gt   # 未知模式降级为 global_t

    # ── 批量预览 ────────────────────────────────────────────────────────
    def preview(self, group_unit_pairs: list[tuple[str, str]]) -> list[dict]:
        """
        批量预览 effective_t，不写入 history
        :param group_unit_pairs: [(group_name, unit_id), ...]
        :return: [{"group":..., "unit":..., "effective_t":...}, ...]
        """
        saved = list(self._history)
        results = []
        for g, u in group_unit_pairs:
            eff = self.effective_t(g, u)
            results.append({"group": g, "unit": u, "effective_t": eff})
        self._history = saved   # 回滚 history
        return results

    # ── 统计 ────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        if not self._history:
            return {"calls": 0}
        effs = [r["effective_t"] for r in self._history]
        return {
            "calls":   len(self._history),
            "eff_min": round(min(effs), 4),
            "eff_max": round(max(effs), 4),
            "eff_mean":round(sum(effs) / len(effs), 4),
            "global_t":self.global_t,
            "groups_configured": list(self._group_cfgs.keys()),
        }

    def clear_history(self):
        self._history.clear()

    def __repr__(self) -> str:
        return (f"<MultiTEngine global_t={self.global_t} "
                f"mode={self.blend_mode} groups={list(self._group_cfgs.keys())}>")


# ══════════════════════════════════════════════════════════════════════
# 配置文件解析：从 model_global.conf 读取多级 t 配置
# ══════════════════════════════════════════════════════════════════════
def parse_multi_t_from_conf(cfg) -> MultiTEngine:
    """
    从已加载的 configparser 对象解析 [MultiTConfig] 段，
    构建 MultiTEngine 实例。
    conf 格式示例见 model_global.conf [MultiTConfig] 段。
    """
    global_t   = float(cfg.get("GlobalControl", "GlobalT", fallback="0.0"))
    blend_mode = cfg.get("MultiTConfig", "BlendMode", fallback="weighted") \
        if cfg.has_section("MultiTConfig") else "weighted"

    engine = MultiTEngine(global_t=global_t, blend_mode=blend_mode)

    if not cfg.has_section("MultiTConfig"):
        return engine

    # 解析组级配置：GroupT_<组名>=<t值>[,scale=x][,offset=x][,mode=x]
    for key, val in cfg.items("MultiTConfig"):
        if key.lower().startswith("groupt_"):
            group_name = key[len("groupt_"):]
            parts = [p.strip() for p in val.split(",")]
            t_val = float(parts[0]) if parts else _UNSET
            g_cfg = GroupTConfig(group_name=group_name, t=t_val)
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip().lower()
                    if k == "scale":   g_cfg.scale      = float(v)
                    if k == "offset":  g_cfg.offset     = float(v)
                    if k == "mode":    g_cfg.blend_mode = v.strip()
            engine.set_group(g_cfg)

        # 解析单元级配置：UnitT_<组名>_<单元ID>=<t值>[,frozen][,freeze_at=x]
        elif key.lower().startswith("unitt_"):
            rest  = key[len("unitt_"):]
            # 格式：组名__单元ID（双下划线分隔，避免与单元ID中的单下划线冲突）
            if "__" in rest:
                group_name, unit_id = rest.split("__", 1)
            else:
                continue
            parts = [p.strip() for p in val.split(",")]
            t_val = float(parts[0]) if parts else _UNSET
            u_cfg = UnitTConfig(unit_id=unit_id, t=t_val)
            for p in parts[1:]:
                p_low = p.lower()
                if p_low == "frozen":
                    u_cfg.frozen = True
                elif p_low.startswith("freeze_at="):
                    u_cfg.freeze_at = float(p_low.split("=")[1])
                elif p_low.startswith("scale="):
                    u_cfg.scale = float(p_low.split("=")[1])
                elif p_low.startswith("offset="):
                    u_cfg.offset = float(p_low.split("=")[1])
            engine.set_unit(group_name, u_cfg)

    return engine
