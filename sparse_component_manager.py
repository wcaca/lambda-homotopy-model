"""
sparse_component_manager.py — 支持稀疏路由的组件调度器（增量新增）
继承 ComponentManager，在 parallel 模式下替换为 SparseRouter 执行，
serial 模式行为与原有完全一致。
原有 component_manager.py 零修改。

λ函数单元 + 同伦形变大模型系统 · 稀疏路由组件层
"""

from __future__ import annotations
import numpy as np
from component_manager import ComponentManager
from sparse_router import SparseRouter, DEFAULT_TOP_K, DEFAULT_THRESHOLD


class SparseComponentManager(ComponentManager):
    """
    稀疏路由组件调度器
    - serial 组：行为与父类完全一致（串联无需路由）
    - parallel 组：用 SparseRouter 替换全激活并联，
                   O(N) → O(k)，k = top_k
    """

    def __init__(
        self,
        group_dir:  str,
        lazy_load:  bool  = False,
        top_k:      int   = DEFAULT_TOP_K,
        threshold:  float = DEFAULT_THRESHOLD,
        merge_rule: str   = "weighted_sum",
    ):
        self.top_k      = top_k
        self.threshold  = threshold
        self.merge_rule = merge_rule
        self._router: SparseRouter | None = None

        super().__init__(group_dir, lazy_load=lazy_load)

        # 若并联组且已非惰性加载，立刻建路由器
        if self.group_type == "parallel" and self.units:
            self._init_router()

    # ── 路由器初始化 ────────────────────────────────────────────────────
    def _init_router(self):
        self._router = SparseRouter(
            units      = self.units,
            top_k      = self.top_k,
            threshold  = self.threshold,
            merge_rule = self.merge_rule,
        )
        print(f"  [{self.group_name}] 稀疏路由器就绪: "
              f"top_k={self.top_k}/{len(self.units)}  "
              f"threshold={self.threshold}")

    # ── 覆盖父类 forward ─────────────────────────────────────────────
    def forward(self, x: np.ndarray, global_t: float) -> np.ndarray:
        """
        serial 模式：直接调用父类串联逻辑（零修改）
        parallel 模式：稀疏路由替换全激活并联（O(k) 复杂度）
        """
        self._ensure_loaded()

        if not self.units:
            return x

        current_t = max(0.0, min(1.0, global_t + self.local_t_offset))

        if self.group_type == "serial":
            # 串联：直接复用父类逻辑
            return self._forward_serial(x, current_t)

        elif self.group_type == "parallel":
            # 并联：稀疏路由
            if self._router is None:
                self._init_router()
            return self._router.route_forward(x, current_t)

        else:
            return super().forward(x, global_t)

    # ── 延迟加载后建路由器 ───────────────────────────────────────────
    def _ensure_loaded(self):
        super()._ensure_loaded()
        if self.group_type == "parallel" and self.units and self._router is None:
            self._init_router()

    # ── 路由统计 ─────────────────────────────────────────────────────
    def routing_stats(self) -> dict:
        """返回并联组的稀疏路由统计，串联组返回空字典"""
        if self._router:
            return self._router.stats()
        return {"note": f"{self.group_name} 是串联组，无路由统计"}

    def __repr__(self) -> str:
        base = super().__repr__()
        if self._router:
            s = self._router.stats()
            return f"{base} [sparse top_k={self.top_k} act_rate={s.get('activation_rate', 0):.2%}]"
        return base
