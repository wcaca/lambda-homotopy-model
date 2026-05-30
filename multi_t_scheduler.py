"""
multi_t_scheduler.py — 三级t参数全局调度器（增量新增，原有文件零修改）
继承 ModelScheduler，在每次 forward 时按组/单元查询 effective_t，
替换原有单一 global_t 传入各组件，原有串并联/惰性加载逻辑完全保留。

λ函数单元 + 同伦形变大模型系统 · 多级t调度层

用法：
    python multi_t_scheduler.py
"""

from __future__ import annotations
import os, sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from model_scheduler import ModelScheduler
from component_manager import ComponentManager
from multi_t import (
    MultiTEngine, GroupTConfig, UnitTConfig,
    parse_multi_t_from_conf, _UNSET, _is_set,
)


class MultiTScheduler(ModelScheduler):
    """
    三级 t 参数调度器
    ─────────────────────────────────────────────────────────────────
    新增接口：
      set_global_t(t)                      ← 原有接口，完全保留
      set_group_t(group, t, **kwargs)       ← 设置 Level-1 组级 t
      set_unit_t(group, unit, t, **kwargs)  ← 设置 Level-2 单元级 t
      freeze_unit(group, unit, at=0.0)      ← 冻结单个λ单元
      set_blend_mode(mode)                  ← 切换融合模式
      t_snapshot()                          ← 打印当前三级t全貌
    ─────────────────────────────────────────────────────────────────
    forward() 内部改动：
      对每个组调用 engine.effective_t(group_name, unit_id)，
      将 effective_t 透传给 unit.forward()，而非统一用 global_t。
    """

    def __init__(self, model_root: str):
        super().__init__(model_root)
        # 构建多级t引擎（从 conf 读取初始配置）
        self.engine = parse_multi_t_from_conf(self.global_cfg)
        print(f"  [MultiT] 三级t引擎就绪: {self.engine}")

    # ── Level-0：全局 t（原有接口，完全保留）───────────────────────────
    def set_global_t(self, t: float):
        super().set_global_t(t)
        self.engine.set_global_t(t)

    # ── Level-1：组级 t ────────────────────────────────────────────────
    def set_group_t(
        self,
        group_name: str,
        t:          float,
        scale:      float = 1.0,
        offset:     float = 0.0,
        blend_mode: str   = "weighted",
    ):
        """
        为指定组设置独立 t 值
        示例：让 attn_group 比全局更激进压缩
            scheduler.set_group_t("attn_group", t=0.8)
        """
        cfg = self.engine._group_cfgs.get(group_name)
        if cfg is None:
            cfg = GroupTConfig(group_name=group_name)
        cfg.t          = t
        cfg.scale      = scale
        cfg.offset     = offset
        cfg.blend_mode = blend_mode
        self.engine.set_group(cfg)

    def clear_group_t(self, group_name: str):
        """移除组级 t 配置，回退到继承 global_t"""
        self.engine.remove_group(group_name)

    # ── Level-2：单元级 t ──────────────────────────────────────────────
    def set_unit_t(
        self,
        group_name: str,
        unit_id:    str,
        t:          float,
        scale:      float = 1.0,
        offset:     float = 0.0,
    ):
        """
        为指定λ单元设置独立 t 值
        示例：将某个单元保持全量形态（t=0），不随全局压缩
            scheduler.set_unit_t("attn_group", "attn_001", t=0.0)
        """
        self.engine.set_unit(
            group_name,
            UnitTConfig(unit_id=unit_id, t=t, scale=scale, offset=offset),
        )

    def freeze_unit(self, group_name: str, unit_id: str, at: float = 0.0):
        """
        冻结λ单元：无论 global_t 如何变化，此单元始终使用 at 值
        用途：保留"专家单元"始终全量运行，其他单元随 t 压缩
        """
        self.engine.set_unit(
            group_name,
            UnitTConfig(unit_id=unit_id, frozen=True, freeze_at=at),
        )
        print(f"  [MultiT] 已冻结 {group_name}/{unit_id} → t_fixed={at}")

    def unfreeze_unit(self, group_name: str, unit_id: str):
        """解冻单元，恢复正常三级 t 计算"""
        cfg = self.engine._group_cfgs.get(group_name)
        if cfg and unit_id in cfg.unit_configs:
            cfg.unit_configs[unit_id].frozen = False
            print(f"  [MultiT] 已解冻 {group_name}/{unit_id}")

    # ── 融合模式切换 ────────────────────────────────────────────────────
    def set_blend_mode(self, mode: str):
        """
        切换全局融合模式
        mode: "weighted" | "additive" | "multiply" | "override"
        """
        self.engine.blend_mode = mode
        print(f"  [MultiT] 融合模式切换为: {mode}")

    # ── 三级t快照（可视化当前状态）─────────────────────────────────────
    def t_snapshot(self):
        """打印当前所有组/单元的 effective_t 全貌"""
        print(f"\n{'─'*60}")
        print(f"  三级t快照  global_t={self.global_t:.3f}  "
              f"mode={self.engine.blend_mode}")
        print(f"{'─'*60}")

        for gname, cm in self.components.items():
            cm._ensure_loaded()
            g_cfg = self.engine._group_cfgs.get(gname)
            g_t_str = f"{g_cfg.t:.3f}" if (g_cfg and g_cfg.t == g_cfg.t) else "inherit"
            print(f"  [{gname}]  group_t={g_t_str}")

            for unit in cm.units:
                eff = self.engine.effective_t(gname, unit.unit_id)
                u_cfg = g_cfg.get_unit_cfg(unit.unit_id) if g_cfg else None
                frozen_tag = " [FROZEN]" if (u_cfg and u_cfg.frozen) else ""
                u_t_str    = f"{u_cfg.t:.3f}" if (u_cfg and _is_set(u_cfg.t) and not u_cfg.frozen) else "inherit"
                print(f"    ├─ {unit.unit_id:16s} "
                      f"unit_t={u_t_str:8s} → eff={eff:.4f}{frozen_tag}")
        self.engine.clear_history()
        print(f"{'─'*60}\n")

    # ── 覆盖核心 forward：将 effective_t 注入各单元 ─────────────────────
    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        三级 t 前向推理
        与父类 forward 逻辑相同，区别：
          每个单元使用 engine.effective_t(group, unit_id) 而非统一 global_t
        """
        import re
        chain   = self.topology_chain.replace(" ", "")
        steps   = chain.split("->")
        current = input_vec

        for step in steps:
            m = re.match(r"(serial|parallel)\((\w+)\)", step)
            if not m:
                continue
            g_name = m.group(2)
            if g_name not in self.components:
                continue
            cm = self.components[g_name]
            current = self._forward_component(cm, g_name, current)

        return current

    def _forward_component(
        self, cm: ComponentManager, g_name: str, x: np.ndarray
    ) -> np.ndarray:
        """
        对单个组件执行前向，为组内每个单元注入 effective_t
        保持串联/并联逻辑不变，仅替换 t 值来源
        """
        import threading
        cm._ensure_loaded()
        if not cm.units:
            return x

        current_t = self.engine.effective_t(g_name)  # 组级 effective_t（无unit_id）

        if cm.group_type == "serial":
            res = x
            for unit in cm.units:
                eff_t = self.engine.effective_t(g_name, unit.unit_id)
                res   = unit.forward(res, eff_t)
            return res

        elif cm.group_type == "parallel":
            results = [None] * len(cm.units)
            lock    = threading.Lock()

            def task(idx, unit):
                eff_t = self.engine.effective_t(g_name, unit.unit_id)
                out   = unit.forward(x, eff_t)
                with lock:
                    results[idx] = out

            threads = [threading.Thread(target=task, args=(i, u))
                       for i, u in enumerate(cm.units)]
            for th in threads: th.start()
            for th in threads: th.join()

            valid = [r for r in results if r is not None]
            if not valid:
                return x
            min_dim = min(r.shape[-1] for r in valid)
            if cm.merge_rule == "sum":
                return np.sum([r[..., :min_dim] for r in valid], axis=0)
            elif cm.merge_rule == "concat":
                return np.concatenate(valid, axis=-1)
            else:
                return np.mean([r[..., :min_dim] for r in valid], axis=0)
        else:
            return x


# ════════════════════════════════════════════════════════════════════════
# 主程序入口（完整验证留给 verify_multi_t.py）
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MODEL_ROOT = os.path.join(script_dir, "model_root")
    s = MultiTScheduler(MODEL_ROOT)

    np.random.seed(0)
    x = np.random.randn(4096).astype(np.float32)

    print("\n── 快速冒烟测试 ──")
    s.set_global_t(0.5)
    s.set_group_t("attn_group", t=0.8)     # 注意力层更激进压缩
    s.set_unit_t("ffn_group", "ffn_001", t=0.1)  # ffn_001 保守压缩
    s.freeze_unit("norm_group", "norm_001", at=0.0)  # norm_001 完全冻结

    s.t_snapshot()

    out = s.forward(x)
    print(f"输出 shape={out.shape}  范数={np.linalg.norm(out):.4f}")
    print("── 冒烟测试通过 ✓ ──")
