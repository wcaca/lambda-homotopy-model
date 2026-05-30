"""
component_manager.py — 组件调度类
功能：加载组件目录、解析 group.conf、执行串联/并联逻辑
λ函数单元 + 同伦形变大模型系统 · 组件层
"""

import os
import threading
import numpy as np
from base_utils import LambdaUnit, load_ini_config


class ComponentManager:
    """
    组件层调度器
    管理同一功能组内的所有 λ 单元，支持串联(serial)与并联(parallel)两种执行模式。
    """
    def __init__(self, group_dir: str, lazy_load: bool = False):
        self.group_dir    = group_dir
        self.lazy_load    = lazy_load
        self.group_cfg    = None
        self.group_name   = os.path.basename(group_dir)
        self.group_type   = "serial"
        self.merge_rule   = "sum"
        self.local_t_offset = 0.0
        self.unit_names:  list[str]       = []
        self.units:       list[LambdaUnit] = []
        self._load_group()

    # ─── 内部：加载组配置 + 单元 ──────────────────────────────────────
    def _load_group(self):
        cfg_path = os.path.join(self.group_dir, "group.conf")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"group.conf 缺失: {cfg_path}")
        self.group_cfg = load_ini_config(cfg_path)

        self.group_type      = self.group_cfg.get("GroupBase", "GroupType", fallback="serial")
        self.local_t_offset  = float(self.group_cfg.get("GroupBase", "LocalTOffset", fallback="0.0"))
        self.merge_rule      = self.group_cfg.get("MergeConfig", "DefaultMerge", fallback="sum")
        self.group_name      = self.group_cfg.get("GroupBase", "GroupName", fallback=self.group_name)

        raw_list = self.group_cfg.get("GroupBase", "UnitList", fallback="")
        self.unit_names = [n.strip() for n in raw_list.split(",") if n.strip()]

        if not self.lazy_load:
            self._load_all_units()

    def _load_all_units(self):
        self.units = []
        for name in self.unit_names:
            path = os.path.join(self.group_dir, name)
            if not os.path.exists(path):
                print(f"  [警告] 单元文件不存在，已跳过: {path}")
                continue
            unit = LambdaUnit(path)
            unit.load()
            self.units.append(unit)
        print(f"  [{self.group_name}] 已加载 {len(self.units)} 个λ单元 (模式={self.group_type})")

    def _ensure_loaded(self):
        if self.lazy_load and not self.units:
            self._load_all_units()

    # ─── 前向推理 ───────────────────────────────────────────────────────
    def forward(self, x: np.ndarray, global_t: float) -> np.ndarray:
        """
        组件前向执行
        current_t = clamp(global_t + local_t_offset, 0, 1)
        串联：依次经过每个单元
        并联：多线程并行 → 结果融合（sum / concat / mean）
        """
        self._ensure_loaded()

        if not self.units:
            return x  # 空组直接透传

        current_t = max(0.0, min(1.0, global_t + self.local_t_offset))

        if self.group_type == "serial":
            return self._forward_serial(x, current_t)
        elif self.group_type == "parallel":
            return self._forward_parallel(x, current_t)
        else:
            raise ValueError(f"未知 GroupType: {self.group_type}")

    def _forward_serial(self, x: np.ndarray, t: float) -> np.ndarray:
        res = x
        for unit in self.units:
            res = unit.forward(res, t)
        return res

    def _forward_parallel(self, x: np.ndarray, t: float) -> np.ndarray:
        results = [None] * len(self.units)
        lock = threading.Lock()

        def task(idx: int, unit: LambdaUnit):
            out = unit.forward(x, t)
            with lock:
                results[idx] = out

        threads = [threading.Thread(target=task, args=(i, u)) for i, u in enumerate(self.units)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        valid = [r for r in results if r is not None]
        if not valid:
            return x

        if self.merge_rule == "sum":
            # 对齐维度后求和（处理 concat 后维度不一致的情况）
            min_dim = min(r.shape[-1] for r in valid)
            return np.sum([r[..., :min_dim] for r in valid], axis=0)
        elif self.merge_rule == "concat":
            return np.concatenate(valid, axis=-1)
        elif self.merge_rule == "mean":
            min_dim = min(r.shape[-1] for r in valid)
            return np.mean([r[..., :min_dim] for r in valid], axis=0)
        else:
            # weight 加权：等权平均兜底
            min_dim = min(r.shape[-1] for r in valid)
            return np.mean([r[..., :min_dim] for r in valid], axis=0)

    # ─── 实用方法 ────────────────────────────────────────────────────────
    def unit_count(self) -> int:
        return len(self.unit_names)

    def __repr__(self) -> str:
        return (f"<ComponentManager name={self.group_name} type={self.group_type} "
                f"units={self.unit_count()} offset={self.local_t_offset}>")
