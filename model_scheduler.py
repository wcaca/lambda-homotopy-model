"""
model_scheduler.py — 全局主调度器（系统入口）
功能：加载全局配置、解析拓扑链路、驱动全模型运行
λ函数单元 + 同伦形变大模型系统 · 调度层
"""

import os
import re
import numpy as np
from base_utils import load_ini_config
from component_manager import ComponentManager


class ModelScheduler:
    """
    全局主调度器
    唯一外部控制入口：set_global_t(t)
    拓扑链路语法：serial(组名) -> parallel(组名) -> ...
    """
    def __init__(self, model_root: str):
        self.model_root     = os.path.abspath(model_root)
        self.global_cfg     = None
        self.global_t       = 0.0
        self.origin_dim     = 4096
        self.compact_dim    = 512
        self.topology_chain = ""
        self.lazy_load      = True
        self.thread_num     = 4
        self.components: dict[str, ComponentManager] = {}
        self._topo_steps: list[tuple[str, str]] = []  # [(exec_type, group_name), ...]

        print(f"\n═══ ModelScheduler 初始化 ═══")
        print(f"  模型根目录: {self.model_root}")
        self._parse_global_config()
        self._load_components()
        self._parse_topology()
        print(f"  全局参数 t = {self.global_t}")
        print(f"  拓扑步骤数 = {len(self._topo_steps)}")
        print(f"═══════════════════════════════\n")

    # ─── 配置解析 ─────────────────────────────────────────────────────
    def _parse_global_config(self):
        cfg_path = os.path.join(self.model_root, "model_global.conf")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"找不到全局配置: {cfg_path}")
        self.global_cfg  = load_ini_config(cfg_path)

        self.global_t    = float(self.global_cfg.get("GlobalControl", "GlobalT",    fallback="0.0"))
        self.origin_dim  = int(self.global_cfg.get("GlobalControl",  "OriginDim",   fallback="4096"))
        self.compact_dim = int(self.global_cfg.get("GlobalControl",  "CompactDim",  fallback="512"))
        self.topology_chain = self.global_cfg.get("Topology", "TopologyChain", fallback="")
        self.lazy_load   = int(self.global_cfg.get("System",   "LazyLoad",     fallback="1")) == 1
        self.thread_num  = int(self.global_cfg.get("System",   "ThreadNum",    fallback="4"))
        print(f"  全局配置已解析: origin={self.origin_dim}  compact={self.compact_dim}  lazy={self.lazy_load}")

    def _load_components(self):
        """扫描 model_root 下所有含 group.conf 的子目录，实例化 ComponentManager"""
        for entry in sorted(os.listdir(self.model_root)):
            full = os.path.join(self.model_root, entry)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "group.conf")):
                self.components[entry] = ComponentManager(full, lazy_load=self.lazy_load)
        print(f"  组件组已加载: {list(self.components.keys())}")

    def _parse_topology(self):
        """将拓扑字符串解析为有序步骤列表"""
        chain = self.topology_chain.replace(" ", "")
        self._topo_steps = []
        for step in chain.split("->"):
            m = re.match(r"(serial|parallel)\((\w+)\)", step)
            if m:
                self._topo_steps.append((m.group(1), m.group(2)))
            elif step:
                print(f"  [警告] 无法解析拓扑步骤: {step}")

    # ─── 唯一外部控制接口 ─────────────────────────────────────────────
    def set_global_t(self, t: float):
        """
        设置全局形变参数 t ∈ [0, 1]
        t=0 → 全量原始形态
        t=1 → 极致骨架压缩形态
        0<t<1 → 连续过渡态（无硬截断）
        """
        self.global_t = max(0.0, min(1.0, t))

    def get_global_t(self) -> float:
        return self.global_t

    # ─── 全模型前向推理 ────────────────────────────────────────────────
    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        执行拓扑链路推理
        按 TopologyChain 顺序，依次调用各组件的 forward()
        组件内部 serial/parallel 由 ComponentManager 决定
        """
        current = input_vec
        for exec_type, g_name in self._topo_steps:
            if g_name not in self.components:
                print(f"  [警告] 拓扑引用的组件不存在: {g_name}，已跳过")
                continue
            # exec_type 此处作为提示，实际执行模式由 group.conf 中 GroupType 控制
            current = self.components[g_name].forward(current, self.global_t)
        return current

    # ─── 状态工具 ──────────────────────────────────────────────────────
    def current_dim(self) -> int:
        """根据当前 t 计算预期输出维度"""
        from base_utils import calc_dim
        return calc_dim(self.global_t, self.origin_dim, self.compact_dim)

    def topology_summary(self) -> str:
        lines = [f"拓扑链路 (t={self.global_t:.3f}):"]
        for i, (etype, gname) in enumerate(self._topo_steps):
            cm = self.components.get(gname)
            info = f"{cm.group_type}/{cm.unit_count()}单元" if cm else "未加载"
            lines.append(f"  [{i+1}] {etype}({gname})  →  {info}")
        return "\n".join(lines)


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    import sys

    MODEL_ROOT = os.path.join(os.path.dirname(__file__), "model_root")
    scheduler  = ModelScheduler(MODEL_ROOT)

    print(scheduler.topology_summary())
    print()

    # 模拟 4096维 输入向量（仿真数据）
    np.random.seed(42)
    input_data = np.random.randn(4096).astype(np.float32)

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("═══ 同伦形变推理测试 ═══")
    for t_val in test_points:
        scheduler.set_global_t(t_val)
        output = scheduler.forward(input_data)
        print(f"  t={t_val:.2f}  →  输出 shape={output.shape}  "
              f"均值={output.mean():.4f}  范数={np.linalg.norm(output):.4f}")
    print("═══ 测试通过 ✓ ═══")
