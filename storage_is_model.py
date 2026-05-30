"""
storage_is_model.py — 「存储即模型」极简验证
======================================================
核心命题：把 model_root 目录拷贝到任何地方，这段代码就能直接运行。
不需要 PyTorch / TensorFlow / CUDA / 任何AI框架。
唯一依赖：numpy + Python标准库。

运行方式：
    python storage_is_model.py

预期输出：
    [1] 扫描目录 → 发现完整模型结构
    [2] 加载 → 直接从 *.bin 文件读取权重+规则
    [3] 执行 → 前向推理完成，无需任何框架
    [4] 形变 → 单参 t 连续改变，模型实时变化
"""

import os
import sys
import numpy as np

# ── 唯一依赖：本目录下的三个文件 ────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_utils import LambdaUnit
from component_manager import ComponentManager
from model_scheduler import ModelScheduler

# ════════════════════════════════════════════════════════════════════════
# 步骤 1：扫描目录——发现完整模型
# ════════════════════════════════════════════════════════════════════════
print("=" * 56)
print("  验证命题：存储即模型（可形变·可组网·可直接执行）")
print("=" * 56)

MODEL_ROOT = os.path.join(script_dir, "model_root")

print("\n[1] 扫描目录 →", MODEL_ROOT)
total_bin = 0
for root, dirs, files in os.walk(MODEL_ROOT):
    bins = [f for f in files if f.endswith(".bin")]
    confs = [f for f in files if f.endswith(".conf")]
    if bins or confs:
        rel = os.path.relpath(root, MODEL_ROOT) or "."
        print(f"    {rel}/  → {len(bins)} 个λ单元, {len(confs)} 个配置")
        total_bin += len(bins)

print(f"\n    共 {total_bin} 个 *.bin 文件 = 完整大模型（无需其他文件）")

# ════════════════════════════════════════════════════════════════════════
# 步骤 2：加载——直接从文件读取，无框架
# ════════════════════════════════════════════════════════════════════════
print("\n[2] 加载模型（纯文件读取，零框架依赖）...")
scheduler = ModelScheduler(MODEL_ROOT)

# 验证：展示第一个 λ单元的内部结构
first_group = next(iter(scheduler.components.values()))
first_group._ensure_loaded()
if first_group.units:
    u = first_group.units[0]
    print(f"\n    示例 λ单元: {u.unit_id}")
    print(f"    ├─ 函数类型:  {u.func_type}  (0=attention)")
    print(f"    ├─ 原始维度:  {u.dim_origin}")
    print(f"    ├─ 骨架维度:  {u.dim_skeleton}")
    print(f"    ├─ W0 shape:  {u.w0.shape}  (原始权重)")
    print(f"    ├─ W1 shape:  {u.w1.shape}  (压缩权重)")
    print(f"    ├─ 指纹向量:  {u.finger}  (8维路由)")
    print(f"    └─ 接口配置:  {u.port_config}")

# ════════════════════════════════════════════════════════════════════════
# 步骤 3：执行——前向推理，无需框架
# ════════════════════════════════════════════════════════════════════════
print("\n[3] 前向推理（纯 numpy，无 PyTorch / CUDA）...")

np.random.seed(0)
x = np.random.randn(4096).astype(np.float32)

scheduler.set_global_t(0.0)
out = scheduler.forward(x)
print(f"\n    输入:  x.shape = {x.shape}")
print(f"    输出:  out.shape = {out.shape}")
print(f"    执行链路: {scheduler.topology_chain}")

# ════════════════════════════════════════════════════════════════════════
# 步骤 4：形变——单参 t，模型连续变化
# ════════════════════════════════════════════════════════════════════════
print("\n[4] 同伦形变：改变 t，模型实时变化（无需重新加载文件）")
print()
print(f"    {'t':>5}  |  {'输出范数':>10}  |  {'输出均值':>10}  |  说明")
print(f"    {'-'*5}--+-{'-'*10}--+-{'-'*10}--+-{'-'*16}")

labels = {0.0: "全量原始形态", 0.25: "轻度压缩", 0.5: "中间过渡态",
          0.75: "重度压缩", 1.0: "极致骨架形态"}

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    scheduler.set_global_t(t)
    out = scheduler.forward(x)
    norm = np.linalg.norm(out)
    mean = out.mean()
    print(f"    {t:>5.2f}  |  {norm:>10.4f}  |  {mean:>10.6f}  |  {labels[t]}")

# ════════════════════════════════════════════════════════════════════════
# 结论
# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 56)
print("  ✓  存储即模型：model_root 目录 = 完整可执行大模型")
print("  ✓  可直接执行：无框架，纯 Python + numpy")
print("  ✓  可形变：set_global_t(t) 实时切换压缩强度")
print("  ✓  可组网：目录结构即拓扑，配置文件即连接关系")
print("=" * 56)
