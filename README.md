# λ函数单元 + 同伦形变大模型系统

> 纯逻辑 · 无AI框架依赖 · 单参 `t` 全局管控 · Python 3.8+ 通用环境

## 核心思想

| 概念 | 说明 |
|------|------|
| **λ演算单参化** | 一切计算逻辑封装为单参λ函数，外部仅暴露全局参数 `t ∈ [0,1]` |
| **同伦连续形变** | `t=0` 全量原始模型 → `t=1` 极致压缩骨架，中间态无硬截断 |
| **模块化存储** | 每个λ单元独立 `*.bin` 文件，可单独加载/替换 |
| **串并联组网** | `serial` 串联 + `parallel` 并联，支持任意混合拓扑 |

## 形变公式

```
维度形变：d(t) = d_origin·(1-t) + d_compact·t
权重形变：W(t) = W_0·(1-t) + W_1·t
偏置形变：b(t) = b_0·(1-t) + b_1·t
```

## 目录结构

```
lambda_homotopy/
├─ base_utils.py           # 基础工具：二进制读写、形变计算
├─ component_manager.py    # 组件层：串联/并联调度
├─ model_scheduler.py      # 调度层：全局总控入口
├─ generate_demo_unit.py   # 仿真单元生成工具
├─ requirements.txt
└─ model_root/
   ├─ model_global.conf    # 全局总控配置（唯一控制入口）
   ├─ attn_group/          # 注意力组（并联）
   │  ├─ group.conf
   │  └─ lambda_attn_*.bin
   ├─ ffn_group/           # 前馈网络组（串联）
   ├─ norm_group/          # 归一化组（串联）
   └─ route_group/         # 路由组（串联）
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成仿真单元（零GPU门槛）
```bash
python generate_demo_unit.py
```

### 3. 运行全链路验证
```bash
python model_scheduler.py
```

预期输出：
```
t=0.00  →  输出 shape=(512,)  ...
t=0.50  →  输出 shape=(512,)  ...
t=1.00  →  输出 shape=(512,)  ...
═══ 测试通过 ✓ ═══
```

### 4. 动态控制形变参数
```python
from model_scheduler import ModelScheduler

scheduler = ModelScheduler("./model_root")

# 动态切换压缩强度（唯一外部接口）
scheduler.set_global_t(0.0)   # 全量形态
output = scheduler.forward(input_vec)

scheduler.set_global_t(0.5)   # 中间过渡态
output = scheduler.forward(input_vec)

scheduler.set_global_t(1.0)   # 骨架压缩态
output = scheduler.forward(input_vec)
```

## λ单元 .bin 文件格式

| 区块 | 大小 | 说明 |
|------|------|------|
| 头部元数据 | 64 Byte（定长） | UnitID、FuncType、维度信息、版本 |
| 控制规则区 | 变长文本 | 形变公式、启停标记，`\n` 结尾 |
| 骨架权重区 | FP16 二进制 | W0 + W1 + Bias，行优先存储 |
| 指纹区 | 16 Byte（定长） | 8维 FP16 语义路由向量 |
| 接口配置区 | 变长 JSON | InPort、OutPort、MergeRule，`\0` 结尾 |

## 函数类型枚举

| 值 | 类型 |
|----|------|
| 0 | 注意力单元 (attention) |
| 1 | 前馈网络 (ffn) |
| 2 | 归一化 (normalization) |
| 3 | 投影 (projection) |
| 4 | 路由 (route) |

## 拓扑语法

```ini
# model_global.conf → [Topology]
TopologyChain=serial(norm_group) -> parallel(attn_group) -> serial(ffn_group) -> serial(route_group)
```

- `serial(组名)` — 组内单元依次串联执行
- `parallel(组名)` — 组内单元多线程并联，结果融合（sum/concat/mean）
- `->` 分隔执行步骤

## 扩展方向

- **8维指纹路由增强**：根据指纹稀疏激活并联分支
- **多级 t 参数**：全局 t + 分层子 t
- **拓扑校验**：向量相似度检测同伦等价性
- **跨设备组网**：多节点分布式串并联
- **INT8量化**：进一步压缩存储体积

## 依赖

```
numpy >= 1.21
Python 内置：struct, os, json, configparser, threading
无 PyTorch / TensorFlow / CUDA 依赖
```
