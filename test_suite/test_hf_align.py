"""
test_suite/test_hf_align.py
第二部分：HF封装能力对标（无LLM路线下改为：信号处理有效性验证）
  H01  NativeComponentManager 接口兼容（numpy降级模式）
  H02  三段式拓扑（pre→native→post）数据流通
  H03  t=0 时 native 单元输出稳定（基准态）
  H04  t 连续变化时 native 单元输出平滑（无突变）
  H05  混合链路（λ单元 + native 单元）维度连续
  H06  NativeComponentManager lazy_load 行为验证
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_suite.utils import TestSuite, fixed_input, assert_finite, assert_changing, cosine_sim, MODEL_ROOT

MOCK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_mock_native")
os.makedirs(MOCK_DIR, exist_ok=True)

def run() -> bool:
    suite = TestSuite("封装接口 & 信号处理有效性验证")
    x = fixed_input(4096)

    # ── H01 NativeComponentManager 接口 ─────────────────────────────
    def h01(r):
        from native_wrapper import NativeComponentManager
        cm = NativeComponentManager(MOCK_DIR, group_name="test_native",
                                    input_dim=4096, output_dim=4096)
        out = cm.forward(x, global_t=0.0)
        assert out.shape[-1] == 4096, f"输出维度 {out.shape[-1]} != 4096"
        assert_finite(out, "native forward")
        r.passed(out_dim=out.shape[-1], norm=round(float(np.linalg.norm(out)),3))
    suite.run_case("H01 NativeComponentManager 接口兼容", h01)

    # ── H02 三段式拓扑数据流通 ──────────────────────────────────────
    def h02(r):
        from native_wrapper import NativeComponentManager
        from component_manager import ComponentManager
        pre  = ComponentManager(os.path.join(MODEL_ROOT, "norm_group"),  lazy_load=False)
        nat  = NativeComponentManager(MOCK_DIR, input_dim=4096, output_dim=4096)
        post = ComponentManager(os.path.join(MODEL_ROOT, "route_group"), lazy_load=False)
        h = pre.forward(x, global_t=0.3)
        h = nat.forward(h, global_t=0.3)
        h = post.forward(h, global_t=0.3)
        assert_finite(h, "三段式输出")
        r.passed(final_dim=h.shape[-1], norm=round(float(np.linalg.norm(h)),3))
    suite.run_case("H02 三段式拓扑数据流通", h02)

    # ── H03 t=0 基准态稳定性（多次调用结果一致）────────────────────
    def h03(r):
        from native_wrapper import NativeComponentManager
        cm = NativeComponentManager(MOCK_DIR, input_dim=4096, output_dim=4096)
        outs = [cm.forward(x.copy(), global_t=0.0) for _ in range(3)]
        diffs = [float(np.linalg.norm(outs[i].astype(np.float32) -
                                       outs[0].astype(np.float32)))
                 for i in range(1, 3)]
        assert max(diffs) < 1e-3, f"t=0 重复调用结果不一致: {diffs}"
        r.passed(max_diff=round(max(diffs), 6))
    suite.run_case("H03 t=0 基准态输出稳定", h03)

    # ── H04 t 连续变化输出平滑 ──────────────────────────────────────
    def h04(r):
        from native_wrapper import NativeComponentManager
        cm = NativeComponentManager(MOCK_DIR, input_dim=4096, output_dim=4096)
        t_seq = [i/10 for i in range(11)]
        outs  = [cm.forward(x.copy(), global_t=t) for t in t_seq]
        # 相邻 t 步输出余弦相似度不应骤降
        sims = [cosine_sim(outs[i], outs[i+1]) for i in range(len(outs)-1)]
        min_sim = min(sims)
        assert min_sim > 0.3, f"t 变化时输出出现突变（最小相似度={min_sim:.4f}）"
        # 输出范数应随 t 连续变化（不全相同）
        norms = [float(np.linalg.norm(o)) for o in outs]
        assert max(norms) - min(norms) > 1e-3, "输出范数不随 t 变化"
        r.passed(min_cosine=round(min_sim,3), norm_range=round(max(norms)-min(norms),3))
    suite.run_case("H04 t连续变化输出平滑", h04)

    # ── H05 混合链路维度连续 ────────────────────────────────────────
    def h05(r):
        from native_wrapper import NativeComponentManager
        from component_manager import ComponentManager
        cm_attn = ComponentManager(os.path.join(MODEL_ROOT, "attn_group"), lazy_load=False)
        cm_nat  = NativeComponentManager(MOCK_DIR, input_dim=512, output_dim=512)
        cm_ffn  = ComponentManager(os.path.join(MODEL_ROOT, "ffn_group"),  lazy_load=False)
        for t in [0.0, 0.5, 1.0]:
            h = cm_attn.forward(x, global_t=t)       # 4096 → 512
            h = cm_nat.forward(h,  global_t=t)        # 512 → 512
            h = cm_ffn.forward(h,  global_t=t)        # 512 → 512
            assert_finite(h, f"混合链路 t={t}")
        r.passed(chain="attn→native→ffn", dims="4096→512→512→512")
    suite.run_case("H05 混合链路维度连续", h05)

    # ── H06 lazy_load 行为 ──────────────────────────────────────────
    def h06(r):
        from native_wrapper import NativeComponentManager
        cm = NativeComponentManager(MOCK_DIR, input_dim=4096, output_dim=4096, lazy_load=True)
        assert not cm._unit._loaded, "lazy_load=True 时不应提前加载"
        cm.forward(x, 0.0)   # 触发懒加载
        assert cm._unit._loaded, "首次 forward 后应已加载"
        r.passed(lazy_load_ok=True)
    suite.run_case("H06 lazy_load 行为验证", h06)

    return suite.print_summary()

if __name__ == "__main__":
    run()
