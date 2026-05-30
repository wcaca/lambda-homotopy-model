"""
test_suite/test_basic_unit.py
第一部分：基础功能验证
  B01  单λ单元读写 + t插值前向
  B02  串联组端到端
  B03  并联组 sum/concat/mean 融合
  B04  全局拓扑调度（纯λ链路）
  B05  三级t（group/unit 覆盖）
  B06  INT8 单元加载 + 前向
  B07  拓扑静态校验（V01-V06 全部 PASS）
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_suite.utils import TestSuite, fixed_input, assert_shape, assert_finite, assert_changing, MODEL_ROOT

def run() -> bool:
    suite = TestSuite("基础功能验证")
    x = fixed_input(4096)

    # ── B01 单λ单元读写 + t插值 ─────────────────────────────────────
    def b01(r):
        from base_utils import LambdaUnit
        import glob
        bins = glob.glob(os.path.join(MODEL_ROOT, "attn_group", "*.bin"))
        assert bins, "attn_group 无 bin 文件"
        unit = LambdaUnit(bins[0]); unit.load()
        outs = []
        for t in [0.0, 0.5, 1.0]:
            out = unit.forward(x, t)
            assert_shape(out, unit.dim_skeleton, f"t={t}")
            assert_finite(out, f"t={t}")
            outs.append(out)
        assert_changing(outs, "单λ单元 t 插值")
        r.passed(dim_skeleton=unit.dim_skeleton, unit_id=unit.unit_id[:12])
    suite.run_case("B01 单λ单元读写+t插值", b01)

    # ── B02 串联组 ──────────────────────────────────────────────────
    def b02(r):
        from component_manager import ComponentManager
        cm = ComponentManager(os.path.join(MODEL_ROOT, "ffn_group"), lazy_load=False)
        assert cm.group_type == "serial"
        out = cm.forward(x, global_t=0.3)
        assert_finite(out)
        r.passed(units=len(cm.units), out_shape=out.shape[-1])
    suite.run_case("B02 串联组端到端", b02)

    # ── B03 并联组融合 ───────────────────────────────────────────────
    def b03(r):
        from component_manager import ComponentManager
        cm = ComponentManager(os.path.join(MODEL_ROOT, "attn_group"), lazy_load=False)
        assert cm.group_type == "parallel"
        outs = []
        for rule in ["sum", "mean"]:
            cm.merge_rule = rule
            out = cm.forward(x, global_t=0.3)
            assert_finite(out, f"rule={rule}")
            outs.append(float(np.linalg.norm(out)))
        r.passed(units=len(cm.units), sum_norm=round(outs[0],3), mean_norm=round(outs[1],3))
    suite.run_case("B03 并联组融合", b03)

    # ── B04 全局拓扑调度 ─────────────────────────────────────────────
    def b04(r):
        from model_scheduler import ModelScheduler
        s = ModelScheduler(MODEL_ROOT)
        outs = []
        for t in [0.0, 0.5, 1.0]:
            s.set_global_t(t)
            out = s.forward(x.copy())
            assert_finite(out, f"t={t}")
            outs.append(out)
        assert_changing(outs, "全局拓扑")
        r.passed(steps=len(s._topo_steps), out_dim=outs[0].shape[-1])
    suite.run_case("B04 全局拓扑调度", b04)

    # ── B05 三级t覆盖 ────────────────────────────────────────────────
    def b05(r):
        from multi_t_scheduler import MultiTScheduler
        from multi_t import GroupTConfig, UnitTConfig
        s = MultiTScheduler(MODEL_ROOT)
        s.set_global_t(0.5)
        s.set_group_t("attn_group", t=0.8, blend_mode="override")
        s.freeze_unit("norm_group", "norm_001", at=0.0)
        eff_attn = s.engine.effective_t("attn_group", "attn_001")
        eff_norm = s.engine.effective_t("norm_group", "norm_001")
        assert abs(eff_attn - 0.8) < 0.01, f"attn group_t 覆盖失败: {eff_attn}"
        assert abs(eff_norm) < 0.001,       f"norm_001 冻结失败: {eff_norm}"
        out = s.forward(x.copy())
        assert_finite(out)
        r.passed(eff_attn=round(eff_attn,3), eff_norm=round(eff_norm,3))
    suite.run_case("B05 三级t覆盖", b05)

    # ── B06 INT8 单元 ────────────────────────────────────────────────
    def b06(r):
        import tempfile, glob
        from int8_quantizer import INT8Quantizer
        from int8_lambda_unit import INT8LambdaUnit
        attn_dir = os.path.join(MODEL_ROOT, "attn_group")
        with tempfile.TemporaryDirectory() as tmp:
            q = INT8Quantizer()
            results = q.quantize_group(attn_dir, tmp)
            assert results, "量化未输出文件"
            bins = glob.glob(os.path.join(tmp, "*.bin"))
            u8 = INT8LambdaUnit(bins[0]); u8.load()
            assert u8.is_int8,  "版本号未识别为 INT8"
            outs = []
            for t in [0.0, 0.5, 1.0]:
                out = u8.forward(x, t)
                assert_finite(out, f"INT8 t={t}")
                outs.append(out)
            assert_changing(outs, "INT8 单元 t 插值")
            compress = results[0]["size_reduction"]
            err      = results[0]["w0_error"]["rel_error"]
            r.passed(compress=compress, quant_err=round(err,4))
    suite.run_case("B06 INT8单元加载+前向", b06)

    # ── B07 拓扑静态校验 ─────────────────────────────────────────────
    def b07(r):
        from topology_validator import TopologyValidator
        v = TopologyValidator(MODEL_ROOT)
        report = v.validate_static()
        fails  = report.failures
        assert not fails, f"静态校验失败: {[f.message for f in fails]}"
        r.passed(checks=len(report.results), passed=len(report.passed))
    suite.run_case("B07 拓扑静态校验", b07)

    return suite.print_summary()


if __name__ == "__main__":
    run()
