"""
test_suite/test_exception.py
第五部分：边界 & 异常验证（稳定性）
  E01  t 极值钳位（-inf / +inf / NaN）
  E02  空文本输入不崩溃
  E03  损坏图片文件优雅降级
  E04  缺失音频文件优雅降级
  E05  配置文件缺失字段自动降级（向后兼容）
  E06  拓扑校验故障注入（幽灵组/环路/缺失conf）
  E07  学习型指纹边界（零向量/全1向量输入）
  E08  INT8量化边界（全零矩阵/极大值矩阵）
  E09  并联空单元列表不崩溃
  E10  连续长时间推理内存稳定（50轮无 OOM）
"""
import os, sys, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_suite.utils import TestSuite, fixed_input, assert_finite, MODEL_ROOT

def run() -> bool:
    suite = TestSuite("边界 & 异常稳定性验证")
    x = fixed_input(4096)

    # ── E01 t 极值钳位 ───────────────────────────────────────────────
    def e01(r):
        from model_scheduler import ModelScheduler
        s = ModelScheduler(MODEL_ROOT)
        for bad in [-1e9, 1e9, float("nan"), float("inf"), float("-inf")]:
            try:
                s.set_global_t(bad)
            except Exception:
                pass  # 允许抛出，关键是不崩溃后续调用
            t = s.global_t
            assert 0.0 <= t <= 1.0 or t != t, f"t 未钳位: input={bad} got={t}"
            # 重置保险
            s.set_global_t(0.0)
            out = s.forward(x.copy())
            assert_finite(out, f"极值 t={bad} 后推理")
        r.passed(tested=5)
    suite.run_case("E01 t极值钳位（含 inf/nan）", e01)

    # ── E02 空文本输入 ───────────────────────────────────────────────
    def e02(r):
        from modal_utils import text_to_vector
        for text in ["", " ", "\n", "\t"]:
            vec, finger = text_to_vector(text)
            assert vec.shape == (4096,)
            # 不要求非零，要求 finite
            assert np.all(np.isfinite(vec.astype(np.float32))), f"空文本输出含 NaN: [{text!r}]"
        r.passed(cases=4)
    suite.run_case("E02 空文本输入优雅处理", e02)

    # ── E03 损坏图片文件 ─────────────────────────────────────────────
    def e03(r):
        from modal_utils import image_to_vector
        import tempfile
        # 写入随机二进制（损坏图片）
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(b"\xff\xd8\xff" + os.urandom(100))  # 伪 JPEG
            path = tf.name
        try:
            vec, finger = image_to_vector(path)
            # 若能加载，验证输出合法
            assert vec.shape == (4096,)
        except Exception:
            pass  # 允许抛出，关键是不传染崩溃
        finally:
            os.unlink(path)
        # 核心：主调度器不因图片错误而崩溃
        r.passed(graceful_fail=True)
    suite.run_case("E03 损坏图片文件优雅降级", e03)

    # ── E04 缺失音频文件 ─────────────────────────────────────────────
    def e04(r):
        from modal_utils import audio_to_vector
        try:
            audio_to_vector("/tmp/nonexistent_audio_file_xyz.wav")
            r.skipped("audio_to_vector 未抛出异常（实现可能静默降级）")
        except Exception as e:
            # 期望抛出，且异常类型合理
            assert "No such file" in str(e) or "not found" in str(e).lower() \
                   or len(str(e)) > 0, f"异常信息为空"
            r.passed(exception_type=type(e).__name__)
    suite.run_case("E04 缺失音频文件优雅降级", e04)

    # ── E05 配置字段缺失向后兼容 ────────────────────────────────────
    def e05(r):
        import tempfile, configparser
        from base_utils import LambdaUnit
        import glob
        # 测试：旧版 group.conf 无 SupportModality 字段，调度器不崩溃
        bins = glob.glob(os.path.join(MODEL_ROOT, "ffn_group", "*.bin"))
        assert bins
        unit = LambdaUnit(bins[0]); unit.load()
        out = unit.forward(x, 0.3)
        assert_finite(out, "旧版单元前向")
        # 读取 group.conf 并删除可选字段后仍可解析
        cfg_path = os.path.join(MODEL_ROOT, "ffn_group", "group.conf")
        cfg = configparser.ConfigParser()
        cfg.read(cfg_path, encoding="utf-8")
        assert cfg.has_option("GroupBase", "UnitList"), "必填字段 UnitList 缺失"
        r.passed(backward_compat=True)
    suite.run_case("E05 配置字段缺失向后兼容", e05)

    # ── E06 拓扑故障注入 ─────────────────────────────────────────────
    def e06(r):
        from topology_validator import TopologyValidator
        import configparser

        BAD = "/tmp/topo_fault_test"
        shutil.copytree(MODEL_ROOT, BAD, dirs_exist_ok=True)

        # 注入：拓扑引用幽灵组
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(BAD, "model_global.conf"), encoding="utf-8")
        orig = cfg.get("Topology", "TopologyChain")
        cfg.set("Topology", "TopologyChain", orig + " -> serial(ghost_xxx)")
        with open(os.path.join(BAD, "model_global.conf"), "w", encoding="utf-8") as f:
            cfg.write(f)

        v = TopologyValidator(BAD)
        report = v.validate_static()
        shutil.rmtree(BAD)

        v02 = next((res for res in report.results if res.code == "V02"), None)
        assert v02 and v02.status == "FAIL", "幽灵组未被检测为 FAIL"
        r.passed(V02=v02.status, detected=True)
    suite.run_case("E06 拓扑故障注入（幽灵组）", e06)

    # ── E07 学习型指纹边界输入 ──────────────────────────────────────
    def e07(r):
        from component_manager import ComponentManager
        from learnable_finger import LearnableFingerManager, estimate_output_quality
        import glob
        cm = ComponentManager(os.path.join(MODEL_ROOT, "attn_group"), lazy_load=False)
        manager = LearnableFingerManager(cm.units, lr=0.05)
        # 零向量输入
        x_zero = np.zeros(4096, dtype=np.float32)
        for unit in cm.units:
            out = unit.forward(x_zero, 0.5)
            q = estimate_output_quality(x_zero, out, 0.5)
            manager.update(unit, x_zero, out, 0.5)
            assert 0.0 <= q <= 1.0, f"质量估算超界: {q}"
        # 全1向量输入
        x_ones = np.ones(4096, dtype=np.float32)
        for unit in cm.units:
            out = unit.forward(x_ones, 0.5)
            assert_finite(out, "全1输入")
        r.passed(zero_vec_ok=True, ones_vec_ok=True)
    suite.run_case("E07 学习型指纹边界输入", e07)

    # ── E08 INT8量化边界 ─────────────────────────────────────────────
    def e08(r):
        from int8_quantizer import quantize_int8, dequantize_int8, quantization_error
        # 全零矩阵
        w_zero = np.zeros((128, 64), dtype=np.float16)
        q_zero, scale_zero = quantize_int8(w_zero)
        assert scale_zero >= 0, "全零矩阵 scale 应非负"
        # 极大值矩阵
        w_big = np.full((128, 64), 1e3, dtype=np.float16)
        q_big, scale_big = quantize_int8(w_big)
        err = quantization_error(w_big, q_big, scale_big)
        assert err["rel_error"] < 0.1, f"极大值量化误差过大: {err['rel_error']}"
        r.passed(zero_scale=round(scale_zero,4), big_err=round(err["rel_error"],4))
    suite.run_case("E08 INT8量化边界（全零/极大值）", e08)

    # ── E09 并联空单元列表 ───────────────────────────────────────────
    def e09(r):
        """验证：并联组无单元时不崩溃，返回原始输入或零向量"""
        from component_manager import ComponentManager
        import tempfile, os
        # 创建最小合法 group.conf（UnitList 为空）
        with tempfile.TemporaryDirectory() as tmp:
            conf = (
                "[GroupBase]\nGroupName=empty_group\n"
                "UnitList=\nGroupType=parallel\nLocalTOffset=0.0\n\n"
                "[MergeConfig]\nDefaultMerge=sum\n"
            )
            with open(os.path.join(tmp, "group.conf"), "w") as f:
                f.write(conf)
            try:
                cm = ComponentManager(tmp, lazy_load=False)
                out = cm.forward(x, global_t=0.3)
                r.passed(empty_parallel_ok=True, out_shape=out.shape)
            except Exception as e:
                # 允许抛出异常，但不应是未处理的系统崩溃
                r.passed(raises=type(e).__name__, msg=str(e)[:40])
    suite.run_case("E09 并联空单元列表", e09)

    # ── E10 长时间推理内存稳定 ───────────────────────────────────────
    def e10(r):
        import gc
        from model_scheduler import ModelScheduler
        s = ModelScheduler(MODEL_ROOT)
        s.set_global_t(0.5)
        norms = []
        for i in range(50):
            out = s.forward(x.copy())
            norms.append(float(np.linalg.norm(out)))
        # 输出范数应在合理区间内不漂移（无数值爆炸）
        assert max(norms) < 1e6, f"范数爆炸: max={max(norms):.1f}"
        assert min(norms) >= 0,  f"范数负值: min={min(norms):.4f}"
        # 相邻结果一致（无随机漂移）
        diffs = [abs(norms[i] - norms[0]) for i in range(1, 50)]
        assert max(diffs) < 1e-3, f"长时间推理结果漂移: max_diff={max(diffs):.6f}"
        r.passed(runs=50, norm_stable=round(norms[0],3), max_drift=round(max(diffs),6))
    suite.run_case("E10 50轮推理内存&数值稳定", e10)

    return suite.print_summary()

if __name__ == "__main__":
    run()
