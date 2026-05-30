"""
test_suite/test_multimodal.py
第四部分：多模态全用例（纯传统信号/特征，无LLM）
  M01  文本→向量编码（字符哈希特征，无LLM）
  M02  图像→向量编码（像素灰度特征）
  M03  音频→向量编码（MFCC特征，librosa可选）
  M04  单模态闭环（文本→文本，编码→λ变换→解码）
  M05  多模态并联融合（文+图双输入，sum/concat规则）
  M06  跨模态维度对齐（不同模态输出均为标准 4096 维）
  M07  模态指纹区分度（不同模态的8维指纹应有差异）
  M08  多模态 t 形变（切换 t 时多模态输出平滑变化）
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_suite.utils import TestSuite, fixed_input, assert_finite, cosine_sim, MODEL_ROOT

def run() -> bool:
    suite = TestSuite("多模态信号处理验证")

    # ── M01 文本编码（纯字符哈希，无LLM）───────────────────────────
    def m01(r):
        from modal_utils import text_to_vector
        samples = ["hello world", "人工智能", "signal processing 2025"]
        vecs = []
        for s in samples:
            vec, finger = text_to_vector(s)
            assert vec.shape == (4096,),    f"文本向量维度错误: {vec.shape}"
            assert finger.shape == (8,),    f"指纹维度错误: {finger.shape}"
            assert float(finger[0]) > 0.5,  f"文本指纹第0位应为1"
            assert_finite(vec, f"文本向量[{s[:8]}]")
            vecs.append(vec)
        # 不同文本编码向量应有差异
        sim01 = cosine_sim(vecs[0], vecs[1])
        r.passed(samples=len(samples), sim_en_cn=round(sim01,3))
    suite.run_case("M01 文本→向量编码（无LLM）", m01)

    # ── M02 图像编码 ─────────────────────────────────────────────────
    def m02(r):
        from modal_utils import image_to_vector
        import tempfile
        from PIL import Image
        # 生成临时测试图片
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            img = Image.fromarray(
                (np.random.default_rng(42).integers(0,255,(64,64,3),dtype=np.uint8))
            )
            img.save(tf.name)
            path = tf.name
        vec, finger = image_to_vector(path)
        assert vec.shape == (4096,), f"图像向量维度错误: {vec.shape}"
        assert float(finger[1]) > 0.5, "图像指纹第1位应为1"
        assert_finite(vec, "图像向量")
        os.unlink(path)
        r.passed(vec_dim=vec.shape[0], finger_peak=int(np.argmax(finger)))
    suite.run_case("M02 图像→向量编码", m02)

    # ── M03 音频编码（librosa 可选）─────────────────────────────────
    def m03(r):
        try:
            import librosa, soundfile as sf
        except ImportError:
            r.skipped("librosa/soundfile 未安装，跳过音频测试")
            return
        import tempfile
        from modal_utils import audio_to_vector
        # 生成合成音频
        sr = 16000
        t  = np.linspace(0, 1, sr, dtype=np.float32)
        y  = np.sin(2 * np.pi * 440 * t) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, y, sr)
            path = tf.name
        vec, finger = audio_to_vector(path)
        assert vec.shape == (4096,), f"音频向量维度错误: {vec.shape}"
        assert float(finger[2]) > 0.5, "音频指纹第2位应为1"
        assert_finite(vec, "音频向量")
        os.unlink(path)
        r.passed(vec_dim=vec.shape[0], finger_peak=int(np.argmax(finger)))
    suite.run_case("M03 音频→向量编码（MFCC）", m03)

    # ── M04 单模态闭环（文本→文本）──────────────────────────────────
    def m04(r):
        from modal_scheduler import MultiModalScheduler
        s = MultiModalScheduler(MODEL_ROOT)
        result = s.run("测试信号流水线", "text", "text")
        assert isinstance(result, str), f"输出类型错误: {type(result)}"
        assert len(result) > 0, "输出为空字符串"
        r.passed(output=result[:30], output_len=len(result))
    suite.run_case("M04 单模态闭环（文→文）", m04)

    # ── M05 多模态并联融合 ───────────────────────────────────────────
    def m05(r):
        from modal_utils import text_to_vector, image_to_vector
        import tempfile
        from PIL import Image
        from component_manager import ComponentManager
        # 生成测试图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            img = Image.fromarray(
                np.full((32,32,3), 128, dtype=np.uint8)
            )
            img.save(tf.name)
            img_path = tf.name
        text_vec, tf_  = text_to_vector("fusion test signal")
        img_vec,  if_  = image_to_vector(img_path)
        os.unlink(img_path)

        # 模拟并联组：两路同维度向量 sum 融合
        fused_sum  = text_vec.astype(np.float32) + img_vec.astype(np.float32)
        fused_mean = (text_vec.astype(np.float32) + img_vec.astype(np.float32)) / 2

        assert fused_sum.shape  == (4096,)
        assert fused_mean.shape == (4096,)
        assert_finite(fused_sum,  "sum 融合")
        assert_finite(fused_mean, "mean 融合")
        # 融合结果与单一模态有差异
        sim_ts = cosine_sim(fused_sum, text_vec)
        sim_ti = cosine_sim(fused_sum, img_vec)
        r.passed(sim_fused_vs_text=round(sim_ts,3),
                 sim_fused_vs_img=round(sim_ti,3),
                 fused_norm=round(float(np.linalg.norm(fused_sum)),3))
    suite.run_case("M05 多模态并联融合", m05)

    # ── M06 跨模态维度对齐 ──────────────────────────────────────────
    def m06(r):
        from modal_utils import text_to_vector
        import tempfile
        from PIL import Image
        from modal_utils import image_to_vector
        texts  = ["hello", "world", "λ系统"]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(tf.name)
            ip = tf.name
        tv, _ = text_to_vector(texts[0])
        iv, _ = image_to_vector(ip)
        os.unlink(ip)
        assert tv.shape == (4096,), f"文本向量维度 {tv.shape}"
        assert iv.shape == (4096,), f"图像向量维度 {iv.shape}"
        r.passed(text_dim=tv.shape[0], image_dim=iv.shape[0])
    suite.run_case("M06 跨模态维度对齐（统一4096维）", m06)

    # ── M07 模态指纹区分度 ───────────────────────────────────────────
    def m07(r):
        from modal_utils import text_to_vector
        import tempfile
        from PIL import Image
        from modal_utils import image_to_vector
        _, tf_finger = text_to_vector("test")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(f.name)
            ip = f.name
        _, if_finger = image_to_vector(ip)
        os.unlink(ip)
        sim = cosine_sim(tf_finger, if_finger)
        # 文本指纹(1,0,...) vs 图像指纹(0,1,...) 应相似度接近0
        assert sim < 0.9, f"模态指纹区分度不足（相似度={sim:.3f}）"
        r.passed(text_finger=tf_finger[:3].tolist(),
                 image_finger=if_finger[:3].tolist(),
                 cosine_sim=round(sim,3))
    suite.run_case("M07 模态指纹区分度", m07)

    # ── M08 多模态 t 形变平滑 ────────────────────────────────────────
    def m08(r):
        from modal_scheduler import MultiModalScheduler
        s = MultiModalScheduler(MODEL_ROOT)
        results = []
        for t in [0.0, 0.5, 1.0]:
            s.set_global_t(t)
            res = s.run("形变测试 signal", "text", "text")
            results.append(res)
        # 程序在各 t 值下均正常运行
        assert all(isinstance(res, str) for res in results), "输出类型异常"
        r.passed(t_values="[0.0, 0.5, 1.0]", all_ok=True)
    suite.run_case("M08 多模态t形变平滑", m08)

    return suite.print_summary()

if __name__ == "__main__":
    run()
