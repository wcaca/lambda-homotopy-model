"""
audio_filter_demo.py — 小波基 λ单元音频低通滤波端到端演示
======================================================================
任务：
  用 DCT 基的 λ单元对含噪音频信号做连续强度低通滤波。
  t=0 → 全频率保留（原始信号）
  t=1 → 强低通滤波（高频噪声抑制）
  中间 → 连续渐变截止频率

量化指标：
  - 信噪比（SNR）对比
  - 高频能量占比（随 t 增大应下降）
  - 低频能量保留率（随 t 增大应稳定）

零LLM，纯 numpy（不依赖 librosa/soundfile 生成合成信号）。

用法：
  python audio_filter_demo.py               # 合成信号演示
  python audio_filter_demo.py --wav x.wav   # 真实音频（需 soundfile）
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from wavelet_basis import WaveletBasisBuilder


# ══════════════════════════════════════════════════════════════════════
# 合成音频信号生成
# ══════════════════════════════════════════════════════════════════════
def make_synthetic_signal(
    sr:          int   = 8000,    # 采样率
    duration:    float = 0.5,     # 时长（秒）
    f_low:       float = 200.0,   # 低频信号频率（Hz）
    f_high:      float = 3000.0,  # 高频噪声频率（Hz）
    noise_ratio: float = 0.4,     # 高频噪声幅值/低频信号幅值
    seed:        int   = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (clean_signal, noisy_signal, t_axis)
    clean = 纯低频正弦波
    noisy = clean + 高频噪声
    """
    n   = int(sr * duration)
    t   = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(seed)

    clean = np.sin(2 * np.pi * f_low  * t).astype(np.float32)
    hf    = np.sin(2 * np.pi * f_high * t).astype(np.float32) * noise_ratio
    noise = rng.normal(0, noise_ratio * 0.5, n).astype(np.float32)
    noisy = clean + hf + noise

    return clean, noisy, t


def snr_db(signal: np.ndarray, reference: np.ndarray) -> float:
    """信噪比（dB），reference 为干净信号"""
    noise  = signal - reference
    s_pow  = np.mean(reference ** 2)
    n_pow  = np.mean(noise ** 2)
    if n_pow < 1e-12:
        return float("inf")
    return float(10 * np.log10(s_pow / n_pow))


def high_freq_ratio(signal: np.ndarray, n_fft: int, cutoff_bin: int) -> float:
    """高频能量占总能量比例（cutoff_bin 以上为高频）"""
    spec  = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
    total = spec.sum() + 1e-10
    return float(spec[cutoff_bin:].sum() / total)


# ══════════════════════════════════════════════════════════════════════
# 音频滤波演示器
# ══════════════════════════════════════════════════════════════════════
class AudioFilterDemo:
    """
    用 DCT 基的 λ单元对音频帧做低通滤波。
    帧长 = frame_size，逐帧处理，结果拼接。
    """

    def __init__(
        self,
        frame_size:  int = 256,
        compact_dim: int = 64,
        low_k:       int = 8,
        seed:        int = 42,
    ):
        self.frame_size  = frame_size
        self.compact_dim = compact_dim
        self.low_k       = low_k

        builder = WaveletBasisBuilder(
            origin_dim  = frame_size,
            compact_dim = compact_dim,
            basis_type  = "dct",
            low_k       = low_k,
            seed        = seed,
        )
        self.W0, self.W1 = builder.build_weights()
        print(f"  [AudioFilter] frame={frame_size}  compact={compact_dim}  low_k={low_k}")

    def _filter_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        W0f = self.W0.astype(np.float32)
        W1f = self.W1.astype(np.float32)
        W_t = (1 - t) * W0f + t * W1f

        x     = frame.astype(np.float32)
        proj  = x @ W_t               # (compact_dim,)
        recon = proj @ W_t.T          # (frame_size,) 伪逆重建

        # 能量保持
        scale  = np.linalg.norm(x) / (np.linalg.norm(recon) + 1e-8)
        return (recon * scale).astype(np.float32)

    def filter(self, signal: np.ndarray, t: float) -> np.ndarray:
        """逐帧低通滤波，返回与输入等长的信号"""
        fs     = self.frame_size
        n      = len(signal)
        out    = np.zeros(n, dtype=np.float32)
        count  = np.zeros(n, dtype=np.float32)

        # 50% 重叠帧（减少块边界伪影）
        hop = fs // 2
        for start in range(0, n, hop):
            end   = min(start + fs, n)
            frame = np.zeros(fs, dtype=np.float32)
            frame[:end-start] = signal[start:end]
            proc  = self._filter_frame(frame, t)
            out[start:end]   += proc[:end-start]
            count[start:end] += 1.0

        count = np.maximum(count, 1.0)
        return out / count

    def run_demo(
        self,
        clean:    np.ndarray,
        noisy:    np.ndarray,
        sr:       int  = 8000,
        t_values: list = None,
    ) -> list[dict]:
        t_values  = t_values or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        n_fft     = 512
        cutoff    = n_fft // 8   # 前 1/8 为低频

        snr_in  = snr_db(noisy, clean)
        hfr_in  = high_freq_ratio(noisy, n_fft, cutoff)
        print(f"\n  输入含噪信号 SNR={snr_in:.2f}dB  高频能量占比={hfr_in:.4f}")
        print(f"\n  {'t':>5}  {'SNR(dB)':>9}  {'高频能量比':>10}  {'低频保留率':>10}  效果")
        print(f"  {'─'*5}  {'─'*9}  {'─'*10}  {'─'*10}  ──────")

        results = []
        for t in t_values:
            filtered = self.filter(noisy, t)
            s        = snr_db(filtered, clean)
            hfr      = high_freq_ratio(filtered, n_fft, cutoff)
            # 低频保留率：filtered 低频能量 / clean 低频能量
            spec_f   = np.abs(np.fft.rfft(filtered, n=n_fft)) ** 2
            spec_c   = np.abs(np.fft.rfft(clean,    n=n_fft)) ** 2
            lfr      = float(spec_f[:cutoff].sum() / (spec_c[:cutoff].sum() + 1e-10))
            lfr      = min(lfr, 2.0)   # 截断异常值

            if s > snr_in + 0.5:
                effect = "↑改善"
            elif s > snr_in - 0.5:
                effect = "─持平"
            else:
                effect = "↓下降"

            results.append({"t": t, "snr": round(s,2),
                             "hfr": round(hfr,4), "lfr": round(lfr,4)})
            print(f"  {t:>5.2f}  {s:>9.2f}  {hfr:>10.4f}  {lfr:>10.4f}  {effect}")

        return results


# ════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav",    type=str,  default=None)
    parser.add_argument("--sr",     type=int,  default=8000)
    parser.add_argument("--frame",  type=int,  default=256)
    parser.add_argument("--low-k",  type=int,  default=6)
    parser.add_argument("--noise",  type=float,default=0.4)
    args = parser.parse_args()

    SEP = "═" * 56
    print(f"\n{SEP}")
    print("  λ单元音频低通滤波演示（小波DCT基，零LLM）")
    print(SEP)

    if args.wav and os.path.exists(args.wav):
        try:
            import soundfile as sf
            raw, sr = sf.read(args.wav, always_2d=False)
            raw = raw.astype(np.float32)
            if raw.ndim > 1:
                raw = raw[:, 0]
            clean  = raw / (np.max(np.abs(raw)) + 1e-8)
            rng    = np.random.default_rng(0)
            noise  = rng.normal(0, args.noise * 0.5, len(clean)).astype(np.float32)
            noisy  = clean + noise
            sr_use = sr
            print(f"  真实音频: {args.wav}  时长={len(clean)/sr:.2f}s  sr={sr}")
        except ImportError:
            print("  soundfile 未安装，切换为合成信号")
            clean, noisy, _ = make_synthetic_signal(args.sr, noise_ratio=args.noise)
            sr_use = args.sr
    else:
        print(f"  使用合成信号（sr={args.sr}，噪声比={args.noise}）")
        clean, noisy, _ = make_synthetic_signal(args.sr, noise_ratio=args.noise)
        sr_use = args.sr

    demo    = AudioFilterDemo(frame_size=args.frame, low_k=args.low_k)
    results = demo.run_demo(clean, noisy, sr=sr_use)

    # 结论
    best = max(results, key=lambda r: r["snr"])
    snr_gain = best["snr"] - snr_db(noisy, clean)
    hfr_drop = results[0]["hfr"] - results[-1]["hfr"]
    print(f"\n  最佳 t={best['t']}  SNR={best['snr']}dB  SNR提升={snr_gain:+.2f}dB")
    print(f"  高频能量比变化: {results[0]['hfr']:.4f} → {results[-1]['hfr']:.4f}  "
          f"（下降{hfr_drop:.4f}）")

    if abs(hfr_drop) > 0.01:
        print(f"\n  结论: ✓ t 形变有效控制高频能量（低通效果实测）")
    else:
        print(f"\n  结论: △ 高频变化不明显，建议降低 --low-k")

    print(f"\n{SEP}\n")
    return results


if __name__ == "__main__":
    main()
