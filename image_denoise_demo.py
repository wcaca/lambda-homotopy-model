"""
image_denoise_demo.py — 小波基 λ单元图像降噪端到端演示
======================================================================
任务：
  用小波(DCT)基的 λ单元，对含噪图像做连续强度降噪。
  t=0 → 原始信号（全频保留）
  t=1 → 强低通滤波（高频抑制）
  中间 → 连续渐变降噪强度

这是第一个"存储即模型"的端到端功能性验证：
  - 用 BasisFactory 把小波基写入 *.bin 单元
  - 用 ModelScheduler 调度整条链路
  - 输入含噪图像，输出不同 t 下的降噪结果
  - 量化 PSNR、噪声能量比，证明 t 形变有实际物理效果

零LLM，纯 numpy + Pillow。

用法：
  python image_denoise_demo.py                  # 生成合成含噪图像并演示
  python image_denoise_demo.py --image xx.jpg   # 使用真实图片
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from PIL import Image
from basis_factory import BasisFactory
from base_utils import LambdaUnit
from wavelet_basis import WaveletBasisBuilder


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════
def psnr(img_ref: np.ndarray, img_noisy: np.ndarray) -> float:
    """峰值信噪比（越高=越接近参考图）"""
    ref  = img_ref.astype(np.float32)
    noisy = img_noisy.astype(np.float32)
    mse  = np.mean((ref - noisy) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def noise_energy_ratio(img_orig: np.ndarray, img_proc: np.ndarray,
                        img_clean: np.ndarray) -> float:
    """
    噪声去除率 = 1 - (处理后与干净图的差) / (含噪与干净图的差)
    越接近1 = 去噪越彻底
    """
    orig_noise = np.linalg.norm(img_orig.astype(np.float32) - img_clean.astype(np.float32))
    proc_noise = np.linalg.norm(img_proc.astype(np.float32) - img_clean.astype(np.float32))
    return float(1.0 - proc_noise / (orig_noise + 1e-8))


def make_synthetic_image(size: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    生成合成测试图像对（干净版 + 含噪版）
    干净图：低频渐变 + 简单几何图案（明确的低频结构）
    含噪图：干净图 + 高斯噪声（sigma=40）
    """
    rng = np.random.default_rng(seed)
    # 干净图：渐变 + 圆形图案
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    clean = (
        128 + 60 * np.sin(2 * np.pi * x * 1.5) * np.cos(2 * np.pi * y * 1.5)
        + 40 * ((x - 0.5)**2 + (y - 0.5)**2 < 0.1).astype(float)
    ).astype(np.uint8)
    # 含噪图
    noise = rng.normal(0, 40, clean.shape).astype(np.float32)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return clean, noisy


class ImageDenoiseDemo:
    """
    图像降噪演示器
    内部用 WaveletBasisBuilder 直接构造投影矩阵，
    不依赖完整的 model_root 调度器，保持独立可运行。
    """

    def __init__(
        self,
        patch_size: int = 64,   # 图像块大小（同时也是 λ单元的 origin_dim）
        low_k:      int = 8,    # W1 保留的低频分量数
        seed:       int = 42,
    ):
        self.patch_size = patch_size
        self.dim        = patch_size * patch_size   # 展平后的向量维度
        # compact_dim 选 patch_size（压缩一半）
        self.compact_dim = patch_size
        self.low_k       = low_k
        self.seed        = seed

        # 构造小波基 W0/W1
        builder    = WaveletBasisBuilder(
            origin_dim  = self.dim,
            compact_dim = self.compact_dim,
            basis_type  = "dct",
            low_k       = self.low_k,
            seed        = self.seed,
        )
        self.W0, self.W1 = builder.build_weights()
        print(f"  [ImageDenoiser] patch={patch_size}px  "
              f"dim={self.dim}→{self.compact_dim}  low_k={low_k}")

    def _forward_patch(self, patch: np.ndarray, t: float) -> np.ndarray:
        """对单个图像块做 t 形变前向计算"""
        W0f = self.W0.astype(np.float32)
        W1f = self.W1.astype(np.float32)
        W_t = (1 - t) * W0f + t * W1f

        x   = patch.astype(np.float32).flatten() / 255.0
        # 投影到 compact_dim，再伪逆投回 origin_dim
        # 伪逆：W^T（正交基的伪逆就是转置）
        proj  = x @ W_t                      # (compact_dim,)
        recon = proj @ W_t.T                 # (dim,) 近似重建

        # 能量归一化（避免整体亮度偏移）
        scale  = np.linalg.norm(x) / (np.linalg.norm(recon) + 1e-8)
        recon *= scale

        out = np.clip(recon * 255.0, 0, 255).astype(np.uint8)
        return out.reshape(self.patch_size, self.patch_size)

    def denoise(self, img_arr: np.ndarray, t: float) -> np.ndarray:
        """
        对整幅图像做降噪（逐块处理）
        img_arr: (H, W) uint8 灰度图
        """
        H, W = img_arr.shape
        ps   = self.patch_size
        out  = np.zeros_like(img_arr)

        for r in range(0, H, ps):
            for c in range(0, W, ps):
                # 提取块（边界自动填充）
                r_end = min(r + ps, H)
                c_end = min(c + ps, W)
                patch = np.zeros((ps, ps), dtype=np.uint8)
                patch[:r_end-r, :c_end-c] = img_arr[r:r_end, c:c_end]

                proc = self._forward_patch(patch, t)
                out[r:r_end, c:c_end] = proc[:r_end-r, :c_end-c]

        return out

    def run_demo(
        self,
        clean_arr:  np.ndarray,
        noisy_arr:  np.ndarray,
        t_values:   list = None,
        save_dir:   str  = None,
    ) -> list[dict]:
        """
        在多个 t 值上运行降噪演示，返回量化指标列表。
        """
        t_values = t_values or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results  = []

        psnr_noisy = psnr(clean_arr, noisy_arr)
        print(f"\n  含噪图 PSNR vs 干净图: {psnr_noisy:.2f} dB  "
              f"(噪声能量: {np.std(noisy_arr.astype(np.float32)-clean_arr.astype(np.float32)):.1f})")
        print(f"\n  {'t':>5}  {'PSNR(dB)':>10}  {'去噪率':>8}  {'输出能量':>10}  效果")
        print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*10}  ──────")

        for t in t_values:
            denoised = self.denoise(noisy_arr, t)
            p        = psnr(clean_arr, denoised)
            ner      = noise_energy_ratio(noisy_arr, denoised, clean_arr)
            energy   = float(np.std(denoised.astype(np.float32)))

            # 效果评估
            if p > psnr_noisy + 1.0:
                effect = "↑改善"
            elif p > psnr_noisy - 0.5:
                effect = "─持平"
            else:
                effect = "↓下降"

            results.append({
                "t":      t,
                "psnr":   round(p, 2),
                "ner":    round(ner, 4),
                "energy": round(energy, 2),
            })
            print(f"  {t:>5.2f}  {p:>10.2f}  {ner:>8.4f}  {energy:>10.2f}  {effect}")

            # 保存输出图像
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(denoised).save(
                    os.path.join(save_dir, f"denoised_t{t:.1f}.png")
                )

        return results


# ════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="小波基λ单元图像降噪演示")
    parser.add_argument("--image",    type=str,  default=None,  help="输入图片路径（不填则生成合成图）")
    parser.add_argument("--size",     type=int,  default=64,    help="合成图像尺寸（默认64×64）")
    parser.add_argument("--patch",    type=int,  default=16,    help="处理块大小（默认16）")
    parser.add_argument("--low-k",    type=int,  default=4,     help="W1 低频保留数（默认4）")
    parser.add_argument("--noise",    type=float,default=40.0,  help="合成图像噪声强度（默认40）")
    parser.add_argument("--save-dir", type=str,  default=None,  help="结果图保存目录")
    args = parser.parse_args()

    SEP = "═" * 56
    print(f"\n{SEP}")
    print("  λ单元图像降噪演示（小波DCT基，零LLM）")
    print(SEP)

    # ── 准备图像 ────────────────────────────────────────────────────
    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("L")
        clean_arr = np.array(img)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, args.noise, clean_arr.shape).astype(np.float32)
        noisy_arr = np.clip(clean_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        print(f"  输入图片: {args.image}  尺寸: {clean_arr.shape}")
    else:
        print(f"  使用合成图像（{args.size}×{args.size}，噪声σ={args.noise}）")
        clean_arr, noisy_arr = make_synthetic_image(args.size, seed=0)
        # 覆盖合成噪声强度
        if args.noise != 40.0:
            rng = np.random.default_rng(0)
            noise = rng.normal(0, args.noise, clean_arr.shape).astype(np.float32)
            noisy_arr = np.clip(clean_arr.astype(np.float32)+noise, 0, 255).astype(np.uint8)

    # ── 初始化演示器 ────────────────────────────────────────────────
    demo = ImageDenoiseDemo(
        patch_size = args.patch,
        low_k      = args.low_k,
    )

    # ── 运行 ────────────────────────────────────────────────────────
    results = demo.run_demo(
        clean_arr = clean_arr,
        noisy_arr = noisy_arr,
        t_values  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        save_dir  = args.save_dir,
    )

    # ── 结论 ────────────────────────────────────────────────────────
    best  = max(results, key=lambda r: r["psnr"])
    worst = min(results, key=lambda r: r["psnr"])
    psnr_range = best["psnr"] - worst["psnr"]

    print(f"\n  最佳 t={best['t']}  PSNR={best['psnr']}dB")
    print(f"  t 形变 PSNR 变化范围: {psnr_range:.2f}dB")
    print(f"  t=0→1 最大去噪率变化: "
          f"{results[0]['ner']:.4f} → {results[-1]['ner']:.4f}")

    # 判定：t 形变是否产生实际效果（PSNR 有变化）
    if psnr_range > 0.5:
        print(f"\n  结论: ✓ t 形变对图像降噪有实际物理效果（PSNR变化>{psnr_range:.2f}dB）")
    else:
        print(f"\n  结论: △ t 形变效果弱（PSNR变化<0.5dB），建议调小 --low-k 或调大 --noise")

    if args.save_dir:
        # 额外保存干净图和含噪图
        Image.fromarray(clean_arr).save(os.path.join(args.save_dir, "clean.png"))
        Image.fromarray(noisy_arr).save(os.path.join(args.save_dir, "noisy.png"))
        print(f"\n  图像已保存至: {args.save_dir}")

    print(f"\n{SEP}\n")
    return results


if __name__ == "__main__":
    main()
