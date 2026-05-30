#!/usr/bin/env python3
"""
lambda_pipeline.py — λ单元信号处理流水线命令行工具
======================================================================
用法：
  # 图像降噪
  python lambda_pipeline.py denoise  --input photo.jpg --output out/ --t 0.6

  # 音频低通滤波
  python lambda_pipeline.py filter   --input voice.wav --output out/ --t 0.5

  # 信号分类（多文件批量，csv 标签）
  python lambda_pipeline.py classify --input data.npy  --labels labels.npy

  # 批量对整个目录处理
  python lambda_pipeline.py denoise  --input ./photos/ --output ./denoised/ --t 0.7

  # 对比多个 t 值（生成对比报告）
  python lambda_pipeline.py denoise  --input photo.jpg --output out/ --sweep

  # 查看基矩阵对比报告
  python lambda_pipeline.py compare  --dim 128 --compact 32

λ函数单元 + 同伦形变大模型系统 · 命令行工具 v1.8
"""

from __future__ import annotations
import os, sys, argparse, time, json
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from PIL import Image
from image_denoise_demo import ImageDenoiseDemo, make_synthetic_image, psnr
from audio_filter_demo  import AudioFilterDemo, make_synthetic_signal, snr_db
from classify_demo      import knn_accuracy, separation_ratio
from lda_basis          import LDABasisBuilder
from basis_factory      import BasisFactory

# ── 颜色 ─────────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; RESET = "\033[0m"
BOLD = "\033[1m"

def hdr(msg): print(f"\n{B}{BOLD}{msg}{RESET}")
def ok(msg):  print(f"  {G}✓{RESET} {msg}")
def warn(msg):print(f"  {Y}△{RESET} {msg}")
def err(msg): print(f"  {R}✗{RESET} {msg}")


# ══════════════════════════════════════════════════════════════════════
# 子命令：denoise（图像降噪）
# ══════════════════════════════════════════════════════════════════════
def cmd_denoise(args):
    hdr("图像降噪 · 小波DCT基λ单元")

    # 收集输入文件
    if os.path.isdir(args.input):
        exts  = {".jpg",".jpeg",".png",".bmp",".tiff"}
        files = [os.path.join(args.input, f)
                 for f in sorted(os.listdir(args.input))
                 if os.path.splitext(f)[1].lower() in exts]
        if not files:
            err(f"目录 {args.input} 中未找到图片文件")
            return
    elif os.path.exists(args.input):
        files = [args.input]
    else:
        warn(f"未找到文件 {args.input}，使用合成测试图像")
        files = ["__synthetic__"]

    os.makedirs(args.output, exist_ok=True)
    demo  = ImageDenoiseDemo(patch_size=args.patch, low_k=args.low_k)

    t_vals = list(np.linspace(0, 1, 7)) if args.sweep else [args.t]
    summary = []

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"\n  处理: {fname}")

        if fpath == "__synthetic__":
            clean, noisy = make_synthetic_image(args.synth_size, seed=0)
            # 额外加更强噪声展示效果
            rng   = np.random.default_rng(99)
            noise = rng.normal(0, args.noise, clean.shape).astype(np.float32)
            noisy = np.clip(clean.astype(np.float32)+noise, 0, 255).astype(np.uint8)
            stem  = "synthetic"
        else:
            img   = Image.open(fpath).convert("L")
            clean = np.array(img)
            rng   = np.random.default_rng(0)
            noise = rng.normal(0, args.noise, clean.shape).astype(np.float32)
            noisy = np.clip(clean.astype(np.float32)+noise, 0, 255).astype(np.uint8)
            stem  = os.path.splitext(fname)[0]

        psnr_in = psnr(clean, noisy)
        # 保存含噪原图
        Image.fromarray(noisy).save(os.path.join(args.output, f"{stem}_noisy.png"))

        results = []
        for t in t_vals:
            denoised = demo.denoise(noisy, t)
            p        = psnr(clean, denoised)
            results.append({"t": round(t,2), "psnr": round(p,2)})
            out_name = f"{stem}_t{t:.2f}.png"
            Image.fromarray(denoised).save(os.path.join(args.output, out_name))

        best = max(results, key=lambda r: r["psnr"])
        gain = best["psnr"] - psnr_in
        summary.append({"file": fname, "psnr_in": round(psnr_in,2),
                         "psnr_best": best["psnr"], "best_t": best["t"],
                         "gain": round(gain,2)})

        ok(f"含噪 {psnr_in:.2f}dB → 最佳 {best['psnr']:.2f}dB "
           f"(t={best['t']}, 提升{gain:+.2f}dB)")
        if args.sweep:
            print(f"    {'t':>5} " + " ".join(f"{r['t']:>5}" for r in results))
            print(f"    {'PSNR':>5} " + " ".join(f"{r['psnr']:>5}" for r in results))

    # 保存 JSON 报告
    report_path = os.path.join(args.output, "denoise_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    hdr("完成")
    ok(f"处理 {len(files)} 张图像  输出目录: {args.output}")
    ok(f"报告: {report_path}")
    avg_gain = sum(r["gain"] for r in summary) / len(summary)
    print(f"  平均 PSNR 提升: {avg_gain:+.2f}dB")


# ══════════════════════════════════════════════════════════════════════
# 子命令：filter（音频低通滤波）
# ══════════════════════════════════════════════════════════════════════
def cmd_filter(args):
    hdr("音频低通滤波 · 小波DCT基λ单元")

    demo = AudioFilterDemo(
        frame_size  = args.frame,
        compact_dim = args.frame // 2,
        low_k       = args.low_k,
    )

    if args.input and os.path.exists(args.input):
        try:
            import soundfile as sf
            raw, sr = sf.read(args.input, always_2d=False)
            raw = raw.astype(np.float32)
            if raw.ndim > 1: raw = raw[:,0]
            clean  = raw / (np.max(np.abs(raw)) + 1e-8)
            rng    = np.random.default_rng(0)
            noisy  = clean + rng.normal(0, args.noise, len(clean)).astype(np.float32)
            print(f"  真实音频: {args.input}  时长={len(clean)/sr:.2f}s  sr={sr}")
        except ImportError:
            warn("soundfile 未安装，使用合成信号")
            clean, noisy, _ = make_synthetic_signal(args.sr, noise_ratio=args.noise)
            sr = args.sr
    else:
        warn(f"未找到音频文件，使用合成信号（sr={args.sr}）")
        clean, noisy, _ = make_synthetic_signal(args.sr, noise_ratio=args.noise)
        sr = args.sr

    snr_in  = snr_db(noisy, clean)
    t_vals  = list(np.linspace(0, 1, 7)) if args.sweep else [args.t]

    print(f"  输入 SNR: {snr_in:.2f}dB")
    results = []
    for t in t_vals:
        filtered = demo.filter(noisy, t)
        s        = snr_db(filtered, clean)
        results.append({"t": round(t,2), "snr": round(s,2)})

    best     = max(results, key=lambda r: r["snr"])
    snr_gain = best["snr"] - snr_in
    ok(f"含噪 {snr_in:.2f}dB → 最佳 {best['snr']:.2f}dB (t={best['t']}, 提升{snr_gain:+.2f}dB)")

    if args.sweep:
        print(f"    {'t':>5} " + "  ".join(f"{r['t']:>5}" for r in results))
        print(f"    {'SNR':>5} " + "  ".join(f"{r['snr']:>5}" for r in results))

    # 保存最佳结果
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        best_filtered = demo.filter(noisy, best["t"])
        try:
            import soundfile as sf
            out_path = os.path.join(args.output, f"filtered_t{best['t']:.2f}.wav")
            sf.write(out_path, best_filtered, sr)
            ok(f"已保存: {out_path}")
        except ImportError:
            warn("soundfile 未安装，跳过音频保存（numpy 数组已在内存中）")

    hdr("完成")


# ══════════════════════════════════════════════════════════════════════
# 子命令：classify（信号分类）
# ══════════════════════════════════════════════════════════════════════
def cmd_classify(args):
    hdr("信号分类路由 · LDA判别基λ单元")

    # 加载或生成数据
    if args.input and os.path.exists(args.input):
        X = np.load(args.input).astype(np.float32)
        if args.labels and os.path.exists(args.labels):
            y = np.load(args.labels).astype(np.int32)
        else:
            err("分类任务需要 --labels 标签文件（numpy .npy）")
            return
        print(f"  数据: {X.shape}  类别数: {len(np.unique(y))}")
    else:
        warn("未提供数据文件，使用合成多类数据演示")
        rng     = np.random.default_rng(42)
        n_cls   = args.n_classes
        dim     = args.dim
        centers = rng.standard_normal((n_cls, dim)).astype(np.float32) * 1.5
        X_list, y_list = [], []
        for c, mu in enumerate(centers):
            Xi = rng.standard_normal((80, dim)).astype(np.float32) * args.noise + mu
            X_list.append(Xi); y_list.append(np.full(80, c, np.int32))
        X = np.vstack(X_list); y = np.hstack(y_list)
        dim = X.shape[1]

    # 分层切分
    rng    = np.random.default_rng(0)
    idx    = rng.permutation(len(X))
    split  = int(len(X) * 0.75)
    Xtr, ytr = X[idx[:split]], y[idx[:split]]
    Xte, yte = X[idx[split:]], y[idx[split:]]

    dim     = Xtr.shape[1]
    compact = args.compact or max(8, dim // 8)

    builder = LDABasisBuilder(dim, compact, seed=42)
    builder.fit(Xtr, ytr)
    W0, W1 = builder.build_weights()

    acc_raw = knn_accuracy(Xtr, ytr, Xte, yte)
    print(f"  原始 {dim}D 准确率: {acc_raw:.4f}")

    t_vals  = list(np.linspace(0, 1, 7)) if args.sweep else [args.t]
    results = []
    for t in t_vals:
        Wt  = (1-t)*W0.astype(np.float32) + t*W1.astype(np.float32)
        acc = knn_accuracy(Xtr@Wt, ytr, Xte@Wt, yte)
        sep = separation_ratio(Xtr@Wt, ytr)
        results.append({"t": round(t,2), "acc": round(acc,4), "sep": round(sep,4)})

    best     = max(results, key=lambda r: r["acc"])
    acc_gain = best["acc"] - acc_raw
    ok(f"最佳 t={best['t']}  准确率={best['acc']:.4f}  vs 原始{acc_raw:.4f}  Δ={acc_gain:+.4f}")

    if args.sweep:
        print(f"\n  {'t':>5}  {'准确率':>8}  {'分离比':>10}")
        for r in results:
            print(f"  {r['t']:>5.2f}  {r['acc']:>8.4f}  {r['sep']:>10.4f}")

    hdr("完成")


# ══════════════════════════════════════════════════════════════════════
# 子命令：compare（基矩阵对比）
# ══════════════════════════════════════════════════════════════════════
def cmd_compare(args):
    hdr("基矩阵对比报告（PCA / 小波 / LDA / 随机）")

    factory = BasisFactory(
        origin_dim        = args.dim,
        compact_dim       = args.compact,
        pca_n_samples     = 300,
        pca_n_signal_dims = min(32, args.compact // 2),
        lda_n_classes     = 4,
        lda_n_per_class   = 60,
    )
    report = factory.compare_bases()

    print(f"\n  {'基':12s}  {'W0重建误差':>12s}  {'W1重建误差':>12s}  "
          f"{'W0-W1相似度':>12s}  {'压缩增量':>8s}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")
    for basis, stat in report.items():
        if "error" in stat:
            print(f"  {basis:12s}  ERROR: {stat['error']}")
        else:
            marker = " ←推荐" if basis == "pca" else ""
            print(f"  {basis:12s}  {stat['W0_recon_err']:>12.4f}  {stat['W1_recon_err']:>12.4f}  "
                  f"{stat['W0_W1_cos_sim']:>12.4f}  {stat['compress_delta']:>+8.4f}{marker}")

    print(f"\n  解读：")
    print(f"    W0重建误差越小 = 保留信息越多（PCA/小波 优于随机）")
    print(f"    W0-W1相似度越低 = t 形变幅度越大（LDA/小波 变化空间更大）")
    print(f"    压缩增量(+) = W1 比 W0 损失更多信息（正常，W1 是压缩态）")
    hdr("完成")


# ══════════════════════════════════════════════════════════════════════
# 参数解析 + 主入口
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        prog        = "lambda_pipeline",
        description = "λ单元信号处理流水线命令行工具 v1.8",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
示例：
  python lambda_pipeline.py denoise  --input photo.jpg --output out/ --t 0.6
  python lambda_pipeline.py denoise  --input photo.jpg --output out/ --sweep
  python lambda_pipeline.py filter   --sr 8000 --t 0.5 --sweep
  python lambda_pipeline.py classify --n-classes 4 --dim 32 --noise 1.5 --sweep
  python lambda_pipeline.py compare  --dim 64 --compact 16
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── denoise ──────────────────────────────────────────────────────
    p_dn = sub.add_parser("denoise", help="图像降噪（小波DCT基）")
    p_dn.add_argument("--input",      default="",       help="输入图片或目录（不填=合成图）")
    p_dn.add_argument("--output",     default="./out",  help="输出目录")
    p_dn.add_argument("--t",          type=float, default=0.0,  help="形变强度 [0,1]（default:0.0）")
    p_dn.add_argument("--sweep",      action="store_true",       help="扫描 t=0~1 所有值并对比")
    p_dn.add_argument("--patch",      type=int,   default=16,   help="图像块大小（default:16）")
    p_dn.add_argument("--low-k",      type=int,   default=3,    help="W1 低频保留数（default:3）")
    p_dn.add_argument("--noise",      type=float, default=40.0, help="合成噪声强度（default:40）")
    p_dn.add_argument("--synth-size", type=int,   default=64,   help="合成图像尺寸（default:64）")

    # ── filter ───────────────────────────────────────────────────────
    p_ft = sub.add_parser("filter", help="音频低通滤波（小波DCT基）")
    p_ft.add_argument("--input",  default="",      help="输入 wav 文件（不填=合成信号）")
    p_ft.add_argument("--output", default=None,    help="输出目录")
    p_ft.add_argument("--t",      type=float, default=0.6,  help="形变强度 [0,1]（default:0.6）")
    p_ft.add_argument("--sweep",  action="store_true",       help="扫描 t=0~1 所有值")
    p_ft.add_argument("--frame",  type=int,   default=128,  help="帧大小（default:128）")
    p_ft.add_argument("--low-k",  type=int,   default=10,   help="W1 低频保留数（default:10）")
    p_ft.add_argument("--sr",     type=int,   default=8000, help="合成信号采样率（default:8000）")
    p_ft.add_argument("--noise",  type=float, default=0.5,  help="合成噪声比例（default:0.5）")

    # ── classify ──────────────────────────────────────────────────────
    p_cl = sub.add_parser("classify", help="信号分类路由（LDA判别基）")
    p_cl.add_argument("--input",     default="",   help="输入 .npy 特征文件（不填=合成数据）")
    p_cl.add_argument("--labels",    default="",   help="标签 .npy 文件")
    p_cl.add_argument("--t",         type=float, default=1.0,  help="形变强度 [0,1]（default:1.0）")
    p_cl.add_argument("--sweep",     action="store_true",       help="扫描 t=0~1 所有值")
    p_cl.add_argument("--n-classes", type=int,   default=4,    help="合成类别数（default:4）")
    p_cl.add_argument("--dim",       type=int,   default=32,   help="合成特征维度（default:32）")
    p_cl.add_argument("--compact",   type=int,   default=None, help="投影维度（不填=dim//8）")
    p_cl.add_argument("--noise",     type=float, default=1.5,  help="合成噪声强度（default:1.5）")

    # ── compare ──────────────────────────────────────────────────────
    p_cp = sub.add_parser("compare", help="四种基矩阵定量对比报告")
    p_cp.add_argument("--dim",     type=int, default=128, help="原始维度（default:128）")
    p_cp.add_argument("--compact", type=int, default=32,  help="压缩维度（default:32）")

    args = parser.parse_args()

    SEP = "═" * 56
    print(f"\n{SEP}")
    print(f"  {BOLD}λ-Pipeline{RESET}  v1.8  |  零LLM · 纯numpy · 存储即模型")
    print(SEP)
    t0 = time.perf_counter()

    dispatch = {
        "denoise":  cmd_denoise,
        "filter":   cmd_filter,
        "classify": cmd_classify,
        "compare":  cmd_compare,
    }
    dispatch[args.cmd](args)

    print(f"\n  总耗时: {time.perf_counter()-t0:.2f}s\n")


if __name__ == "__main__":
    main()
