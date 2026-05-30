"""
classify_demo.py — LDA 基 λ单元分类路由端到端演示
======================================================================
任务：
  用 LDA 判别基的 λ单元，把多类信号投影到判别空间，
  实现最近邻分类。

  t=0 → PCA 通用特征空间（类别分离度一般）
  t=1 → LDA 判别特征空间（类别分离度最大）
  中间 → 连续过渡，分类准确率随 t 平滑变化

量化指标：
  - 最近邻分类准确率（随 t 增大应提升或稳定）
  - 类间/类内距离比（随 t 增大应提升）
  - t 从 0→1 时准确率变化趋势

零LLM，纯 numpy。

用法：
  python classify_demo.py
  python classify_demo.py --n-classes 6 --dim 64 --samples 200
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from lda_basis import LDABasisBuilder


# ══════════════════════════════════════════════════════════════════════
# 最近邻分类器
# ══════════════════════════════════════════════════════════════════════
def knn_accuracy(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    k: int = 1,
) -> float:
    """纯 numpy k-NN，返回准确率"""
    correct = 0
    for xi, yi in zip(X_test, y_test):
        dists = np.linalg.norm(X_train - xi, axis=1)
        top_k = y_train[np.argsort(dists)[:k]]
        pred  = np.bincount(top_k).argmax()
        if pred == yi:
            correct += 1
    return correct / len(y_test)


def separation_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """类间/类内距离比"""
    classes  = np.unique(y)
    mu_all   = X.mean(axis=0)
    sb, sw   = 0.0, 0.0
    for c in classes:
        Xc   = X[y == c]
        mu_c = Xc.mean(axis=0)
        sb  += len(Xc) * float(np.sum((mu_c - mu_all) ** 2))
        sw  += float(np.sum((Xc - mu_c) ** 2))
    return sb / (sw + 1e-8)


# ══════════════════════════════════════════════════════════════════════
# 主演示逻辑
# ══════════════════════════════════════════════════════════════════════
def run_classify_demo(
    origin_dim:  int   = 128,
    compact_dim: int   = 32,
    n_classes:   int   = 4,
    n_train:     int   = 80,
    n_test:      int   = 20,
    signal_scale:float = 2.5,
    noise_scale: float = 0.6,
    seed:        int   = 42,
    t_values:    list  = None,
) -> list[dict]:

    t_values = t_values or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng      = np.random.default_rng(seed)

    # ── 生成多类合成数据 ──────────────────────────────────────────
    centers = rng.standard_normal((n_classes, origin_dim)).astype(np.float32) * signal_scale
    X_train, y_train, X_test, y_test = [], [], [], []
    for c, mu in enumerate(centers):
        Xi_tr = rng.standard_normal((n_train, origin_dim)).astype(np.float32) * noise_scale + mu
        Xi_te = rng.standard_normal((n_test,  origin_dim)).astype(np.float32) * noise_scale + mu
        X_train.append(Xi_tr);  y_train.append(np.full(n_train, c, dtype=np.int32))
        X_test.append(Xi_te);   y_test.append(np.full(n_test,   c, dtype=np.int32))

    X_train = np.vstack(X_train);  y_train = np.hstack(y_train)
    X_test  = np.vstack(X_test);   y_test  = np.hstack(y_test)

    # ── 构造 LDA 基（从训练集）──────────────────────────────────
    builder = LDABasisBuilder(origin_dim, compact_dim, seed=seed)
    builder.fit(X_train, y_train)
    W0_pca, W1_lda = builder.build_weights()

    # 原始空间基准准确率
    acc_raw = knn_accuracy(X_train, y_train, X_test, y_test)
    sep_raw = separation_ratio(X_train, y_train)

    print(f"\n  原始空间（{origin_dim}维）  1-NN准确率={acc_raw:.4f}  分离比={sep_raw:.4f}")
    print(f"\n  {'t':>5}  {'准确率':>8}  {'分离比':>10}  {'投影维度':>8}  vs 原始空间")
    print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*8}  ──────────")

    results = []
    for t in t_values:
        W0f  = W0_pca.astype(np.float32)
        W1f  = W1_lda.astype(np.float32)
        W_t  = (1 - t) * W0f + t * W1f

        Xtr_proj = X_train @ W_t
        Xte_proj = X_test  @ W_t

        acc = knn_accuracy(Xtr_proj, y_train, Xte_proj, y_test)
        sep = separation_ratio(Xtr_proj, y_train)

        delta_acc = acc - acc_raw
        if delta_acc > 0.02:
            effect = "↑优于原始"
        elif delta_acc > -0.02:
            effect = "─与原始持平"
        else:
            effect = "↓低于原始"

        results.append({"t": t, "acc": round(acc, 4), "sep": round(sep, 4)})
        print(f"  {t:>5.2f}  {acc:>8.4f}  {sep:>10.4f}  {compact_dim:>8}D  "
              f"{delta_acc:+.4f} {effect}")

    return results, acc_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",       type=int,   default=128)
    parser.add_argument("--compact",   type=int,   default=32)
    parser.add_argument("--n-classes", type=int,   default=4)
    parser.add_argument("--samples",   type=int,   default=80)
    parser.add_argument("--noise",     type=float, default=0.6)
    args = parser.parse_args()

    SEP = "═" * 56
    print(f"\n{SEP}")
    print("  λ单元分类路由演示（LDA判别基，零LLM）")
    print(f"  {args.n_classes}类  特征维度 {args.dim}→{args.compact}  噪声={args.noise}")
    print(SEP)

    results, acc_raw = run_classify_demo(
        origin_dim  = args.dim,
        compact_dim = args.compact,
        n_classes   = args.n_classes,
        n_train     = args.samples,
        n_test      = max(20, args.samples // 4),
        noise_scale = args.noise,
    )

    # 结论
    best      = max(results, key=lambda r: r["acc"])
    worst     = min(results, key=lambda r: r["acc"])
    acc_range = best["acc"] - worst["acc"]
    sep_range = max(r["sep"] for r in results) - min(r["sep"] for r in results)
    lda_gain  = results[-1]["acc"] - results[0]["acc"]

    print(f"\n  最佳 t={best['t']}  准确率={best['acc']:.4f}")
    print(f"  t=0→1 准确率变化: {results[0]['acc']:.4f} → {results[-1]['acc']:.4f}  "
          f"（Δ={lda_gain:+.4f}）")
    print(f"  t=0→1 分离比变化: {results[0]['sep']:.4f} → {results[-1]['sep']:.4f}")

    if lda_gain >= 0:
        print(f"\n  结论: ✓ LDA 方向(t=1)分类性能不低于 PCA(t=0)")
    else:
        print(f"\n  结论: △ 合成数据结构简单，PCA 已够用（LDA增益={lda_gain:+.4f}）")

    print(f"\n{SEP}\n")
    return results


if __name__ == "__main__":
    main()
