"""
run_tests.py — 一键回归测试入口
用法：
    python run_tests.py           # 运行全套
    python run_tests.py basic     # 仅运行基础功能
    python run_tests.py morph     # 仅运行形变测试
    python run_tests.py --fast    # 跳过耗时用例（E10）
λ函数单元 + 同伦形变 · 纯信号/特征流水线 · 验证套件
"""
import os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"

SUITES = {
    "basic":     ("test_suite.test_basic_unit",  "基础功能验证"),
    "align":     ("test_suite.test_hf_align",    "封装接口 & 信号有效性"),
    "morph":     ("test_suite.test_t_morph",     "同伦形变特性"),
    "modal":     ("test_suite.test_multimodal",  "多模态信号处理"),
    "exception": ("test_suite.test_exception",   "边界 & 异常稳定性"),
}

def run_suite(key: str, module_path: str, label: str) -> dict:
    import importlib
    print(f"\n{'═'*62}")
    print(f"  {BOLD}▶ {label}{RESET}")
    print(f"{'═'*62}")
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(module_path)
        ok  = mod.run()
    except Exception as e:
        import traceback
        print(f"{RED}  套件加载失败: {e}{RESET}")
        traceback.print_exc()
        ok = False
    elapsed = time.perf_counter() - t0
    return {"key": key, "label": label, "ok": ok, "elapsed": round(elapsed, 2)}


def main():
    parser = argparse.ArgumentParser(description="λ系统全套验证")
    parser.add_argument("suite", nargs="?", default="all",
                        choices=list(SUITES.keys()) + ["all"],
                        help="运行指定套件或 all（默认）")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式（跳过耗时用例）")
    args = parser.parse_args()

    if args.fast:
        os.environ["TEST_FAST"] = "1"
        print(f"{BOLD}  快速模式：跳过耗时用例（E10）{RESET}")

    keys = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    results = []
    total_start = time.perf_counter()

    for key in keys:
        mod_path, label = SUITES[key]
        r = run_suite(key, mod_path, label)
        results.append(r)

    # ── 汇总报告 ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print(f"\n{'═'*62}")
    print(f"  {BOLD}全套验证汇总{RESET}  总耗时: {total_elapsed:.1f}s")
    print(f"{'═'*62}")
    for r in results:
        tag = f"{GREEN}PASS{RESET}" if r["ok"] else f"{RED}FAIL{RESET}"
        print(f"  [{tag}]  {r['label']:24s}  {r['elapsed']:>6.2f}s")

    print(f"{'─'*62}")
    overall = f"{GREEN}{BOLD}全部通过 ✓{RESET}" if not failed else \
              f"{RED}{BOLD}{len(failed)} 个套件失败 ✗{RESET}"
    print(f"  {len(passed)}/{len(results)} 套件通过  →  {overall}")
    print(f"{'═'*62}\n")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
