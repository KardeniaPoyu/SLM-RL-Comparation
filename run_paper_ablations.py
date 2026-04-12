#!/usr/bin/env python3
"""
run_paper_ablations.py — 论文第 5 章消融流水线（B0–B4）+ 评估

默认使用仓库内 .venv_reasoning；也可通过环境变量 PYTHON_EXE 指定 D 盘虚拟环境的 python。

用法:
    set PYTHON_EXE=D:\\path\\to\\venv\\Scripts\\python.exe
    python run_paper_ablations.py --max-steps 200 --group-size 32 --dry-run
    python run_paper_ablations.py --max-steps 50 --skip-eval   # 仅训练
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def default_python_exe() -> str:
    env = os.environ.get("PYTHON_EXE", "").strip()
    if env and Path(env).exists():
        return env
    for candidate in (
        REPO_ROOT / ".venv_reasoning" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_cmd(py: str, args: list[str], cwd: Path) -> bool:
    cmd = [py, *args]
    print("\n" + "=" * 60)
    print("运行:", " ".join(cmd))
    print("=" * 60)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(cwd))
    print(f"结束 code={r.returncode} 用时 {time.time() - t0:.1f}s")
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--sft-path", type=str, default="saved_models/sft_final")
    p.add_argument("--data-file", type=str, default="data/train.csv")
    p.add_argument("--group-size", "-G", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None, help="限制训练题数，快速试跑")
    p.add_argument("--train-out", type=str, default="saved_models/ablations")
    p.add_argument("--log-dir", type=str, default="logs/ablations")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--eval-samples", type=int, default=80)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    py = default_python_exe()
    print(f"Python: {py}")

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    sft_resolved = Path(args.sft_path)
    if not sft_resolved.is_absolute():
        sft_resolved = REPO_ROOT / sft_resolved
    args.sft_path = str(sft_resolved)

    if not args.skip_sft and not sft_resolved.exists():
        out_sft = REPO_ROOT / "saved_models" / "sft_ablation_warmup"
        sft_cmd = [
            "train_sft.py",
            "--model-name", args.model_name,
            "--output-dir", str(out_sft),
            "--epochs", "2",
            "--batch-size", "4",
            "--grad-accum", "8",
        ]
        if args.max_samples:
            sft_cmd += ["--max-samples", str(args.max_samples)]
        if args.dry_run:
            print("[dry-run] 将运行 SFT:", sft_cmd)
        else:
            if not run_cmd(py, sft_cmd, REPO_ROOT):
                sys.exit(1)
        args.sft_path = str(out_sft)
    elif sft_resolved.exists():
        print(f"使用已有 SFT: {args.sft_path}")
    elif args.skip_sft and not sft_resolved.exists():
        print(f"❌ --skip-sft 但 SFT 不存在: {args.sft_path}")
        sys.exit(1)

    ablations = ["B0", "B1", "B2", "B3", "B4"]
    final_dirs: list[str] = []

    for tag in ablations:
        G = args.group_size
        log_tag = f"grpo_{tag.lower()}_G{G}"
        final_path = REPO_ROOT / args.train_out / f"{log_tag}_final"
        final_dirs.append(str(final_path))

        train_args = [
            "train_grpo.py",
            "--ablation", tag,
            "--model-name", args.model_name,
            "--sft-path", args.sft_path,
            "--data-file", args.data_file,
            "--output-dir", args.train_out,
            "--log-dir", args.log_dir,
            "--group-size", str(G),
            "--batch-size", str(args.batch_size),
            "--max-steps", str(args.max_steps),
        ]
        if args.max_samples:
            train_args += ["--max-samples", str(args.max_samples)]

        if args.dry_run:
            print(f"[dry-run] {tag}: ", " ".join(train_args))
            continue

        if args.skip_train:
            continue

        if not run_cmd(py, train_args, REPO_ROOT):
            print(f"❌ 训练失败: {tag}")
            sys.exit(1)

    if args.dry_run:
        print("dry-run 完成。")
        return

    if args.skip_train or args.skip_eval:
        if args.skip_eval:
            print("已跳过评估。")
        return

    # 仅评估本次消融产生的 final 目录（避免扫到历史模型）
    existing = [d for d in final_dirs if Path(d).exists()]
    if not existing:
        print("❌ 未找到任何 *_final 目录，无法评估。")
        sys.exit(1)

    eval_cmd = [
        "evaluate.py",
        "--model-name", args.model_name,
        "--models", *existing,
        "--n-samples", str(args.eval_samples),
        "--output-dir", args.log_dir,
    ]
    if not run_cmd(py, eval_cmd, REPO_ROOT):
        sys.exit(1)

    if not run_cmd(py, ["analyze_ablation.py", "--log-dir", args.log_dir, "--train-out", args.train_out], REPO_ROOT):
        sys.exit(1)

    print("\n✅ 消融 + 评估 + 分析 全部完成。")
    print(f"   指标 CSV: {args.log_dir}/grpo_*_metrics.csv")
    print(f"   汇总图: {args.log_dir}/ablation_*.png")


if __name__ == "__main__":
    main()
