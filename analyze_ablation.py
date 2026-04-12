#!/usr/bin/env python3
"""
analyze_ablation.py — 读取消融训练日志与评估明细，生成论文用汇总表与曲线图。

用法:
    python analyze_ablation.py --log-dir logs/ablations --train-out saved_models/ablations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_metrics_csv(log_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(log_dir.glob("grpo_*_metrics.csv")):
        key = f.stem.replace("_metrics", "")
        try:
            out[key] = pd.read_csv(f)
        except Exception as e:
            print(f"跳过 {f}: {e}")
    return out


def _summarize_training(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    tail = df.tail(min(20, len(df)))
    row = {
        "final_success_mean": float(tail["success_rate"].mean()),
        "final_mean_len": float(tail["mean_response_length"].mean()),
    }
    if "hallucination_rate" in df.columns:
        row["final_halluc_mean"] = float(tail["hallucination_rate"].mean())
    if "mean_adv_sum_abs" in df.columns:
        row["final_mean_adv_sum_abs"] = float(tail["mean_adv_sum_abs"].mean())
    return row


def _eval_jsonl_metrics(eval_dir: Path, metrics_stem: str) -> dict | None:
    """从 evaluate.py 写出的 jsonl 估计幻觉率与平均长度（字符级近似）。
    metrics_stem 如 grpo_b0_G32；evaluate 的 model 名为 grpo_b0_G32_final。
    """
    from env import Arithmetic24Env

    env = Arithmetic24Env()
    model_tag = metrics_stem + "_final"
    halluc_total = 0
    n = 0
    char_lens = []
    correct = 0
    for p in eval_dir.glob(f"eval_{model_tag}_*.jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                resp = rec.get("response", "")
                char_lens.append(len(resp))
                if rec.get("correct"):
                    correct += 1
                nums = rec.get("nums", "")
                diag = env.diagnose_output(nums, resp)
                if diag["hallucination"]:
                    halluc_total += 1
                n += 1
    if n == 0:
        return None
    return {
        "eval_halluc_rate": halluc_total / n,
        "eval_mean_chars": sum(char_lens) / n,
        "eval_acc": correct / n,
    }


def plot_training_curves(metrics: dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    names = sorted(metrics.keys())

    for name in names:
        df = metrics[name]
        if "step" not in df.columns:
            continue
        label = name.replace("grpo_", "")
        axes[0].plot(df["step"], df["success_rate"], label=label, alpha=0.85)
        axes[1].plot(df["step"], df["mean_response_length"], label=label, alpha=0.85)
        if "hallucination_rate" in df.columns:
            axes[2].plot(df["step"], df["hallucination_rate"], label=label, alpha=0.85)

    axes[0].set_title("Success rate (train rollout)")
    axes[0].set_xlabel("update_step")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Mean response length (tokens)")
    axes[1].set_xlabel("update_step")
    axes[2].set_title("Hallucination rate (train)")
    axes[2].set_xlabel("update_step")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=str, default="logs/ablations")
    ap.add_argument("--train-out", type=str, default="saved_models/ablations")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    metrics = _read_metrics_csv(log_dir)
    if not metrics:
        print(f"未在 {log_dir} 找到 grpo_*_metrics.csv")
        return

    rows = []
    for name, df in sorted(metrics.items()):
        summ = _summarize_training(df)
        summ["run"] = name
        em = _eval_jsonl_metrics(log_dir, name)
        if em:
            summ.update(em)
        rows.append(summ)

    summary_csv = log_dir / "ablation_training_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"已保存: {summary_csv}")

    plot_training_curves(metrics, log_dir / "ablation_train_curves.png")

    eval_sum = log_dir / "eval_summary.csv"
    if eval_sum.exists():
        edf = pd.read_csv(eval_sum)
        merged = log_dir / "ablation_with_eval.csv"
        edf.to_csv(merged, index=False)
        print(f"评估汇总: {eval_sum}（已复制为 {merged}）")


if __name__ == "__main__":
    main()
