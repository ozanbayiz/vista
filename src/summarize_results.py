"""Summarize all experimental results for PaliGemma2 (or any VE).

Reads SDF results, evaluation results, and training metrics, then prints
a structured report to stdout.

Usage:
    python -m src.summarize_results \
        --sdf-dir /scratch/current/ozanbayiz/results/sdf_paligemma2 \
        --eval-dir /scratch/current/ozanbayiz/results/eval_paligemma2 \
        --training-dir /scratch/current/ozanbayiz/outputs/lightning_logs/version_3 \
        --probe-dir /scratch/current/ozanbayiz/outputs/lightning_logs/version_2 \
        --ve-name PaliGemma2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---- Label maps ----
RACE_MAP = {
    0: "White", 1: "Black", 2: "Latino/Hispanic",
    3: "East Asian", 4: "Southeast Asian", 5: "Indian", 6: "Middle Eastern",
}
GENDER_MAP = {0: "Male", 1: "Female"}
AGE_MAP = {
    0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
    5: "40-49", 6: "50-59", 7: "60-69", 8: "70+",
}
ATTR_MAPS = {"race": RACE_MAP, "gender": GENDER_MAP, "age": AGE_MAP}


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# SDF Results
# ---------------------------------------------------------------------------


def summarize_sdf(sdf_dir: Path) -> None:
    results_path = sdf_dir / "sdf_results.json"
    if not results_path.exists():
        print("  [SDF results not found]")
        return

    with open(results_path) as f:
        results = json.load(f)

    for attr in ["race", "gender", "age"]:
        if attr not in results:
            continue
        name_map = ATTR_MAPS.get(attr, {})
        r = results[attr]
        sdfs = r["sdfs"]
        alignment = r["alignment"]

        subsection(f"{attr.upper()} SDF Analysis")
        print(f"  Pipeline params: k1={r['params']['k1']}, k2={r['params']['k2']}, "
              f"k3={r['params']['k3']}, k_top={r['params']['k_top']}")

        total_unique = len(set(f for cls_feats in sdfs.values() for f in cls_feats))
        print(f"  Total unique SDFs across all classes: {total_unique}")
        print()
        print(f"  {'Class':<22s} {'#SDFs':>5s}  {'Mean Align':>10s}  "
              f"{'Max':>5s}  {'>0.5':>5s}  {'>0.3':>5s}")
        print(f"  {'-' * 62}")

        for cls_id in sorted(sdfs.keys(), key=int):
            name = name_map.get(int(cls_id), f"Class {cls_id}")
            a = alignment[cls_id]
            rates = a["alignment_rates"]
            high = sum(1 for x in rates if x > 0.5)
            mid = sum(1 for x in rates if x > 0.3)
            print(f"  {name:<22s} {len(sdfs[cls_id]):>5d}  "
                  f"{a['mean_alignment_rate']:>10.3f}  "
                  f"{max(rates):>5.3f}  {high:>5d}  {mid:>5d}")

    # Cross-attribute overlap
    if len(results) > 1:
        subsection("Cross-Attribute SDF Overlap")
        attrs = list(results.keys())
        for i, a1 in enumerate(attrs):
            s1 = set(f for cls_feats in results[a1]["sdfs"].values() for f in cls_feats)
            for a2 in attrs[i + 1:]:
                s2 = set(f for cls_feats in results[a2]["sdfs"].values() for f in cls_feats)
                overlap = s1 & s2
                print(f"  {a1} & {a2}: {len(overlap)} shared features "
                      f"(|{a1}|={len(s1)}, |{a2}|={len(s2)})")


# ---------------------------------------------------------------------------
# Evaluation Results
# ---------------------------------------------------------------------------


def summarize_eval(eval_dir: Path) -> None:
    results_path = eval_dir / "evaluation_results.json"
    if not results_path.exists():
        print("  [Evaluation results not found]")
        return

    with open(results_path) as f:
        results = json.load(f)

    # SAE Health
    if "sae_health" in results:
        subsection("SAE Health Metrics")
        h = results["sae_health"]
        for k, v in h.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Reconstruction Quality
    if "reconstruction_quality" in results:
        subsection("Reconstruction Quality")
        rq = results["reconstruction_quality"]
        for k, v in rq.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Ablation Study
    if "ablation_study" in results:
        subsection("Ablation Study (Necessity / Sufficiency)")
        ab = results["ablation_study"]
        for attr_name, attr_data in ab.items():
            print(f"\n  Attribute: {attr_name}")
            print(f"    Baseline accuracy: {attr_data['baseline_accuracy']:.4f}")
            print(f"    {'k':>6s}  {'Without top-k':>14s}  {'Only top-k':>11s}  "
                  f"{'Necessity drop':>15s}")
            print(f"    {'-' * 54}")
            for k_str, v in sorted(attr_data["ablations"].items(), key=lambda x: int(x[0])):
                print(f"    {k_str:>6s}  {v['acc_without_top_k']:>14.4f}  "
                      f"{v['acc_only_top_k']:>11.4f}  {v['necessity_drop']:>15.4f}")

    # Probe Comparison
    if "probe_comparison" in results:
        subsection("Downstream Probe Comparison (Raw VE vs SAE Latents)")
        pc = results["probe_comparison"]
        for attr_name, attr_data in pc.items():
            print(f"\n  Attribute: {attr_name}")
            for repr_name, repr_data in attr_data.items():
                print(f"    {repr_name}: acc={repr_data['accuracy']:.4f}, "
                      f"macro_f1={repr_data['macro_f1']:.4f}")


# ---------------------------------------------------------------------------
# Training Metrics (SAE)
# ---------------------------------------------------------------------------


def summarize_training(training_dir: Path, label: str = "SAE") -> None:
    metrics_path = training_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"  [{label} training metrics not found]")
        return

    # Read CSV and find val rows
    val_rows = []
    train_tail = []
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("val/loss") and row["val/loss"]:
                val_rows.append(row)
            train_tail.append(row)

    train_tail = train_tail[-5:]
    n_rows = sum(1 for _ in open(metrics_path)) - 1

    print(f"  Total logged steps: {n_rows}")

    if train_tail:
        last = train_tail[-1]
        print(f"  Latest epoch: {last.get('epoch', '?')}, step: {last.get('step', '?')}")
        for k, v in last.items():
            if v and k.startswith("train/"):
                print(f"    {k}: {float(v):.6f}")

    if val_rows:
        subsection(f"{label} Validation Metrics (per epoch)")
        for row in val_rows:
            parts = [f"epoch={row['epoch']}, step={row['step']}"]
            for k, v in row.items():
                if v and k.startswith("val/"):
                    parts.append(f"{k}={float(v):.4f}")
            print(f"  {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IDARVE results")
    parser.add_argument("--sdf-dir", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--probe-dir", type=Path, default=None)
    parser.add_argument("--ve-name", default="PaliGemma2")
    args = parser.parse_args()

    print(f"{'#' * 72}")
    print(f"  IDARVE Results Summary -- {args.ve_name}")
    print(f"{'#' * 72}")

    if args.training_dir:
        section(f"{args.ve_name} SAE Training")
        summarize_training(args.training_dir, label="SAE")

    if args.probe_dir:
        section(f"{args.ve_name} Linear Probe Training")
        summarize_training(args.probe_dir, label="Probe")

    if args.sdf_dir:
        section(f"{args.ve_name} SDF Analysis")
        summarize_sdf(args.sdf_dir)

    if args.eval_dir:
        section(f"{args.ve_name} Evaluation")
        summarize_eval(args.eval_dir)

    print(f"\n{'=' * 72}")
    print("  End of Report")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
