"""Post-training evaluation for SAE and linear probe analysis.

Run as: python -m src.evaluation --checkpoint <path> --hdf5 <path> [options]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.utils import NumpyEncoder, load_sae_from_checkpoint

log = logging.getLogger(__name__)


def load_latents(hdf5_path: str, mode: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        return np.array(f[mode]["encoded"])


def load_labels(hdf5_path: str, mode: str) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        group = f[mode]["labels"]
        return {key: np.array(group[key]) for key in group.keys()}


def encode_dataset(
    model: torch.nn.Module, latents: np.ndarray, batch_size: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], dict, dict]:
    """Encode *latents* through the SAE and return per-sample results.

    For 3D latents (N, seq_len, embed_dim) the function processes token-level
    data through the SAE but only accumulates **mean-pooled** per-sample
    activations to avoid OOM on large sequence-length datasets.

    Returns:
        acts_per_sample: (N, hidden_size) mean-pooled SAE activations.
        raw_pooled:      (N, embed_dim) mean-pooled input latents (for downstream probe comparison).
        batch_metrics:   Per-batch dicts with l2_loss, l0_norm, cosine_sim.
    """
    is_3d = latents.ndim == 3
    n_samples = latents.shape[0]
    embed_dim = latents.shape[-1]

    all_acts_pooled: list[np.ndarray] = []
    all_raw_pooled: list[np.ndarray] = []
    batch_metrics: list[dict[str, float]] = []

    # Streaming health accumulators
    hidden_size: int | None = None
    active_counts: np.ndarray | None = None  # per-feature activation counts
    total_tokens: int = 0
    l0_per_sample_list: list[np.ndarray] = []

    for i in tqdm(range(0, n_samples, batch_size), desc="Encoding"):
        batch_np = latents[i : i + batch_size]
        bs = batch_np.shape[0]

        if is_3d:
            seq_len = batch_np.shape[1]
            flat = batch_np.reshape(-1, embed_dim)
            x = torch.from_numpy(flat).float().to(device)
        else:
            x = torch.from_numpy(batch_np).float().to(device)

        with torch.no_grad():
            out = model(x)

        acts_np = out["feature_acts"].cpu().numpy()
        recon_np = out["reconstructed"].cpu().numpy()

        # --- Streaming health metrics ---
        if hidden_size is None:
            hidden_size = acts_np.shape[-1]
            active_counts = np.zeros(hidden_size, dtype=np.int64)
        active_counts += (acts_np > 0).sum(axis=0).astype(np.int64)
        total_tokens += acts_np.shape[0]

        # --- Batch-level metrics (l2, l0, cosine sim) ---
        orig_t = torch.from_numpy(flat.astype(np.float32) if is_3d else batch_np.astype(np.float32))
        recon_t = torch.from_numpy(recon_np.astype(np.float32))
        cos_sim = float(F.cosine_similarity(orig_t, recon_t, dim=-1).mean().item())
        mse = float(((orig_t - recon_t) ** 2).mean().item())
        batch_metrics.append({
            "l2_loss": mse,
            "l0_norm": out["l0_norm"].item(),
            "cosine_sim": cos_sim,
            "n_tokens": int(acts_np.shape[0]),
        })

        # --- Per-sample pooled activations ---
        if is_3d:
            acts_per_batch = acts_np.reshape(bs, seq_len, -1).mean(axis=1)
            raw_per_batch = batch_np.astype(np.float32).mean(axis=1)
            # Per-sample L0 (mean across tokens)
            l0_per_tok = (acts_np > 0).sum(axis=-1).astype(np.float32)
            l0_per_sample_list.append(l0_per_tok.reshape(bs, seq_len).mean(axis=1))
        else:
            acts_per_batch = acts_np
            raw_per_batch = batch_np.astype(np.float32)
            l0_per_sample_list.append((acts_np > 0).sum(axis=-1).astype(np.float32))

        all_acts_pooled.append(acts_per_batch)
        all_raw_pooled.append(raw_per_batch)

        del acts_np, recon_np, out  # free token-level memory immediately

    acts_per_sample = np.concatenate(all_acts_pooled)
    raw_pooled = np.concatenate(all_raw_pooled)

    # Attach streaming aggregates to metrics for downstream use
    l0_all = np.concatenate(l0_per_sample_list)
    _health_summary = {
        "num_features": int(hidden_size or 0),
        "num_samples": int(n_samples),
        "total_tokens": int(total_tokens),
        "dead_feature_count": int((active_counts == 0).sum()) if active_counts is not None else 0,
        "dead_feature_ratio": float((active_counts == 0).mean()) if active_counts is not None else 0.0,
        "l0_mean": float(l0_all.mean()),
        "l0_std": float(l0_all.std()),
        "l0_median": float(np.median(l0_all)),
    }
    if active_counts is not None:
        _health_summary["activation_frequency_percentiles"] = {
            str(p): float(np.percentile(active_counts, p)) for p in [0, 25, 50, 75, 90, 99, 100]
        }

    # Aggregate reconstruction quality across batches (token-weighted)
    total_tok = sum(m["n_tokens"] for m in batch_metrics)
    _recon_summary = {
        "mse": sum(m["l2_loss"] * m["n_tokens"] for m in batch_metrics) / max(total_tok, 1),
        "cosine_similarity": sum(m["cosine_sim"] * m["n_tokens"] for m in batch_metrics) / max(total_tok, 1),
    }

    return acts_per_sample, raw_pooled, batch_metrics, _health_summary, _recon_summary


def sae_health_metrics(feature_acts: np.ndarray) -> dict:
    active_per_feature = (feature_acts > 0).sum(axis=0)
    dead_mask = active_per_feature == 0
    l0_per_sample = (feature_acts > 0).sum(axis=1)

    return {
        "num_features": int(feature_acts.shape[1]),
        "num_samples": int(feature_acts.shape[0]),
        "dead_feature_count": int(dead_mask.sum()),
        "dead_feature_ratio": float(dead_mask.mean()),
        "l0_mean": float(l0_per_sample.mean()),
        "l0_std": float(l0_per_sample.std()),
        "l0_median": float(np.median(l0_per_sample)),
        "activation_frequency_percentiles": {
            str(p): float(np.percentile(active_per_feature, p)) for p in [0, 25, 50, 75, 90, 99, 100]
        },
    }


def reconstruction_quality(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    original_f = original.astype(np.float64)
    reconstructed_f = reconstructed.astype(np.float64)

    if original_f.ndim == 3:
        original_f = original_f.reshape(-1, original_f.shape[-1])

    n = min(len(original_f), len(reconstructed_f))
    original_f = original_f[:n]
    reconstructed_f = reconstructed_f[:n]

    mse = float(np.mean((original_f - reconstructed_f) ** 2))

    orig_t = torch.from_numpy(original_f).float()
    recon_t = torch.from_numpy(reconstructed_f).float()
    cosine = float(F.cosine_similarity(orig_t, recon_t, dim=-1).mean().item())

    ss_res = np.sum((original_f - reconstructed_f) ** 2)
    ss_tot = np.sum((original_f - original_f.mean(axis=0)) ** 2)
    r_squared = float(1 - ss_res / (ss_tot + 1e-12))

    return {"mse": mse, "cosine_similarity": cosine, "r_squared": r_squared}


def feature_demographic_alignment(
    feature_acts: np.ndarray, labels: dict[str, np.ndarray]
) -> dict[str, dict]:
    results = {}

    n_samples_feat = feature_acts.shape[0]

    for attr_name, attr_labels in labels.items():
        n_labels = len(attr_labels)
        if n_samples_feat != n_labels:
            log.warning(
                "Feature acts (%d) and labels (%d) length mismatch for '%s', truncating.",
                n_samples_feat,
                n_labels,
                attr_name,
            )
        n = min(n_samples_feat, n_labels)
        acts = feature_acts[:n]
        labs = attr_labels[:n]

        unique_classes = np.unique(labs)
        mean_per_class = {}
        for c in unique_classes:
            mask = labs == c
            mean_per_class[int(c)] = acts[mask].mean(axis=0)

        mean_matrix = np.stack(list(mean_per_class.values()))
        max_mean = mean_matrix.max(axis=0)
        min_mean = mean_matrix.min(axis=0)
        # Use a safe denominator to avoid RuntimeWarning from np.where
        # evaluating both branches eagerly.
        safe_denom = np.maximum(min_mean, 1e-12)
        selectivity = max_mean / safe_denom

        variance_across_classes = mean_matrix.var(axis=0)
        top_k = 20
        top_features = np.argsort(variance_across_classes)[-top_k:][::-1]

        results[attr_name] = {
            "top_variant_features": top_features.tolist(),
            "top_variant_scores": variance_across_classes[top_features].tolist(),
            "selectivity_mean": float(selectivity.mean()),
            "selectivity_max": float(selectivity.max()),
            "mean_per_class_top_features": {
                int(c): mean_matrix[i, top_features].tolist() for i, c in enumerate(unique_classes)
            },
        }

    return results


def ablation_study(
    model: torch.nn.Module,
    latents: np.ndarray,
    labels: dict[str, np.ndarray],
    ks: list[int],
    batch_size: int,
    device: torch.device,
) -> dict:
    """Run ablation study by re-encoding raw latents through the SAE.

    Prefer ``ablation_study_from_acts`` when mean-pooled activations are
    already available, to avoid redundant encoding and potential OOM on
    high-sequence-length data.
    """
    if latents.ndim == 3:
        flat_latents = latents.reshape(-1, latents.shape[-1])
    else:
        flat_latents = latents

    with torch.no_grad():
        all_acts = []
        for i in range(0, len(flat_latents), batch_size):
            x = torch.from_numpy(flat_latents[i : i + batch_size]).float().to(device)
            out = model(x)
            all_acts.append(out["feature_acts"].cpu().numpy())
        full_acts = np.concatenate(all_acts)

    if latents.ndim == 3:
        full_acts_per_sample = full_acts.reshape(len(latents), latents.shape[1], -1).mean(axis=1)
    else:
        full_acts_per_sample = full_acts

    return ablation_study_from_acts(full_acts_per_sample, labels, ks)


def ablation_study_from_acts(
    acts_per_sample: np.ndarray,
    labels: dict[str, np.ndarray],
    ks: list[int],
) -> dict:
    """Run necessity/sufficiency ablation on pre-computed mean-pooled activations."""
    results = {}

    for attr_name, attr_labels in labels.items():
        n = min(len(acts_per_sample), len(attr_labels))
        acts = acts_per_sample[:n]
        labs = attr_labels[:n]

        # Use a proper train/test split so we don't evaluate on training data.
        acts_train, acts_test, labs_train, labs_test = train_test_split(
            acts, labs, test_size=0.2, random_state=42, stratify=labs,
        )

        unique_classes = np.unique(labs_train)
        mean_per_class = np.stack([acts_train[labs_train == c].mean(axis=0) for c in unique_classes])
        variance = mean_per_class.var(axis=0)
        ranked = np.argsort(variance)[::-1]

        # lbfgs is the recommended solver for L2-penalized logistic regression:
        # it converges much faster than saga on moderate-sized datasets with
        # many features and does not require tuning a learning rate.
        _lr_kwargs = dict(max_iter=1000, solver="lbfgs", C=0.1)

        log.info("  [%s] Fitting baseline probe (%d classes) ...", attr_name, len(unique_classes))
        baseline_clf = LogisticRegression(**_lr_kwargs)
        baseline_clf.fit(acts_train, labs_train)
        baseline_acc = accuracy_score(labs_test, baseline_clf.predict(acts_test))
        log.info("  [%s] Baseline accuracy = %.4f", attr_name, baseline_acc)

        attr_results = {"baseline_accuracy": float(baseline_acc), "ablations": {}}

        for k in ks:
            top_k_features = ranked[:k]

            log.info("  [%s] k=%d: fitting without-top-k probe ...", attr_name, k)
            ablated_train = acts_train.copy()
            ablated_train[:, top_k_features] = 0
            ablated_test = acts_test.copy()
            ablated_test[:, top_k_features] = 0
            clf = LogisticRegression(**_lr_kwargs)
            clf.fit(ablated_train, labs_train)
            acc_necessity = accuracy_score(labs_test, clf.predict(ablated_test))

            log.info("  [%s] k=%d: fitting only-top-k probe ...", attr_name, k)
            kept_train = np.zeros_like(acts_train)
            kept_train[:, top_k_features] = acts_train[:, top_k_features]
            kept_test = np.zeros_like(acts_test)
            kept_test[:, top_k_features] = acts_test[:, top_k_features]
            clf2 = LogisticRegression(**_lr_kwargs)
            clf2.fit(kept_train, labs_train)
            acc_sufficiency = accuracy_score(labs_test, clf2.predict(kept_test))

            log.info(
                "  [%s] k=%d: without=%.4f, only=%.4f, drop=%.4f",
                attr_name, k, acc_necessity, acc_sufficiency, baseline_acc - acc_necessity,
            )

            attr_results["ablations"][k] = {
                "acc_without_top_k": float(acc_necessity),
                "acc_only_top_k": float(acc_sufficiency),
                "necessity_drop": float(baseline_acc - acc_necessity),
                "sufficiency_retained": float(acc_sufficiency),
            }

        results[attr_name] = attr_results

    return results


def downstream_probe_comparison(
    raw_latents: np.ndarray,
    sae_acts_per_sample: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict:
    if raw_latents.ndim == 3:
        raw_pooled = raw_latents.mean(axis=1)
    else:
        raw_pooled = raw_latents

    results = {}
    for attr_name, attr_labels in labels.items():
        n = min(len(raw_pooled), len(sae_acts_per_sample), len(attr_labels))
        raw = raw_pooled[:n]
        sae = sae_acts_per_sample[:n]
        labs = attr_labels[:n]

        # Use a proper train/test split so we don't evaluate on training data.
        indices = np.arange(n)
        idx_train, idx_test = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labs,
        )

        _lr_kwargs = dict(max_iter=1000, solver="lbfgs", C=0.1)
        clf_raw = LogisticRegression(**_lr_kwargs)
        clf_raw.fit(raw[idx_train], labs[idx_train])
        acc_raw = accuracy_score(labs[idx_test], clf_raw.predict(raw[idx_test]))

        clf_sae = LogisticRegression(**_lr_kwargs)
        clf_sae.fit(sae[idx_train], labs[idx_train])
        acc_sae = accuracy_score(labs[idx_test], clf_sae.predict(sae[idx_test]))

        report = classification_report(
            labs[idx_test], clf_sae.predict(sae[idx_test]),
            output_dict=True, zero_division=0,
        )

        results[attr_name] = {
            "raw_latent_accuracy": float(acc_raw),
            "sae_latent_accuracy": float(acc_sae),
            "delta": float(acc_sae - acc_raw),
            "per_class_report": report,
        }

    return results


def run_evaluation(
    checkpoint_path: str,
    hdf5_path: str,
    output_dir: str,
    batch_size: int = 256,
    ablation_ks: Optional[list[int]] = None,
    mode: str = "validation",
) -> dict:
    if ablation_ks is None:
        ablation_ks = [10, 20, 50, 100]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading model from %s", checkpoint_path)
    model = load_sae_from_checkpoint(checkpoint_path, device)

    log.info("Loading data from %s (mode=%s)", hdf5_path, mode)
    latents = load_latents(hdf5_path, mode)
    labels = load_labels(hdf5_path, mode)

    log.info("Encoding dataset through SAE (streaming) ...")
    acts_per_sample, raw_pooled, batch_metrics, health, recon_quality = encode_dataset(
        model, latents, batch_size, device,
    )
    del latents  # free the large raw array as early as possible
    log.info(
        "Encoding complete: %d samples, %d features, MSE=%.4f, cosine=%.4f",
        health["num_samples"], health["num_features"],
        recon_quality["mse"], recon_quality["cosine_similarity"],
    )

    log.info("Computing feature-demographic alignment ...")
    alignment = feature_demographic_alignment(acts_per_sample, labels)

    log.info("Running ablation studies (on mean-pooled activations) ...")
    ablation = ablation_study_from_acts(acts_per_sample, labels, ablation_ks)

    log.info("Running downstream probe comparison ...")
    probe_comparison = downstream_probe_comparison(raw_pooled, acts_per_sample, labels)

    results = {
        "sae_health": health,
        "reconstruction_quality": recon_quality,
        "feature_demographic_alignment": alignment,
        "ablation_study": ablation,
        "probe_comparison": probe_comparison,
    }

    results_path = out_path / "evaluation_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log.info("Results saved to %s", results_path)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Post-training SAE evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to SAE Lightning checkpoint")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 file with encoded latents and labels")
    parser.add_argument("--output-dir", default="eval_output", help="Directory for evaluation results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mode", default="validation", choices=["training", "validation"])
    parser.add_argument("--ablation-ks", type=int, nargs="+", default=[10, 20, 50, 100])
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        ablation_ks=args.ablation_ks,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
