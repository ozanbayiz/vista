"""Three-stage SDF (Sparse Dictionary Feature) filtering pipeline.

Identifies demographic-correlated features in SAE activations via:
  Stage 1: Intra-group activation frequency  (top k1 per class)
  Stage 2: Intra-group mean activation       (top k2 subset)
  Stage 3: Inter-group label entropy          (top k3 with lowest entropy)

Usage:
    python -m src.analysis \
        --sae-checkpoint checkpoints/sae_paligemma2.ckpt \
        --hdf5 data/fairface_paligemma2.hdf5 \
        --output-dir results/sdf_paligemma2 \
        --attributes race gender age
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
from tqdm.auto import tqdm

from src.utils import NumpyEncoder, load_sae_from_checkpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAE encoding of FairFace latents
# ---------------------------------------------------------------------------


def encode_fairface_through_sae(
    model: torch.nn.Module,
    hdf5_path: str,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode FairFace VE latents through the SAE.

    Returns:
        acts_per_sample: (N, hidden_size) mean-pooled SAE activations per image.
        labels: dict mapping attribute name to (N,) label array.
    """
    with h5py.File(hdf5_path, "r") as f:
        encoded = np.array(f[mode]["encoded"])
        labels = {key: np.array(f[mode]["labels"][key]) for key in f[mode]["labels"].keys()}

    n_samples = encoded.shape[0]
    seq_len = encoded.shape[1]
    embed_dim = encoded.shape[2]
    d_hidden = model.hidden_size if hasattr(model, "hidden_size") else model.W_enc.shape[1]

    all_acts = []
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="SAE encoding"):
            batch = encoded[i : i + batch_size]
            # Flatten all patch tokens into one big batch for SAE encoding
            flat = batch.reshape(-1, embed_dim)
            x = torch.from_numpy(flat).float().to(device)
            acts = model.encode(x).cpu().numpy()
            # Reshape back to per-sample and mean-pool across tokens
            bs = batch.shape[0]
            per_sample = acts.reshape(bs, seq_len, d_hidden).mean(axis=1)
            all_acts.append(per_sample)

    acts_per_sample = np.concatenate(all_acts, axis=0)
    return acts_per_sample, labels


# ---------------------------------------------------------------------------
# Stage 1: Intra-group activation frequency
# ---------------------------------------------------------------------------


def stage1_activation_frequency(
    acts: np.ndarray,
    labels: np.ndarray,
    k1: int = 200,
) -> dict[int, np.ndarray]:
    """For each class, find the top-k1 features by activation frequency.

    Activation frequency = fraction of samples in that class where the feature
    fires (activation > 0).

    Returns:
        Dict mapping class_id to array of feature indices (shape (k1,)).
    """
    unique_classes = np.unique(labels)
    result: dict[int, np.ndarray] = {}

    for c in unique_classes:
        mask = labels == c
        class_acts = acts[mask]
        freq = (class_acts > 0).mean(axis=0)
        top_k = np.argsort(freq)[-k1:][::-1]
        result[int(c)] = top_k

    return result


# ---------------------------------------------------------------------------
# Stage 2: Intra-group mean activation
# ---------------------------------------------------------------------------


def stage2_mean_activation(
    acts: np.ndarray,
    labels: np.ndarray,
    stage1_features: dict[int, np.ndarray],
    k2: int = 100,
) -> dict[int, np.ndarray]:
    """From Stage 1 candidates, keep top-k2 by mean activation within each class.

    Returns:
        Dict mapping class_id to array of feature indices (shape (k2,)).
    """
    unique_classes = np.unique(labels)
    result: dict[int, np.ndarray] = {}

    for c in unique_classes:
        mask = labels == c
        class_acts = acts[mask]
        candidates = stage1_features[int(c)]
        mean_act = class_acts[:, candidates].mean(axis=0)
        top_k = np.argsort(mean_act)[-k2:][::-1]
        result[int(c)] = candidates[top_k]

    return result


# ---------------------------------------------------------------------------
# Stage 3: Inter-group label entropy
# ---------------------------------------------------------------------------


def compute_top_k_activations(
    acts: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    k_top: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """For each feature, get the top-k_top activating samples' values and labels.

    Returns:
        top_vals:   (n_features, k_top)
        top_labels: (n_features, k_top)
    """
    n_features = len(feature_indices)
    top_vals = np.zeros((n_features, k_top))
    top_labels = np.zeros((n_features, k_top), dtype=np.int64)

    for i, fi in enumerate(feature_indices):
        col = acts[:, fi]
        actual_k = min(k_top, len(col))
        idx = np.argpartition(-col, actual_k)[:actual_k]
        idx_sorted = idx[np.argsort(-col[idx])]
        top_vals[i, :actual_k] = col[idx_sorted]
        top_labels[i, :actual_k] = labels[idx_sorted]

    return top_vals, top_labels


def compute_entropy(
    top_vals: np.ndarray,
    top_labels: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Compute activation-weighted label entropy for each feature.

    For each feature, the probability of each label is proportional to the
    summed activation of that label's top-k samples (not uniform counts).

    Returns:
        entropy: (n_features,) array.  Lower entropy = more label-specific.
    """
    n_features = top_vals.shape[0]
    entropy = np.full(n_features, np.inf)

    for i in range(n_features):
        unique_labels = np.unique(top_labels[i])
        if len(unique_labels) == 0:
            continue
        summed = np.array(
            [top_vals[i][top_labels[i] == lab].sum() for lab in unique_labels]
        )
        total = summed.sum() + eps
        probs = summed / total
        entropy[i] = -np.sum(probs * np.log(probs + eps))

    return entropy


def stage3_entropy_filter(
    acts: np.ndarray,
    labels: np.ndarray,
    stage2_features: dict[int, np.ndarray],
    k3: int = 50,
    k_top_per_feature: int = 20,
) -> dict[int, np.ndarray]:
    """From Stage 2 candidates, keep k3 features with the lowest label entropy.

    Low entropy means the feature's top activations are dominated by a single
    demographic class, i.e. the feature is class-specific.

    Returns:
        Dict mapping class_id to array of feature indices (shape (<=k3,)).
    """
    unique_classes = np.unique(labels)
    result: dict[int, np.ndarray] = {}

    for c in unique_classes:
        candidates = stage2_features[int(c)]
        top_vals, top_labels = compute_top_k_activations(
            acts, labels, candidates, k_top=k_top_per_feature
        )
        entropy = compute_entropy(top_vals, top_labels)
        # Keep k3 features with lowest entropy
        actual_k3 = min(k3, len(candidates))
        low_ent_idx = np.argsort(entropy)[:actual_k3]
        result[int(c)] = candidates[low_ent_idx]

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_sdf_pipeline(
    acts: np.ndarray,
    labels: np.ndarray,
    k1: int = 200,
    k2: int = 100,
    k3: int = 50,
    k_top_per_feature: int = 20,
) -> dict[int, np.ndarray]:
    """Run the full 3-stage SDF filtering pipeline for a single attribute.

    Args:
        acts: (N, D) mean-pooled SAE activations.
        labels: (N,) integer labels for the target attribute.
        k1, k2, k3: Cutoffs for stages 1, 2, 3.
        k_top_per_feature: Number of top-activating samples for entropy.

    Returns:
        Dict mapping class_id to array of SDF feature indices.
    """
    log.info("Stage 1: activation frequency (k1=%d) ...", k1)
    s1 = stage1_activation_frequency(acts, labels, k1)
    for c, feats in s1.items():
        log.info("  Class %d: %d candidate features", c, len(feats))

    log.info("Stage 2: mean activation (k2=%d) ...", k2)
    s2 = stage2_mean_activation(acts, labels, s1, k2)
    for c, feats in s2.items():
        log.info("  Class %d: %d candidate features", c, len(feats))

    log.info("Stage 3: entropy filter (k3=%d, k_top=%d) ...", k3, k_top_per_feature)
    s3 = stage3_entropy_filter(acts, labels, s2, k3, k_top_per_feature)
    for c, feats in s3.items():
        log.info("  Class %d: %d SDFs", c, len(feats))

    return s3


def compute_alignment_rate(
    acts: np.ndarray,
    labels: np.ndarray,
    sdfs: dict[int, np.ndarray],
    k_top: int = 20,
) -> dict[int, dict]:
    """For each class's SDFs, measure what fraction of top-activating samples
    belong to that class (alignment rate).

    Returns:
        Dict[class_id, {"feature_idx": [...], "alignment_rates": [...], "mean_rate": float}]
    """
    result: dict[int, dict] = {}

    for c, feature_indices in sdfs.items():
        rates = []
        for fi in feature_indices:
            col = acts[:, fi]
            actual_k = min(k_top, len(col))
            top_idx = np.argpartition(-col, actual_k)[:actual_k]
            top_labels = labels[top_idx]
            rate = float((top_labels == c).mean())
            rates.append(rate)
        result[int(c)] = {
            "feature_indices": feature_indices.tolist(),
            "alignment_rates": rates,
            "mean_alignment_rate": float(np.mean(rates)) if rates else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SDF filtering pipeline")
    parser.add_argument("--sae-checkpoint", required=True, help="SAE checkpoint path")
    parser.add_argument("--hdf5", required=True, help="FairFace VE latent HDF5 path")
    parser.add_argument("--output-dir", default="results/sdf", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mode", default="validation", choices=["training", "validation"])
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=["race"],
        help="Attributes to analyze (e.g., race gender age)",
    )
    parser.add_argument("--k1", type=int, default=200, help="Stage 1 cutoff")
    parser.add_argument("--k2", type=int, default=100, help="Stage 2 cutoff")
    parser.add_argument("--k3", type=int, default=50, help="Stage 3 cutoff")
    parser.add_argument("--k-top", type=int, default=20, help="Top-k for entropy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading SAE from %s", args.sae_checkpoint)
    model = load_sae_from_checkpoint(args.sae_checkpoint, device)

    log.info("Encoding FairFace latents through SAE ...")
    acts, labels = encode_fairface_through_sae(
        model, args.hdf5, args.mode, args.batch_size, device
    )
    log.info("Activations shape: %s", acts.shape)

    all_results: dict[str, dict] = {}

    for attr in args.attributes:
        if attr not in labels:
            log.warning("Attribute '%s' not found in labels. Skipping.", attr)
            continue

        log.info("=== Running SDF pipeline for '%s' ===", attr)
        sdfs = run_sdf_pipeline(
            acts, labels[attr], args.k1, args.k2, args.k3, args.k_top
        )
        alignment = compute_alignment_rate(acts, labels[attr], sdfs, args.k_top)

        all_results[attr] = {
            "sdfs": {str(c): feats.tolist() for c, feats in sdfs.items()},
            "alignment": alignment,
            "params": {"k1": args.k1, "k2": args.k2, "k3": args.k3, "k_top": args.k_top},
        }

        # Save per-attribute numpy arrays for downstream use
        np.savez(
            out_dir / f"sdfs_{attr}.npz",
            **{str(c): feats for c, feats in sdfs.items()},
        )

    results_path = out_dir / "sdf_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    log.info("Results saved to %s", results_path)

    # Also save the full activations for downstream experiments
    np.save(out_dir / "sae_acts.npy", acts)
    log.info("SAE activations saved to %s", out_dir / "sae_acts.npy")


if __name__ == "__main__":
    main()
