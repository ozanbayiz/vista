"""SAE-based interventions on vision-encoder embeddings.

Implements three intervention modes on SAE feature activations:
  1. Suppression:  zero out target SDFs
  2. Amplification: scale target SDF activations by alpha > 1
  3. Attenuation:  scale target SDF activations by 0 < alpha < 1

The encode-modify-decode loop:
  x -> SAE.encode -> sparse_acts -> modify -> SAE.decode -> x_modified

Usage:
    python -m src.intervene \
        --sae-checkpoint checkpoints/sae_paligemma2.ckpt \
        --hdf5 data/fairface_paligemma2.hdf5 \
        --sdf-dir results/sdf_paligemma2 \
        --output-dir results/intervention_paligemma2 \
        --mode suppression --attributes race
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

from concept_erasure import LeaceEraser

from src.models.sparse_autoencoder import BaseSAE
from src.utils import NumpyEncoder, load_sae_from_checkpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core intervention functions
# ---------------------------------------------------------------------------


def intervene_on_activations(
    sparse_acts: torch.Tensor,
    target_features: list[int] | np.ndarray,
    mode: str = "suppression",
    alpha: float = 0.0,
) -> torch.Tensor:
    """Return a copy of *sparse_acts* with target features modified.

    The input tensor is **not** modified; a clone is made first.

    Args:
        sparse_acts: (batch, hidden_size) or (batch, seq_len, hidden_size).
        target_features: Feature indices to modify.
        mode: One of 'suppression', 'amplification', 'attenuation'.
        alpha: Scaling factor. For suppression, alpha=0. For amplification,
               alpha > 1. For attenuation, 0 < alpha < 1.

    Returns:
        Modified sparse activations (same shape as input).
    """
    modified = sparse_acts.clone()
    target_idx = list(target_features)

    if mode == "suppression":
        modified[..., target_idx] = 0.0
    elif mode == "amplification":
        modified[..., target_idx] = modified[..., target_idx] * alpha
    elif mode == "attenuation":
        modified[..., target_idx] = modified[..., target_idx] * alpha
    elif mode == "passthrough":
        pass  # SAE encode → decode only, no feature modification
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")

    return modified


def encode_modify_decode(
    model: BaseSAE,
    embeddings: torch.Tensor,
    target_features: list[int] | np.ndarray,
    mode: str = "suppression",
    alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full intervention pipeline: encode -> modify -> decode.

    Args:
        model: Trained SAE.
        embeddings: (batch, embed_dim) VE embeddings.
        target_features: SDF indices to intervene on.
        mode: Intervention mode.
        alpha: Scaling factor.

    Returns:
        modified_embeddings: (batch, embed_dim) reconstructed after intervention.
        original_acts: (batch, hidden_size) before modification.
        modified_acts: (batch, hidden_size) after modification.
    """
    with torch.no_grad():
        original_acts = model.encode(embeddings)
        modified_acts = intervene_on_activations(original_acts, target_features, mode, alpha)
        modified_embeddings = model.decode(modified_acts)

    return modified_embeddings, original_acts, modified_acts


# ---------------------------------------------------------------------------
# Batch processing of HDF5 data
# ---------------------------------------------------------------------------


def process_hdf5_with_intervention(
    model: BaseSAE,
    hdf5_path: str,
    split: str,
    target_features: list[int] | np.ndarray,
    mode: str,
    alpha: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Encode VE latents through SAE, apply intervention, return results.

    Returns dict with keys:
        original_acts: (N, D_hidden) mean-pooled original SAE activations
        modified_acts: (N, D_hidden) mean-pooled modified SAE activations
        modified_embeddings: (N, D_embed) mean-pooled modified VE embeddings
        original_embeddings: (N, D_embed) mean-pooled original VE embeddings
    """
    with h5py.File(hdf5_path, "r") as f:
        encoded = np.array(f[split]["encoded"])

    n_samples, seq_len, embed_dim = encoded.shape

    if n_samples == 0:
        d_hidden = model.hidden_size
        return {
            "original_acts": np.empty((0, d_hidden), dtype=np.float32),
            "modified_acts": np.empty((0, d_hidden), dtype=np.float32),
            "modified_embeddings": np.empty((0, embed_dim), dtype=np.float32),
            "original_embeddings": np.empty((0, embed_dim), dtype=np.float32),
        }

    all_orig_acts = []
    all_mod_acts = []
    all_mod_emb = []
    all_orig_emb = []

    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Intervening"):
            end = min(i + batch_size, n_samples)
            batch = encoded[i:end]
            # Flatten all patch tokens for SAE processing
            flat = batch.reshape(-1, embed_dim)
            x = torch.from_numpy(flat).to(device=device, dtype=model_dtype)

            mod_emb, orig_acts, mod_acts = encode_modify_decode(model, x, target_features, mode, alpha)

            bs = end - i
            # Reshape to per-sample and mean-pool across all tokens
            orig_acts_np = orig_acts.cpu().float().numpy().reshape(bs, seq_len, -1).mean(axis=1)
            mod_acts_np = mod_acts.cpu().float().numpy().reshape(bs, seq_len, -1).mean(axis=1)
            mod_emb_np = mod_emb.cpu().float().numpy().reshape(bs, seq_len, embed_dim).mean(axis=1)
            orig_emb_np = flat.reshape(bs, seq_len, embed_dim).mean(axis=1)

            all_orig_acts.append(orig_acts_np)
            all_mod_acts.append(mod_acts_np)
            all_mod_emb.append(mod_emb_np)
            all_orig_emb.append(orig_emb_np)

    return {
        "original_acts": np.concatenate(all_orig_acts),
        "modified_acts": np.concatenate(all_mod_acts),
        "modified_embeddings": np.concatenate(all_mod_emb),
        "original_embeddings": np.concatenate(all_orig_emb),
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def evaluate_suppression(
    original_acts: np.ndarray,
    modified_acts: np.ndarray,
    original_embeddings: np.ndarray,
    modified_embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    target_attribute: str,
) -> dict:
    """Evaluate suppression quality: selectivity and fidelity.

    Returns dict with:
        target_acc_before, target_acc_after: Probe accuracy on targeted attribute.
        non_target_accs: Dict of other attributes' accuracy before/after.
        cosine_similarity: Mean cosine sim between original and modified embeddings.
    """
    n = min(len(original_acts), len(modified_acts))
    orig = original_acts[:n]
    mod = modified_acts[:n]
    orig_emb = original_embeddings[:n]
    mod_emb = modified_embeddings[:n]

    results: dict = {}

    # Cosine similarity between original and modified embeddings
    orig_t = torch.from_numpy(orig_emb).float()
    mod_t = torch.from_numpy(mod_emb).float()
    cos_sim = F.cosine_similarity(orig_t, mod_t, dim=-1).mean().item()
    results["cosine_similarity"] = cos_sim

    # Probe accuracy on each attribute
    for attr_name, attr_labels in labels.items():
        # Ensure labels are aligned with activations (they may be shorter).
        n_attr = min(n, len(attr_labels))
        labs = attr_labels[:n_attr]
        orig_a = orig[:n_attr]
        mod_a = mod[:n_attr]

        # Need at least 2 samples per class for stratified split.
        _, counts = np.unique(labs, return_counts=True)
        if counts.min() < 2:
            log.warning(
                "Attribute '%s' has a class with < 2 samples; skipping.",
                attr_name,
            )
            continue

        # Use a proper train/test split so we don't evaluate on training data.
        indices = np.arange(n_attr)
        idx_train, idx_test = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=labs,
        )

        _lr_kwargs = dict(max_iter=1000, solver="lbfgs", C=0.1)

        log.info("  Fitting probe on original acts for '%s' ...", attr_name)
        clf_orig = LogisticRegression(**_lr_kwargs)
        clf_orig.fit(orig_a[idx_train], labs[idx_train])
        acc_orig = accuracy_score(labs[idx_test], clf_orig.predict(orig_a[idx_test]))
        log.info("    original acc = %.4f", acc_orig)

        log.info("  Fitting probe on modified acts for '%s' ...", attr_name)
        clf_mod = LogisticRegression(**_lr_kwargs)
        clf_mod.fit(mod_a[idx_train], labs[idx_train])
        acc_mod = accuracy_score(labs[idx_test], clf_mod.predict(mod_a[idx_test]))
        log.info("    modified acc = %.4f (delta = %+.4f)", acc_mod, acc_mod - acc_orig)

        entry = {
            "acc_before": float(acc_orig),
            "acc_after": float(acc_mod),
            "delta": float(acc_mod - acc_orig),
        }

        if attr_name == target_attribute:
            results["target"] = entry
        else:
            results.setdefault("non_target", {})[attr_name] = entry

    return results


def evaluate_steering(
    model: BaseSAE,
    hdf5_path: str,
    split: str,
    labels: dict[str, np.ndarray],
    source_class: int,
    target_class: int,
    source_sdfs: np.ndarray,
    target_sdfs: np.ndarray,
    attribute: str,
    alpha_values: list[float],
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate bidirectional steering: attenuate source SDFs, amplify target SDFs.

    Returns dict mapping alpha to steering metrics.
    """
    with h5py.File(hdf5_path, "r") as f:
        encoded = np.array(f[split]["encoded"])

    attr_labels = labels[attribute]
    # Align encoded data and labels (use the shorter length).
    n = min(len(encoded), len(attr_labels))
    encoded = encoded[:n]
    attr_labels = attr_labels[:n]

    source_mask = attr_labels == source_class
    source_indices = np.where(source_mask)[0]

    if len(source_indices) == 0:
        log.warning("No samples for source class %d", source_class)
        return {}

    # Need at least 2 samples per class for stratified split.
    _, counts = np.unique(attr_labels, return_counts=True)
    if counts.min() < 2:
        log.warning("Attribute '%s' has a class with < 2 samples; skipping.", attribute)
        return {}

    n_tokens = encoded.shape[1]
    embed_dim = encoded.shape[2]
    d_hidden = model.hidden_size
    model_dtype = next(model.parameters()).dtype

    # Encode the full dataset once (independent of alpha).
    all_orig_acts = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = encoded[i : i + batch_size]
            flat = batch.reshape(-1, embed_dim)
            x = torch.from_numpy(flat).to(device=device, dtype=model_dtype)
            acts = model.encode(x)
            bs = batch.shape[0]
            per_sample = acts.cpu().float().numpy().reshape(bs, n_tokens, d_hidden).mean(axis=1)
            all_orig_acts.append(per_sample)
    orig_acts_all = np.concatenate(all_orig_acts)

    # Use a proper train/test split so we don't evaluate on training data.
    all_idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=42,
        stratify=attr_labels,
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.1)
    clf.fit(orig_acts_all[idx_train], attr_labels[idx_train])

    # Evaluate only on test-set source samples.
    test_set = set(idx_test.tolist())
    eval_source_indices = np.array([i for i in source_indices if i in test_set])

    if len(eval_source_indices) == 0:
        log.warning("No source-class test samples for class %d", source_class)
        return {}

    orig_preds_source = clf.predict(orig_acts_all[eval_source_indices])
    original_rate = float((orig_preds_source == target_class).mean())

    results: dict = {}

    for alpha in alpha_values:
        all_mod_acts = []
        with torch.no_grad():
            for i in range(0, len(eval_source_indices), batch_size):
                idx_batch = eval_source_indices[i : i + batch_size]
                batch = encoded[idx_batch]
                flat = batch.reshape(-1, embed_dim)
                x = torch.from_numpy(flat).to(device=device, dtype=model_dtype)

                acts = model.encode(x)
                # Attenuate source SDFs
                acts = intervene_on_activations(acts, source_sdfs, "attenuation", alpha=1.0 - alpha)
                # Amplify target SDFs
                acts = intervene_on_activations(acts, target_sdfs, "amplification", alpha=1.0 + alpha)

                bs = len(idx_batch)
                per_sample = acts.cpu().float().numpy().reshape(bs, n_tokens, d_hidden).mean(axis=1)
                all_mod_acts.append(per_sample)

        mod_acts = np.concatenate(all_mod_acts)
        preds = clf.predict(mod_acts)
        steering_rate = float((preds == target_class).mean())

        results[str(alpha)] = {
            "steering_success_rate": steering_rate,
            "original_target_rate": original_rate,
            "delta": steering_rate - original_rate,
        }

    return results


# ---------------------------------------------------------------------------
# S&P Top-K: encoder-centric projection (Barbulau et al., 2509.10809)
# ---------------------------------------------------------------------------


def select_features_by_variance(
    acts: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Select top-k features by inter-class variance (Stylist-like selection).

    For each feature, compute the variance of class-conditional means.
    Features with highest variance are most discriminative for the attribute.

    Returns:
        Array of k feature indices sorted by decreasing variance.
    """
    unique_classes = np.unique(labels)
    class_means = np.stack([acts[labels == c].mean(axis=0) for c in unique_classes])
    variance = class_means.var(axis=0)
    return np.argsort(variance)[-k:][::-1]


def compute_sp_projection(
    model: BaseSAE,
    feature_indices: np.ndarray,
    acts_per_sample: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute the S&P Top-K orthogonal projection matrix.

    Following Barbulau et al. (2509.10809):
      1. Extract encoder weights for selected features.
      2. Train logistic regression on selected features to get importance weights.
      3. Compute control axis: a = E_S^T @ w (weighted sum of encoder weights).
      4. Compute projection: V = I - alpha * a*a^T / ||a||^2.

    Args:
        model: Trained SAE (provides encoder weights W_enc).
        feature_indices: Selected feature indices (e.g., SDFs or variance-ranked).
        acts_per_sample: (N, hidden_size) mean-pooled SAE activations.
        labels: (N,) integer labels for the target attribute.
        alpha: Projection strength in [0, 1]. 1.0 = full projection.

    Returns:
        V: (input_size, input_size) projection matrix.
    """
    # 1. Extract encoder weights for selected features
    W_enc = model.W_enc.detach().cpu().numpy()  # (input_size, hidden_size)
    E_S = W_enc[:, feature_indices]  # (input_size, k)

    # 2. Train logistic regression on selected features
    selected_acts = acts_per_sample[:, feature_indices]  # (N, k)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.1)
    clf.fit(selected_acts, labels)

    # For binary classification, coef_ has shape (1, k).
    # For multiclass, coef_ has shape (n_classes, k).
    # Use mean of absolute coefficients as importance weights for multiclass,
    # or the raw coefficients for binary (sign encodes direction).
    if clf.coef_.shape[0] == 1:
        w = clf.coef_[0]  # (k,) -- binary
    else:
        # For multiclass: the direction that maximally separates classes
        # is the first principal component of the coefficient matrix.
        # Simpler: use mean absolute coefs as feature importance.
        w = np.abs(clf.coef_).mean(axis=0)  # (k,)

    # 3. Compute control axis: a = E_S @ w ∈ R^{input_size}
    a = E_S @ w  # (input_size,)

    # 4. Compute projection: V = I - alpha * a*a^T / ||a||^2
    a_norm_sq = np.dot(a, a)
    if a_norm_sq < 1e-12:
        log.warning("Control axis has near-zero norm; returning identity.")
        return np.eye(W_enc.shape[0])

    d = W_enc.shape[0]  # input_size
    V = np.eye(d) - alpha * np.outer(a, a) / a_norm_sq

    return V


def evaluate_sp_topk(
    model: BaseSAE,
    hdf5_path: str,
    split: str,
    labels: dict[str, np.ndarray],
    target_attribute: str,
    feature_indices: np.ndarray,
    alpha_values: list[float],
    batch_size: int,
    device: torch.device,
    raw_pooled: Optional[np.ndarray] = None,
    acts_per_sample: Optional[np.ndarray] = None,
    original_accs: Optional[dict[str, float]] = None,
    probe_splits: Optional[dict[str, tuple]] = None,
) -> dict:
    """Evaluate S&P Top-K projection for debiasing.

    For each alpha, computes the projection matrix and applies it to mean-pooled
    VE embeddings, then measures probe accuracy on all attributes.

    Original probes (on unmodified embeddings) are fit once and cached across
    alpha values, since they don't depend on the projection.

    Args:
        raw_pooled: Pre-computed mean-pooled VE embeddings (avoids reloading HDF5).
        acts_per_sample: Pre-computed mean-pooled SAE activations.
        original_accs: Pre-computed original probe accuracies (skip re-fitting).
        probe_splits: Pre-computed train/test splits per attribute.

    Returns:
        Dict mapping alpha to metrics (target_acc, non_target_accs, cosine_sim),
        plus "_original_accs" and "_probe_splits" for caching across k values.
    """
    attr_labels = labels[target_attribute]

    # If pre-computed data is not provided, load and compute from HDF5
    if raw_pooled is None or acts_per_sample is None:
        log.info("  Loading and mean-pooling VE embeddings (streaming) ...")
        n_tokens = None
        embed_dim = None
        d_hidden = model.hidden_size
        model_dtype = next(model.parameters()).dtype

        raw_chunks = []
        act_chunks = []

        with h5py.File(hdf5_path, "r") as f:
            ds = f[split]["encoded"]
            n_total = ds.shape[0]
            n = min(n_total, len(attr_labels))
            n_tokens = ds.shape[1]
            embed_dim = ds.shape[2]

            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch = np.array(ds[i:end])  # (bs, seq, dim)
                raw_chunks.append(batch.mean(axis=1))  # (bs, dim)

                flat = batch.reshape(-1, embed_dim)
                x = torch.from_numpy(flat).float().to(device)
                with torch.no_grad():
                    acts = model.encode(x)
                bs = end - i
                per_sample = acts.cpu().float().numpy().reshape(bs, n_tokens, d_hidden).mean(axis=1)
                act_chunks.append(per_sample)

        raw_pooled = np.concatenate(raw_chunks)
        acts_per_sample = np.concatenate(act_chunks)
        log.info(
            "  Loaded %d samples, embed_dim=%d, hidden_size=%d",
            len(raw_pooled),
            raw_pooled.shape[1],
            acts_per_sample.shape[1],
        )

    n = min(len(raw_pooled), len(attr_labels))
    _lr_kwargs = dict(max_iter=1000, solver="lbfgs", C=0.1)

    # --- Fit original (unmodified) probes ONCE, or reuse cache ---
    if original_accs is not None and probe_splits is not None:
        log.info("  Reusing cached original probe results (%d attributes).", len(original_accs))
    else:
        log.info("  Fitting original (baseline) probes ...")
        original_accs = {}
        probe_splits = {}

        for attr_name, attr_labs in labels.items():
            n_attr = min(n, len(attr_labs))
            labs = attr_labs[:n_attr]

            _, counts = np.unique(labs, return_counts=True)
            if counts.min() < 2:
                log.warning("  Attribute '%s' has < 2 samples in some class; skipping.", attr_name)
                continue

            indices = np.arange(n_attr)
            idx_train, idx_test = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                stratify=labs,
            )
            probe_splits[attr_name] = (idx_train, idx_test, labs, n_attr)

            log.info("    Fitting original probe for '%s' ...", attr_name)
            raw_attr = raw_pooled[:n_attr]
            clf = LogisticRegression(**_lr_kwargs)
            clf.fit(raw_attr[idx_train], labs[idx_train])
            acc = accuracy_score(labs[idx_test], clf.predict(raw_attr[idx_test]))
            original_accs[attr_name] = acc
            log.info("    %s original acc = %.4f", attr_name, acc)

    # --- Sweep alpha values ---
    results: dict = {}

    for alpha in alpha_values:
        log.info("  Computing S&P projection (alpha=%.2f) ...", alpha)
        V = compute_sp_projection(
            model,
            feature_indices,
            acts_per_sample[:n],
            attr_labels[:n],
            alpha,
        )

        # Apply projection to raw VE embeddings
        projected = raw_pooled[:n] @ V.T  # (N, embed_dim)

        # Cosine similarity
        orig_t = torch.from_numpy(raw_pooled[:n]).float()
        proj_t = torch.from_numpy(projected).float()
        cos_sim = F.cosine_similarity(orig_t, proj_t, dim=-1).mean().item()
        log.info("    cosine_sim(original, projected) = %.4f", cos_sim)

        alpha_results: dict = {"cosine_similarity": cos_sim}

        for attr_name, (idx_train, idx_test, labs, n_attr) in probe_splits.items():
            proj_attr = projected[:n_attr]
            acc_orig = original_accs[attr_name]

            log.info("    Fitting projected probe for '%s' ...", attr_name)
            clf_proj = LogisticRegression(**_lr_kwargs)
            clf_proj.fit(proj_attr[idx_train], labs[idx_train])
            acc_proj = accuracy_score(labs[idx_test], clf_proj.predict(proj_attr[idx_test]))

            entry = {
                "acc_before": float(acc_orig),
                "acc_after": float(acc_proj),
                "delta": float(acc_proj - acc_orig),
            }
            log.info("    %s: before=%.4f, after=%.4f, delta=%+.4f", attr_name, acc_orig, acc_proj, acc_proj - acc_orig)

            if attr_name == target_attribute:
                alpha_results["target"] = entry
            else:
                alpha_results.setdefault("non_target", {})[attr_name] = entry

        results[str(alpha)] = alpha_results

    # Return cache keys so callers can reuse original probe results across k values
    results["_original_accs"] = original_accs
    results["_probe_splits"] = probe_splits
    return results


# ---------------------------------------------------------------------------
# LEACE (Least-squares Concept Erasure)
# ---------------------------------------------------------------------------


def evaluate_leace(
    hdf5_path: str,
    split: str,
    labels: dict[str, np.ndarray],
    target_attribute: str,
    batch_size: int = 256,
) -> dict:
    """Evaluate LEACE concept erasure on mean-pooled VE embeddings.

    Unlike S&P Top-K, LEACE does not require an SAE — it works directly on the
    raw VE embedding space by computing the optimal linear projection that
    prevents *any* linear classifier from detecting the target concept.

    For comparison, it also evaluates probe accuracy on all non-target attributes
    (e.g., age and race when erasing gender).

    Returns:
        Dict with target and non-target probe accuracies before/after erasure.
    """
    log.info("  Loading and mean-pooling VE embeddings ...")
    with h5py.File(hdf5_path, "r") as f:
        ds = f[split]["encoded"]
        n_total = ds.shape[0]
        n = min(n_total, len(labels[target_attribute]))

        # Stream and mean-pool to avoid loading full tensor into memory
        raw_chunks = []
        for i in tqdm(range(0, n, batch_size), desc="Mean-pooling"):
            end = min(i + batch_size, n)
            batch = np.array(ds[i:end])  # (bs, seq, dim)
            raw_chunks.append(batch.mean(axis=1))
    raw_pooled = np.concatenate(raw_chunks)  # (N, embed_dim)
    del raw_chunks
    log.info("  Loaded %d samples, embed_dim=%d", n, raw_pooled.shape[1])

    # Fit LEACE eraser on the target attribute
    target_labs = labels[target_attribute][:n]
    X_t = torch.from_numpy(raw_pooled).float()
    Z_t = torch.from_numpy(target_labs).long()

    log.info("  Fitting LEACE eraser for '%s' (%d classes) ...", target_attribute, len(np.unique(target_labs)))
    eraser = LeaceEraser.fit(X_t, Z_t)
    X_erased = eraser(X_t).numpy()

    # Cosine similarity between original and erased embeddings
    cos_sim = F.cosine_similarity(X_t, torch.from_numpy(X_erased).float(), dim=-1).mean().item()
    log.info("  cosine_sim(original, erased) = %.4f", cos_sim)

    # Evaluate probes on all attributes.
    # SGDClassifier with log_loss is linear-time in N*D, much faster than
    # LogisticRegression for high-dimensional multiclass (1152 features, 9 classes).
    results: dict = {"cosine_similarity": cos_sim}

    import time as _time

    def _make_probe() -> make_pipeline:
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=1e-3, max_iter=200, tol=1e-3, random_state=42),
        )

    for attr_name, attr_labs in labels.items():
        n_attr = min(n, len(attr_labs))
        labs = attr_labs[:n_attr]

        _, counts = np.unique(labs, return_counts=True)
        if counts.min() < 2:
            log.warning("  Attribute '%s' has < 2 samples; skipping.", attr_name)
            continue

        indices = np.arange(n_attr)
        idx_train, idx_test = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=labs,
        )

        log.info("    Fitting probe on original embeddings for '%s' ...", attr_name)
        t0 = _time.monotonic()
        clf_orig = _make_probe()
        clf_orig.fit(raw_pooled[:n_attr][idx_train], labs[idx_train])
        acc_orig = accuracy_score(labs[idx_test], clf_orig.predict(raw_pooled[:n_attr][idx_test]))
        log.info("      original acc=%.4f (%.1fs)", acc_orig, _time.monotonic() - t0)

        log.info("    Fitting probe on erased embeddings for '%s' ...", attr_name)
        t0 = _time.monotonic()
        clf_erased = _make_probe()
        clf_erased.fit(X_erased[:n_attr][idx_train], labs[idx_train])
        acc_erased = accuracy_score(labs[idx_test], clf_erased.predict(X_erased[:n_attr][idx_test]))
        log.info("      erased  acc=%.4f (%.1fs)", acc_erased, _time.monotonic() - t0)

        entry = {
            "acc_before": float(acc_orig),
            "acc_after": float(acc_erased),
            "delta": float(acc_erased - acc_orig),
        }
        log.info("    %s: before=%.4f, after=%.4f, delta=%+.4f", attr_name, acc_orig, acc_erased, acc_erased - acc_orig)

        if attr_name == target_attribute:
            results["target"] = entry
        else:
            results.setdefault("non_target", {})[attr_name] = entry

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SAE-based embedding interventions")
    parser.add_argument("--sae-checkpoint", default=None, help="SAE checkpoint (not needed for leace)")
    parser.add_argument("--hdf5", required=True, help="FairFace VE latent HDF5")
    parser.add_argument("--sdf-dir", default=None, help="Dir with sdfs_<attr>.npz files")
    parser.add_argument("--output-dir", default="results/intervention")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument(
        "--mode",
        default="suppression",
        choices=["suppression", "steering", "sp_topk", "leace"],
        help="Intervention mode to run",
    )
    parser.add_argument(
        "--sp-k",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Number of features to select for S&P Top-K projection",
    )
    parser.add_argument(
        "--sp-alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 1.0],
        help="Alpha values for S&P Top-K projection strength",
    )
    parser.add_argument(
        "--sp-selection",
        default="variance",
        choices=["variance", "sdf"],
        help="Feature selection method for S&P Top-K",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=["race"],
        help="Attributes to intervene on",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="Number of SDFs to suppress (suppression mode)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0],
        help="Alpha values for steering",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases experiment tracking",
    )
    parser.add_argument(
        "--wandb-project", default="idarve",
        help="W&B project name (default: idarve)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Initialise wandb ---------------------------------------------------
    _wandb_run = None
    if args.wandb:
        import wandb

        wandb_config = {
            "mode": args.mode,
            "attributes": args.attributes,
            "split": args.split,
            "batch_size": args.batch_size,
        }
        if args.mode == "suppression":
            wandb_config["ks"] = args.ks
        elif args.mode == "sp_topk":
            wandb_config.update({"sp_k": args.sp_k, "sp_alphas": args.sp_alphas, "sp_selection": args.sp_selection})
        elif args.mode == "steering":
            wandb_config["alphas"] = args.alphas

        _wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"intervene_{args.mode}_{'_'.join(args.attributes)}",
            config=wandb_config,
            tags=[args.mode] + args.attributes,
            job_type="intervention",
        )
        log.info("W&B run initialised: %s", _wandb_run.url)

    # Load labels (always needed)
    with h5py.File(args.hdf5, "r") as f:
        labels = {key: np.array(f[args.split]["labels"][key]) for key in f[args.split]["labels"].keys()}

    # LEACE mode doesn't need SAE or SDFs
    if args.mode == "leace":
        all_results: dict = {}
        for attr in args.attributes:
            log.info("=== LEACE concept erasure for '%s' ===", attr)
            all_results[attr] = evaluate_leace(
                args.hdf5,
                args.split,
                labels,
                attr,
                args.batch_size,
            )
        results_path = out_dir / "intervention_leace_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        log.info("Results saved to %s", results_path)
        return

    # All other modes require SAE and SDF directory
    if not args.sae_checkpoint:
        parser.error("--sae-checkpoint is required for mode '%s'" % args.mode)
    if not args.sdf_dir:
        parser.error("--sdf-dir is required for mode '%s'" % args.mode)

    sdf_dir = Path(args.sdf_dir)
    log.info("Loading SAE from %s", args.sae_checkpoint)
    model = load_sae_from_checkpoint(args.sae_checkpoint, device)

    all_results: dict = {}

    for attr in args.attributes:
        sdf_path = sdf_dir / f"sdfs_{attr}.npz"
        if not sdf_path.exists():
            log.warning("SDF file not found: %s. Skipping.", sdf_path)
            continue

        sdf_data = np.load(sdf_path)
        # Merge all class SDFs into one flat array for suppression
        all_sdfs = np.unique(np.concatenate([sdf_data[k] for k in sdf_data.files]))

        if args.mode == "suppression":
            attr_results: dict = {}
            for k in args.ks:
                target_features = all_sdfs[:k].tolist() if k <= len(all_sdfs) else all_sdfs.tolist()
                log.info("Suppressing top-%d SDFs for '%s' ...", len(target_features), attr)

                data = process_hdf5_with_intervention(
                    model,
                    args.hdf5,
                    args.split,
                    target_features,
                    "suppression",
                    0.0,
                    args.batch_size,
                    device,
                )

                metrics = evaluate_suppression(
                    data["original_acts"],
                    data["modified_acts"],
                    data["original_embeddings"],
                    data["modified_embeddings"],
                    labels,
                    attr,
                )
                attr_results[f"k={len(target_features)}"] = metrics

            all_results[attr] = attr_results

        elif args.mode == "steering":
            unique_classes = np.unique(labels[attr])
            attr_results = {}
            for src in unique_classes:
                for tgt in unique_classes:
                    if src == tgt:
                        continue
                    src_key = str(src)
                    tgt_key = str(tgt)
                    if src_key not in sdf_data.files or tgt_key not in sdf_data.files:
                        continue

                    log.info("Steering %s -> %s for '%s'", src, tgt, attr)
                    steer_results = evaluate_steering(
                        model,
                        args.hdf5,
                        args.split,
                        labels,
                        int(src),
                        int(tgt),
                        sdf_data[src_key],
                        sdf_data[tgt_key],
                        attr,
                        args.alphas,
                        args.batch_size,
                        device,
                    )
                    attr_results[f"{src}->{tgt}"] = steer_results

            all_results[attr] = attr_results

        elif args.mode == "sp_topk":
            log.info("=== S&P Top-K for '%s' (selection=%s) ===", attr, args.sp_selection)

            # Pre-load data ONCE: stream through HDF5 to get raw_pooled + sae_acts
            log.info("Pre-loading data (streaming through HDF5) ...")
            d_hidden = model.hidden_size
            model_dtype = next(model.parameters()).dtype

            raw_chunks = []
            act_chunks = []
            with h5py.File(args.hdf5, "r") as f:
                ds = f[args.split]["encoded"]
                n_total = ds.shape[0]
                n_s = min(n_total, len(labels[attr]))
                n_tok = ds.shape[1]
                e_dim = ds.shape[2]

                for i in tqdm(range(0, n_s, args.batch_size), desc="Loading+encoding"):
                    end = min(i + args.batch_size, n_s)
                    batch = np.array(ds[i:end])  # (bs, seq, dim)
                    raw_chunks.append(batch.mean(axis=1))
                    flat = batch.reshape(-1, e_dim)
                    x = torch.from_numpy(flat).float().to(device)
                    with torch.no_grad():
                        acts = model.encode(x)
                    bs = end - i
                    per_sample = acts.cpu().float().numpy().reshape(bs, n_tok, d_hidden).mean(axis=1)
                    act_chunks.append(per_sample)

            raw_pooled = np.concatenate(raw_chunks)
            sae_acts = np.concatenate(act_chunks)
            del raw_chunks, act_chunks
            log.info("Loaded %d samples (embed=%d, hidden=%d)", len(raw_pooled), raw_pooled.shape[1], sae_acts.shape[1])

            # For each k, select features and run evaluation across alphas.
            # Original probes are fit once (first k), then cached for subsequent k values.
            attr_results: dict = {}
            cached_orig_accs: Optional[dict[str, float]] = None
            cached_probe_splits: Optional[dict[str, tuple]] = None

            for k in args.sp_k:
                if args.sp_selection == "sdf":
                    features = all_sdfs[:k] if k <= len(all_sdfs) else all_sdfs
                    log.info("Using %d SDF features (of %d available)", len(features), len(all_sdfs))
                else:
                    log.info("Selecting top-%d features by variance for '%s' ...", k, attr)
                    attr_labs = labels[attr][: len(sae_acts)]
                    features = select_features_by_variance(sae_acts, attr_labs, k)

                log.info("Running S&P Top-K evaluation (k=%d, alphas=%s) ...", len(features), args.sp_alphas)
                sp_results = evaluate_sp_topk(
                    model,
                    args.hdf5,
                    args.split,
                    labels,
                    attr,
                    features,
                    args.sp_alphas,
                    args.batch_size,
                    device,
                    raw_pooled=raw_pooled,
                    acts_per_sample=sae_acts,
                    original_accs=cached_orig_accs,
                    probe_splits=cached_probe_splits,
                )

                # Cache original probe results for subsequent k values
                if cached_orig_accs is None:
                    cached_orig_accs = sp_results.pop("_original_accs", None)
                    cached_probe_splits = sp_results.pop("_probe_splits", None)
                else:
                    sp_results.pop("_original_accs", None)
                    sp_results.pop("_probe_splits", None)

                attr_results[f"k={len(features)}"] = sp_results

            del raw_pooled, sae_acts
            all_results[attr] = attr_results

    results_path = out_dir / f"intervention_{args.mode}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    log.info("Results saved to %s", results_path)

    # -- Log to wandb -------------------------------------------------------
    if _wandb_run is not None:
        import wandb

        # Flatten results into wandb-friendly metrics
        for attr, attr_data in all_results.items():
            if isinstance(attr_data, dict):
                # LEACE mode: attr_data has cosine_similarity, target, non_target
                if "cosine_similarity" in attr_data:
                    wandb.log({
                        f"{attr}/cosine_similarity": attr_data["cosine_similarity"],
                        f"{attr}/target_acc_before": attr_data.get("target", {}).get("acc_before", 0),
                        f"{attr}/target_acc_after": attr_data.get("target", {}).get("acc_after", 0),
                        f"{attr}/target_delta": attr_data.get("target", {}).get("delta", 0),
                    })
                else:
                    # Nested structure (suppression by k, sp_topk by k+alpha, etc.)
                    for key, val in attr_data.items():
                        if isinstance(val, dict):
                            flat = {}
                            for k2, v2 in val.items():
                                if isinstance(v2, (int, float)):
                                    flat[f"{attr}/{key}/{k2}"] = v2
                                elif isinstance(v2, dict):
                                    for k3, v3 in v2.items():
                                        if isinstance(v3, (int, float)):
                                            flat[f"{attr}/{key}/{k2}/{k3}"] = v3
                            if flat:
                                wandb.log(flat)

        artifact = wandb.Artifact(
            name=f"intervention_{args.mode}_{'_'.join(args.attributes)}",
            type="results",
        )
        artifact.add_file(str(results_path))
        _wandb_run.log_artifact(artifact)
        _wandb_run.finish()
        log.info("W&B run finished")


if __name__ == "__main__":
    main()
