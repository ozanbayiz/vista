"""Cross-architecture demographic feature universality analysis.

Measures whether SAE-identified demographic features are *universal* across
vision encoder architectures or architecture-specific.  Three complementary
similarity measures are provided:

1. **Linear CKA** (Kornblith et al., 2019) between mean-pooled SAE
   activation matrices on the same images.
2. **SDF rank-order overlap** via Spearman correlation on inter-class
   variance rankings (feature indices are incomparable across SAEs with
   different hidden sizes).
3. **Demographic subspace similarity** via principal angles (Golub & Van
   Loan, 2013) between the spans of top-k SDF encoder weight vectors.

Usage:
    python -m src.cross_encoder \\
        --sae-checkpoints ckpt_dino.ckpt ckpt_siglip.ckpt ckpt_pali.ckpt \\
        --hdf5-files data/ff_dino.hdf5 data/ff_siglip.hdf5 data/ff_pali.hdf5 \\
        --sdf-dirs sdf_dino/ sdf_siglip/ sdf_pali/ \\
        --encoder-names dinov3 siglip2 paligemma2 \\
        --output-dir results/cross_encoder \\
        --attribute race
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

from src.models.sparse_autoencoder import BaseSAE
from src.utils import NumpyEncoder, build_metadata, load_sae_from_checkpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., 2012).

    Args:
        K: ``(n, n)`` kernel matrix (not modified).
        L: ``(n, n)`` kernel matrix (not modified).

    Returns:
        Unbiased HSIC estimate (scalar).
    """
    n = K.shape[0]
    if n < 4:
        return 0.0

    # Defensive copies: the unbiased estimator requires zero diagonals,
    # but we must not mutate the caller's matrices (they may be reused
    # in subsequent calls, e.g. _hsic(K, K) after _hsic(K, L)).
    K = K.copy()
    L = L.copy()

    # Zero the diagonals (unbiased estimator requirement).
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    ones = np.ones(n)
    # Eq. 3 from Song et al., 2012
    term1 = np.trace(K @ L)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 * (ones @ K @ L @ ones) / (n - 2)

    return float((term1 + term2 - term3) / (n * (n - 3)))


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA similarity between two activation matrices.

    Implements the unbiased CKA estimator from Kornblith et al. (2019)
    using the linear kernel (dot product).

    Args:
        X: ``(N, D1)`` activation matrix from encoder 1.
        Y: ``(N, D2)`` activation matrix from encoder 2.
            Both must have the same number of samples *N*.

    Returns:
        CKA similarity in ``[0, 1]`` (higher = more similar).

    Raises:
        ValueError: If *X* and *Y* have different numbers of rows.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} rows, Y has {Y.shape[0]} rows",
        )

    # Centre features (not strictly required for linear CKA, but improves
    # numerical stability with the unbiased HSIC estimator).
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    K = X @ X.T  # (N, N) linear kernel
    L = Y @ Y.T

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        log.warning("Degenerate CKA denominator (%.2e); returning 0.", denom)
        return 0.0
    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SDF rank-order overlap (Spearman on inter-class variance)
# ---------------------------------------------------------------------------


def _inter_class_variance(
    acts: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-feature variance of class-conditional means.

    Args:
        acts: ``(N, D)`` activations.
        labels: ``(N,)`` integer class labels.

    Returns:
        ``(D,)`` variance array.
    """
    unique = np.unique(labels)
    class_means = np.stack([acts[labels == c].mean(axis=0) for c in unique])
    return class_means.var(axis=0)


def compute_sdf_overlap(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    labels: np.ndarray,
) -> float:
    """KS-based similarity between inter-class variance distributions.

    Because feature indices are not directly comparable across SAEs with
    different hidden sizes, this function compares the *distribution shape*
    of demographic discriminability across features.

    Each SAE's per-feature inter-class variance is sorted descending and
    normalised to form a cumulative proportion curve (analogous to a Lorenz
    curve).  The Kolmogorov-Smirnov statistic measures the maximum
    divergence between the two curves; we return ``1 - KS`` so that 1.0
    means identical distribution shapes and 0.0 means maximally different.

    Args:
        acts_a: ``(N, D_a)`` SAE activations from encoder A.
        acts_b: ``(N, D_b)`` SAE activations from encoder B.
        labels: ``(N,)`` integer demographic labels.

    Returns:
        Similarity score in ``[0, 1]``.
    """
    var_a = np.sort(_inter_class_variance(acts_a, labels))[::-1]
    var_b = np.sort(_inter_class_variance(acts_b, labels))[::-1]

    # Normalise to cumulative proportion (CDF of variance mass).
    total_a = var_a.sum()
    total_b = var_b.sum()
    if total_a == 0 or total_b == 0:
        return 0.0

    cdf_a = np.cumsum(var_a) / total_a
    cdf_b = np.cumsum(var_b) / total_b

    # Interpolate both CDFs onto a common grid of [0, 1] quantiles.
    grid = np.linspace(0, 1, max(len(cdf_a), len(cdf_b)))
    interp_a = np.interp(grid, np.linspace(0, 1, len(cdf_a)), cdf_a)
    interp_b = np.interp(grid, np.linspace(0, 1, len(cdf_b)), cdf_b)

    ks = float(np.max(np.abs(interp_a - interp_b)))
    return 1.0 - ks


# ---------------------------------------------------------------------------
# Demographic subspace similarity (principal angles)
# ---------------------------------------------------------------------------


def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute principal angles between subspaces spanned by columns of
    *A* and *B* (Golub & Van Loan, 2013, sec. 6.4.3).

    Args:
        A: ``(D, k1)`` orthonormal basis for subspace 1.
        B: ``(D, k2)`` orthonormal basis for subspace 2.

    Returns:
        Array of ``min(k1, k2)`` principal angles in radians.
    """
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)

    # SVD of the inner product of the two bases.
    M = Q_A.T @ Q_B
    singular_values = np.linalg.svd(M, compute_uv=False)

    # Clip to [0, 1] before arccos (numerical noise can exceed 1).
    return np.arccos(np.clip(singular_values, 0.0, 1.0))


def demographic_subspace_similarity(
    sae_a: BaseSAE,
    sae_b: BaseSAE,
    sdf_indices_a: np.ndarray,
    sdf_indices_b: np.ndarray,
) -> dict[str, float]:
    """Similarity between demographic subspaces of two SAEs.

    The demographic subspace for each SAE is the span of the encoder
    weight vectors (columns of ``W_enc``) corresponding to the top-k
    SDF features.  Similarity is quantified via:

    * **Mean cosine** of principal angles (1.0 = identical subspaces,
      0.0 = orthogonal).
    * **Grassmann distance** (Frobenius norm of angle vector).

    Args:
        sae_a: First trained SAE.
        sae_b: Second trained SAE.
        sdf_indices_a: SDF feature indices for SAE A.
        sdf_indices_b: SDF feature indices for SAE B.

    Returns:
        Dict with ``mean_cos_principal_angle`` and
        ``grassmann_distance``.
    """
    W_a = sae_a.W_enc.detach().cpu().numpy()  # (input_size_a, hidden_a)
    W_b = sae_b.W_enc.detach().cpu().numpy()

    # Handle case where input sizes differ (different VE architectures).
    d_a = W_a.shape[0]
    d_b = W_b.shape[0]
    if d_a != d_b:
        # Project both into a common dimensionality via zero-padding.
        d_max = max(d_a, d_b)
        if d_a < d_max:
            W_a = np.pad(W_a, ((0, d_max - d_a), (0, 0)))
        if d_b < d_max:
            W_b = np.pad(W_b, ((0, d_max - d_b), (0, 0)))

    # Ensure integer dtype (np.array([]) defaults to float64).
    sdf_indices_a = np.asarray(sdf_indices_a, dtype=np.int64)
    sdf_indices_b = np.asarray(sdf_indices_b, dtype=np.int64)

    if len(sdf_indices_a) == 0 or len(sdf_indices_b) == 0:
        log.warning("Empty SDF index set; returning zero similarity.")
        return {"mean_cos_principal_angle": 0.0, "grassmann_distance": float("inf")}

    A_cols = W_a[:, sdf_indices_a]  # (D, k_a)
    B_cols = W_b[:, sdf_indices_b]  # (D, k_b)

    angles = _principal_angles(A_cols, B_cols)

    return {
        "mean_cos_principal_angle": float(np.cos(angles).mean()),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "n_angles": int(len(angles)),
    }


# ---------------------------------------------------------------------------
# Batch helpers: load activations from HDF5 through SAE
# ---------------------------------------------------------------------------


def _encode_hdf5_through_sae(
    sae: BaseSAE,
    hdf5_path: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode VE latents through an SAE and return mean-pooled activations.

    Args:
        sae: Trained SAE.
        hdf5_path: Path to HDF5 file with VE latents.
        split: ``'training'`` or ``'validation'``.
        batch_size: Batch size for streaming.
        device: Torch device.

    Returns:
        acts: ``(N, hidden_size)`` mean-pooled SAE activations.
        labels: Dict mapping attribute name to ``(N,)`` int array.
    """
    with h5py.File(hdf5_path, "r") as f:
        encoded = np.array(f[split]["encoded"])
        labels = {key: np.array(f[split]["labels"][key]) for key in f[split]["labels"].keys()}

    n_samples, seq_len, embed_dim = encoded.shape
    d_hidden = sae.hidden_size

    all_acts: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(
            range(0, n_samples, batch_size),
            desc="SAE encoding",
        ):
            batch = encoded[i : i + batch_size]
            flat = batch.reshape(-1, embed_dim)
            x = torch.from_numpy(flat).float().to(device)
            acts = sae.encode(x).cpu().numpy()
            bs = batch.shape[0]
            per_sample = acts.reshape(bs, seq_len, d_hidden).mean(axis=1)
            all_acts.append(per_sample)

    return np.concatenate(all_acts), labels


# ---------------------------------------------------------------------------
# Full pairwise analysis
# ---------------------------------------------------------------------------


def run_cross_encoder_analysis(
    sae_checkpoints: list[str],
    hdf5_files: list[str],
    sdf_dirs: list[str],
    encoder_names: list[str],
    attribute: str,
    *,
    split: str = "validation",
    batch_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run full pairwise cross-encoder analysis.

    Args:
        sae_checkpoints: Paths to SAE checkpoints (one per encoder).
        hdf5_files: Paths to HDF5 latent files (one per encoder).
        sdf_dirs: Paths to SDF directories (one per encoder).
        encoder_names: Human-readable encoder names.
        attribute: Demographic attribute for SDF / CKA analysis.
        split: HDF5 split to use.
        batch_size: Batch size for SAE encoding.
        device: Torch device (auto-detected if ``None``).

    Returns:
        Dict with pairwise CKA, SDF overlap, and subspace similarity
        matrices.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(encoder_names)
    assert len(sae_checkpoints) == n
    assert len(hdf5_files) == n
    assert len(sdf_dirs) == n

    # -- Load all SAEs and activations --------------------------------------
    saes: list[BaseSAE] = []
    all_acts: list[np.ndarray] = []
    all_labels: list[dict[str, np.ndarray]] = []
    all_sdf_indices: list[np.ndarray] = []

    for i, name in enumerate(encoder_names):
        log.info("Loading SAE and data for %s ...", name)
        sae = load_sae_from_checkpoint(sae_checkpoints[i], device)
        saes.append(sae)

        acts, labels = _encode_hdf5_through_sae(
            sae,
            hdf5_files[i],
            split,
            batch_size,
            device,
        )
        all_acts.append(acts)
        all_labels.append(labels)

        sdf_path = Path(sdf_dirs[i]) / f"sdfs_{attribute}.npz"
        if sdf_path.exists():
            sdf_data = np.load(sdf_path)
            sdf_idx = np.unique(
                np.concatenate([sdf_data[k] for k in sdf_data.files]),
            )
        else:
            log.warning("SDF file not found: %s; using empty set.", sdf_path)
            sdf_idx = np.array([], dtype=np.int64)
        all_sdf_indices.append(sdf_idx)

        log.info(
            "  %s: %d samples, %d SAE features, %d SDFs",
            name,
            acts.shape[0],
            acts.shape[1],
            len(sdf_idx),
        )

    # -- Align sample counts (use the minimum across all encoders) ----------
    min_n = min(a.shape[0] for a in all_acts)
    log.info("Aligning to %d common samples across all encoders.", min_n)
    all_acts = [a[:min_n] for a in all_acts]

    # Use labels from the first encoder (they should all be the same
    # FairFace images in the same order if extracted consistently).
    common_labels = all_labels[0]
    attr_labels = common_labels[attribute][:min_n]

    # -- Pairwise CKA -------------------------------------------------------
    log.info("Computing pairwise CKA ...")
    cka_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cka_val = compute_cka(all_acts[i], all_acts[j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            log.info(
                "  CKA(%s, %s) = %.4f",
                encoder_names[i],
                encoder_names[j],
                cka_val,
            )

    # -- Pairwise SDF rank overlap ------------------------------------------
    log.info("Computing pairwise SDF rank overlap ...")
    overlap_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = compute_sdf_overlap(
                all_acts[i],
                all_acts[j],
                attr_labels,
            )
            overlap_matrix[i, j] = rho
            overlap_matrix[j, i] = rho
            log.info(
                "  SDF overlap(%s, %s) = %.4f",
                encoder_names[i],
                encoder_names[j],
                rho,
            )

    # -- Pairwise subspace similarity ---------------------------------------
    log.info("Computing pairwise demographic subspace similarity ...")
    subspace_matrix = np.eye(n)
    subspace_details: dict[str, dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim = demographic_subspace_similarity(
                saes[i],
                saes[j],
                all_sdf_indices[i],
                all_sdf_indices[j],
            )
            val = sim["mean_cos_principal_angle"]
            subspace_matrix[i, j] = val
            subspace_matrix[j, i] = val
            key = f"{encoder_names[i]}_vs_{encoder_names[j]}"
            subspace_details[key] = sim
            log.info(
                "  Subspace(%s, %s) = %.4f (Grassmann dist=%.4f)",
                encoder_names[i],
                encoder_names[j],
                val,
                sim["grassmann_distance"],
            )

    return {
        "encoder_names": encoder_names,
        "attribute": attribute,
        "n_common_samples": int(min_n),
        "cka_matrix": cka_matrix.tolist(),
        "sdf_overlap_matrix": overlap_matrix.tolist(),
        "subspace_similarity_matrix": subspace_matrix.tolist(),
        "subspace_details": subspace_details,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Cross-architecture demographic feature universality analysis",
    )
    parser.add_argument(
        "--sae-checkpoints",
        nargs="+",
        required=True,
        help="Paths to SAE checkpoints (one per encoder)",
    )
    parser.add_argument(
        "--hdf5-files",
        nargs="+",
        required=True,
        help="Paths to HDF5 latent files (one per encoder)",
    )
    parser.add_argument(
        "--sdf-dirs",
        nargs="+",
        required=True,
        help="Paths to SDF directories (one per encoder)",
    )
    parser.add_argument(
        "--encoder-names",
        nargs="+",
        required=True,
        help="Human-readable encoder names (e.g. dinov3 siglip2 paligemma2)",
    )
    parser.add_argument(
        "--attribute",
        default="race",
        choices=["race", "gender", "age"],
        help="Demographic attribute for analysis",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["training", "validation"],
        help="HDF5 split to use",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        default="results/cross_encoder",
        help="Output directory for results",
    )
    args = parser.parse_args()

    n_enc = len(args.encoder_names)
    if len(args.sae_checkpoints) != n_enc:
        parser.error("--sae-checkpoints must have the same count as --encoder-names")
    if len(args.hdf5_files) != n_enc:
        parser.error("--hdf5-files must have the same count as --encoder-names")
    if len(args.sdf_dirs) != n_enc:
        parser.error("--sdf-dirs must have the same count as --encoder-names")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_cross_encoder_analysis(
        sae_checkpoints=args.sae_checkpoints,
        hdf5_files=args.hdf5_files,
        sdf_dirs=args.sdf_dirs,
        encoder_names=args.encoder_names,
        attribute=args.attribute,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
    )

    output = {
        "_metadata": build_metadata(args),
        **results,
    }

    results_path = out_dir / f"cross_encoder_{args.attribute}.json"
    with open(results_path, "w") as fp:
        json.dump(output, fp, indent=2, cls=NumpyEncoder)
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
