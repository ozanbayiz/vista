"""Intersectional demographic feature analysis via SAE activations.

Extends the SDF discovery pipeline in ``src.analysis`` to identify SAE
features that are specific to **intersectional** demographic subgroups
(e.g., "young Black women") rather than individual attributes alone.

Three complementary analyses are provided:

1. **Intersectional SDF discovery** -- the existing three-stage pipeline
   from ``src.analysis.run_sdf_pipeline`` applied to composite
   age x gender x race labels.
2. **Interaction effect testing** -- N-way ANOVA (via ``statsmodels``)
   to distinguish genuinely intersectional features from those
   explainable by additive main effects.
3. **Information decomposition** -- binned mutual information between SAE
   features and individual vs. intersectional labels, with bias
   correction (Panzeri & Treves, 1996).

This addresses a documented gap: Krug & Stober (2025) flagged
intersectional bias in vision models as an open problem but did not use
SAEs to probe it.

Usage:
    python -m src.intersectional \\
        --sae-checkpoint checkpoints/sae_paligemma2.ckpt \\
        --hdf5 data/fairface_paligemma2.hdf5 \\
        --output-dir results/intersectional_paligemma2 \\
        --min-subgroup-size 30
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.analysis import (
    compute_alignment_rate,
    encode_fairface_through_sae,
    run_sdf_pipeline,
)
from src.utils import NumpyEncoder, build_metadata, load_sae_from_checkpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intersectional label construction
# ---------------------------------------------------------------------------


def build_intersectional_labels(
    labels: dict[str, np.ndarray],
    *,
    min_subgroup_size: int = 30,
) -> tuple[np.ndarray, dict[int, dict[str, int]], list[str]]:
    """Combine age x gender x race labels into composite intersectional classes.

    Each unique (age, gender, race) triple is assigned a new integer class
    label.  Subgroups with fewer than *min_subgroup_size* samples are
    assigned the sentinel label ``-1`` and are excluded from downstream
    analysis.

    Args:
        labels: Dict mapping ``'age'``, ``'gender'``, ``'race'`` to
            ``(N,)`` int arrays.
        min_subgroup_size: Minimum number of samples for a subgroup to
            be retained.

    Returns:
        composite_labels: ``(N,)`` int array of composite class labels
            (``-1`` for excluded samples).
        mapping: Dict from composite label to constituent attribute
            values, e.g. ``{0: {'age': 3, 'gender': 1, 'race': 1}}``.
        readable_names: List of human-readable subgroup names indexed
            by composite label.
    """
    from src.vlm_generate import AGE_NAMES, GENDER_NAMES, RACE_NAMES

    age = labels["age"]
    gender = labels["gender"]
    race = labels["race"]

    n = len(age)
    assert len(gender) == n and len(race) == n, "Label arrays must have equal length"

    # Build tuples and count subgroup sizes.
    triples = list(zip(age.tolist(), gender.tolist(), race.tolist()))
    counts = Counter(triples)

    # Assign composite labels, skipping rare subgroups.
    triple_to_cls: dict[tuple[int, int, int], int] = {}
    next_cls = 0
    for triple, cnt in counts.items():
        if cnt >= min_subgroup_size:
            triple_to_cls[triple] = next_cls
            next_cls += 1

    composite = np.full(n, -1, dtype=np.int64)
    for i, triple in enumerate(triples):
        cls = triple_to_cls.get(triple, -1)
        composite[i] = cls

    # Build mapping and readable names.
    mapping: dict[int, dict[str, int]] = {}
    readable_names: list[str] = [""] * next_cls
    for triple, cls in triple_to_cls.items():
        a, g, r = triple
        mapping[cls] = {"age": a, "gender": g, "race": r}
        readable_names[cls] = f"{AGE_NAMES.get(a, str(a))}_{GENDER_NAMES.get(g, str(g))}_{RACE_NAMES.get(r, str(r))}"

    n_valid = int((composite >= 0).sum())
    n_classes = next_cls
    log.info(
        "Built %d intersectional classes from %d valid samples (%d excluded below min_subgroup_size=%d)",
        n_classes,
        n_valid,
        n - n_valid,
        min_subgroup_size,
    )
    for cls in range(n_classes):
        cnt = int((composite == cls).sum())
        log.info("  [%d] %s: n=%d", cls, readable_names[cls], cnt)

    return composite, mapping, readable_names


# ---------------------------------------------------------------------------
# Intersectional SDF pipeline
# ---------------------------------------------------------------------------


def intersectional_sdf_pipeline(
    acts: np.ndarray,
    composite_labels: np.ndarray,
    *,
    k1: int = 200,
    k2: int = 100,
    k3: int = 50,
    k_top: int = 20,
) -> dict[int, np.ndarray]:
    """Run the three-stage SDF pipeline on intersectional labels.

    Wraps ``src.analysis.run_sdf_pipeline`` after filtering out excluded
    samples (those with composite label ``-1``).

    Args:
        acts: ``(N, D)`` mean-pooled SAE activations.
        composite_labels: ``(N,)`` composite intersectional labels
            (``-1`` = excluded).
        k1: Stage 1 cutoff (activation frequency).
        k2: Stage 2 cutoff (mean activation).
        k3: Stage 3 cutoff (entropy filter).
        k_top: Top-k for entropy computation.

    Returns:
        Dict mapping composite class to SDF feature indices.
    """
    mask = composite_labels >= 0
    filtered_acts = acts[mask]
    filtered_labels = composite_labels[mask]

    log.info(
        "Running intersectional SDF pipeline on %d samples, %d classes ...",
        len(filtered_acts),
        len(np.unique(filtered_labels)),
    )
    return run_sdf_pipeline(
        filtered_acts,
        filtered_labels,
        k1,
        k2,
        k3,
        k_top,
    )


# ---------------------------------------------------------------------------
# Interaction effect testing (N-way ANOVA)
# ---------------------------------------------------------------------------


def compute_interaction_effects(
    acts: np.ndarray,
    labels: dict[str, np.ndarray],
    candidate_features: np.ndarray,
    *,
    alpha: float = 0.05,
    min_effect_size: float = 0.01,
) -> list[dict[str, Any]]:
    """Test each candidate feature for genuine intersectional effects.

    For each feature, a three-way ANOVA is fitted with age, gender, and
    race as factors.  A feature is classified as *genuinely intersectional*
    if at least one interaction term is significant (Bonferroni-corrected
    *p* < *alpha*) **and** its partial eta-squared exceeds
    *min_effect_size*.

    Args:
        acts: ``(N, D)`` mean-pooled SAE activations.
        labels: Dict with ``'age'``, ``'gender'``, ``'race'`` arrays.
        candidate_features: Feature indices to test.
        alpha: Significance threshold (before Bonferroni correction).
        min_effect_size: Minimum partial eta-squared for the interaction
            term to count as substantive.

    Returns:
        List of per-feature result dicts with classification
        (``'intersectional'`` vs. ``'main_effect_only'``), p-values, and
        effect sizes.
    """
    try:
        import pandas as pd
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError as exc:
        raise ImportError(
            "statsmodels and pandas are required for interaction effect testing. "
            "Install with: pip install statsmodels pandas",
        ) from exc

    n = len(acts)
    n_features = len(candidate_features)
    bonferroni = n_features  # correct for multiple comparisons

    age = labels["age"][:n]
    gender = labels["gender"][:n]
    race = labels["race"][:n]

    results: list[dict[str, Any]] = []

    # Interaction term names in the ANOVA table.
    interaction_terms = [
        "C(age):C(gender)",
        "C(age):C(race)",
        "C(gender):C(race)",
        "C(age):C(gender):C(race)",
    ]

    for idx, feat_idx in enumerate(
        tqdm(candidate_features, desc="ANOVA interaction tests"),
    ):
        y = acts[:, feat_idx].astype(np.float64)

        df = pd.DataFrame(
            {
                "y": y,
                "age": pd.Categorical(age),
                "gender": pd.Categorical(gender),
                "race": pd.Categorical(race),
            }
        )

        try:
            model = ols(
                "y ~ C(age) * C(gender) * C(race)",
                data=df,
            ).fit()
            table = anova_lm(model, typ=2)
        except Exception as exc:
            log.warning(
                "ANOVA failed for feature %d: %s",
                feat_idx,
                exc,
            )
            results.append(
                {
                    "feature_index": int(feat_idx),
                    "classification": "error",
                    "error": str(exc),
                }
            )
            continue

        # Check each interaction term.
        is_intersectional = False
        interaction_results: dict[str, dict[str, float]] = {}

        ss_total = float(table["sum_sq"].sum())

        for term in interaction_terms:
            if term not in table.index:
                continue
            row = table.loc[term]
            p_val = float(row["PR(>F)"]) if "PR(>F)" in row.index else 1.0
            ss = float(row["sum_sq"])
            ss_resid = float(table.loc["Residual", "sum_sq"])
            # Partial eta-squared = SS_effect / (SS_effect + SS_residual)
            partial_eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0

            corrected_p = min(p_val * bonferroni, 1.0)

            interaction_results[term] = {
                "p_value": p_val,
                "p_corrected": corrected_p,
                "partial_eta_squared": partial_eta_sq,
                "significant": corrected_p < alpha and partial_eta_sq > min_effect_size,
            }

            if corrected_p < alpha and partial_eta_sq > min_effect_size:
                is_intersectional = True

        results.append(
            {
                "feature_index": int(feat_idx),
                "classification": "intersectional" if is_intersectional else "main_effect_only",
                "interactions": interaction_results,
            }
        )

        if (idx + 1) % 50 == 0:
            n_int = sum(1 for r in results if r["classification"] == "intersectional")
            log.info(
                "  Tested %d/%d features: %d intersectional so far",
                idx + 1,
                n_features,
                n_int,
            )

    n_intersectional = sum(1 for r in results if r["classification"] == "intersectional")
    n_main = sum(1 for r in results if r["classification"] == "main_effect_only")
    log.info(
        "Interaction testing complete: %d intersectional, %d main-effect-only, %d errors (of %d candidates)",
        n_intersectional,
        n_main,
        n_features - n_intersectional - n_main,
        n_features,
    )

    return results


# ---------------------------------------------------------------------------
# Mutual information decomposition
# ---------------------------------------------------------------------------


def _binned_mi(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Binned mutual information with Panzeri-Treves bias correction.

    Discretises *x* into *n_bins* equal-width bins and computes
    ``I(X; Y)`` from the empirical joint distribution, applying the
    analytical bias correction of Panzeri & Treves (1996):

        ``bias = (|X| - 1)(|Y| - 1) / (2 * N * ln(2))``

    Args:
        x: ``(N,)`` continuous feature activations.
        y: ``(N,)`` integer class labels.
        n_bins: Number of bins for discretising *x*.

    Returns:
        Bias-corrected MI in bits (non-negative, clipped to 0).
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Discretise x into equal-width bins.
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return 0.0
    bins = np.linspace(x_min, x_max, n_bins + 1)
    x_binned = np.digitize(x, bins[1:-1])  # values in [0, n_bins-1]

    # Joint and marginal counts.
    unique_y = np.unique(y)
    n_x = n_bins
    n_y = len(unique_y)

    joint = np.zeros((n_x, n_y), dtype=np.float64)
    y_map = {v: i for i, v in enumerate(unique_y)}
    for xi, yi in zip(x_binned, y):
        joint[xi, y_map[yi]] += 1.0

    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = sum p(x,y) log2(p(x,y) / (p(x) p(y)))
    mi = 0.0
    for i in range(n_x):
        for j in range(n_y):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

    # Panzeri-Treves bias correction.
    bias = (n_x - 1) * (n_y - 1) / (2 * n * np.log(2))
    return float(max(mi - bias, 0.0))


def information_decomposition(
    acts: np.ndarray,
    labels: dict[str, np.ndarray],
    composite_labels: np.ndarray,
    candidate_features: np.ndarray,
    *,
    n_bins: int = 20,
) -> list[dict[str, Any]]:
    """Decompose mutual information between features and demographic labels.

    For each candidate feature, computes MI with:

    * Individual attributes (age, gender, race).
    * Pairwise intersections (age x gender, age x race, gender x race).
    * Full triple intersection (age x gender x race).

    Synergistic information is estimated as the MI with the full
    intersection minus the maximum single-attribute MI.

    Args:
        acts: ``(N, D)`` mean-pooled SAE activations.
        labels: Dict with ``'age'``, ``'gender'``, ``'race'`` arrays.
        composite_labels: ``(N,)`` composite intersectional labels.
        candidate_features: Feature indices to analyse.
        n_bins: Number of bins for MI estimation.

    Returns:
        List of per-feature dicts with MI values and synergy estimates.
    """
    n = len(acts)
    age = labels["age"][:n]
    gender = labels["gender"][:n]
    race = labels["race"][:n]

    # Build pairwise composite labels (deterministic encoding).
    n_gender = int(gender.max()) + 1
    n_race = int(race.max()) + 1

    pair_ag = age * n_gender + gender
    pair_ar = age * n_race + race
    pair_gr = gender * n_race + race

    label_sets = {
        "age": age,
        "gender": gender,
        "race": race,
        "age_x_gender": pair_ag,
        "age_x_race": pair_ar,
        "gender_x_race": pair_gr,
        "age_x_gender_x_race": composite_labels[:n],
    }

    results: list[dict[str, Any]] = []

    for feat_idx in tqdm(candidate_features, desc="MI decomposition"):
        x = acts[:, feat_idx].astype(np.float64)
        mi_vals: dict[str, float] = {}

        for name, y in label_sets.items():
            # Exclude samples with composite = -1 for intersectional labels.
            if "x" in name:
                mask = y >= 0
                mi_vals[name] = _binned_mi(x[mask], y[mask], n_bins)
            else:
                mi_vals[name] = _binned_mi(x, y, n_bins)

        max_single = max(mi_vals["age"], mi_vals["gender"], mi_vals["race"])
        triple_mi = mi_vals.get("age_x_gender_x_race", 0.0)
        synergy = triple_mi - max_single

        results.append(
            {
                "feature_index": int(feat_idx),
                "mi": mi_vals,
                "max_single_mi": max_single,
                "triple_mi": triple_mi,
                "synergy_estimate": synergy,
            }
        )

    n_synergistic = sum(1 for r in results if r["synergy_estimate"] > 0.01)
    log.info(
        "MI decomposition complete: %d/%d features with synergy > 0.01 bits",
        n_synergistic,
        len(results),
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Intersectional demographic feature analysis via SAE activations",
    )
    parser.add_argument(
        "--sae-checkpoint",
        required=True,
        help="Path to trained SAE Lightning checkpoint",
    )
    parser.add_argument(
        "--hdf5",
        required=True,
        help="Path to FairFace VE latent HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/intersectional",
        help="Output directory for results",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--mode",
        default="validation",
        choices=["training", "validation"],
        help="HDF5 split to use",
    )
    parser.add_argument(
        "--min-subgroup-size",
        type=int,
        default=30,
        help="Minimum samples per intersectional subgroup",
    )
    parser.add_argument("--k1", type=int, default=200, help="Stage 1 cutoff")
    parser.add_argument("--k2", type=int, default=100, help="Stage 2 cutoff")
    parser.add_argument("--k3", type=int, default=50, help="Stage 3 cutoff")
    parser.add_argument("--k-top", type=int, default=20, help="Top-k for entropy")
    parser.add_argument(
        "--skip-anova",
        action="store_true",
        help="Skip ANOVA interaction testing (faster)",
    )
    parser.add_argument(
        "--skip-mi",
        action="store_true",
        help="Skip MI decomposition (faster)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load SAE and encode data -------------------------------------------
    log.info("Loading SAE from %s", args.sae_checkpoint)
    sae = load_sae_from_checkpoint(args.sae_checkpoint, device)

    log.info("Encoding FairFace latents through SAE ...")
    acts, labels = encode_fairface_through_sae(
        sae,
        args.hdf5,
        args.mode,
        args.batch_size,
        device,
    )
    log.info("Activations shape: %s", acts.shape)

    # -- Build intersectional labels ----------------------------------------
    composite, mapping, readable = build_intersectional_labels(
        labels,
        min_subgroup_size=args.min_subgroup_size,
    )

    # -- Run intersectional SDF pipeline ------------------------------------
    log.info("=== Intersectional SDF Pipeline ===")
    int_sdfs = intersectional_sdf_pipeline(
        acts,
        composite,
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        k_top=args.k_top,
    )

    # Compute alignment rates.
    mask = composite >= 0
    alignment = compute_alignment_rate(
        acts[mask],
        composite[mask],
        int_sdfs,
        args.k_top,
    )

    # Merge all intersectional SDFs for downstream analysis.
    if int_sdfs:
        all_int_features = np.unique(
            np.concatenate([v for v in int_sdfs.values()]),
        )
    else:
        all_int_features = np.array([], dtype=np.int64)
    log.info("Total unique intersectional SDF features: %d", len(all_int_features))

    # -- Also run single-attribute pipelines for comparison -----------------
    log.info("=== Single-Attribute SDF Pipelines (for comparison) ===")
    single_sdfs: dict[str, dict[int, np.ndarray]] = {}
    for attr in ("age", "gender", "race"):
        if attr not in labels:
            continue
        log.info("  Running SDF pipeline for '%s' ...", attr)
        single_sdfs[attr] = run_sdf_pipeline(
            acts,
            labels[attr],
            args.k1,
            args.k2,
            args.k3,
            args.k_top,
        )

    # Merge all single-attribute features.
    all_single_features: list[np.ndarray] = []
    for attr_sdfs in single_sdfs.values():
        for feat_arr in attr_sdfs.values():
            all_single_features.append(feat_arr)
    if all_single_features:
        single_union = np.unique(np.concatenate(all_single_features))
    else:
        single_union = np.array([], dtype=np.int64)

    # Features unique to intersectional analysis.
    novel_features = np.setdiff1d(all_int_features, single_union)
    log.info(
        "Features unique to intersectional analysis: %d / %d",
        len(novel_features),
        len(all_int_features),
    )

    # -- Interaction effect testing -----------------------------------------
    anova_results: list[dict] | None = None
    if not args.skip_anova and len(all_int_features) > 0:
        log.info("=== ANOVA Interaction Effect Testing ===")
        anova_results = compute_interaction_effects(
            acts,
            labels,
            all_int_features,
        )
    elif args.skip_anova:
        log.info("Skipping ANOVA interaction testing (--skip-anova)")

    # -- MI decomposition ---------------------------------------------------
    mi_results: list[dict] | None = None
    if not args.skip_mi and len(all_int_features) > 0:
        log.info("=== Mutual Information Decomposition ===")
        mi_results = information_decomposition(
            acts,
            labels,
            composite,
            all_int_features,
        )
    elif args.skip_mi:
        log.info("Skipping MI decomposition (--skip-mi)")

    # -- Save results -------------------------------------------------------
    output: dict[str, Any] = {
        "_metadata": build_metadata(args),
        "intersectional_sdfs": {str(c): feats.tolist() for c, feats in int_sdfs.items()},
        "intersectional_alignment": alignment,
        "class_mapping": {str(k): v for k, v in mapping.items()},
        "class_names": readable,
        "comparison": {
            "n_intersectional_features": int(len(all_int_features)),
            "n_single_attribute_features": int(len(single_union)),
            "n_novel_intersectional": int(len(novel_features)),
            "novel_feature_indices": novel_features.tolist(),
        },
        "params": {
            "k1": args.k1,
            "k2": args.k2,
            "k3": args.k3,
            "k_top": args.k_top,
            "min_subgroup_size": args.min_subgroup_size,
        },
    }

    if anova_results is not None:
        output["anova_interaction"] = anova_results
        n_int = sum(1 for r in anova_results if r["classification"] == "intersectional")
        output["anova_summary"] = {
            "n_tested": len(anova_results),
            "n_intersectional": n_int,
            "n_main_effect_only": sum(1 for r in anova_results if r["classification"] == "main_effect_only"),
        }

    if mi_results is not None:
        output["mi_decomposition"] = mi_results
        synergies = [r["synergy_estimate"] for r in mi_results]
        output["mi_summary"] = {
            "n_features": len(mi_results),
            "mean_synergy": float(np.mean(synergies)) if synergies else 0.0,
            "max_synergy": float(np.max(synergies)) if synergies else 0.0,
            "n_synergistic_gt_001": sum(1 for s in synergies if s > 0.01),
        }

    # Save per-class SDF arrays for downstream use.
    np.savez(
        out_dir / "sdfs_intersectional.npz",
        **{str(c): feats for c, feats in int_sdfs.items()},
    )

    results_path = out_dir / "intersectional_results.json"
    with open(results_path, "w") as fp:
        json.dump(output, fp, indent=2, cls=NumpyEncoder)
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
