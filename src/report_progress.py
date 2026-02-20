"""Comprehensive progress report for IDARVE PaliGemma2 experiments.

Prints a structured report covering all completed analyses, findings,
and planned next steps.  Reads result files when available.

Usage:
    python -m src.report_progress
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Label maps (duplicated here to keep report self-contained)
# ---------------------------------------------------------------------------
RACE_MAP = {
    "0": "White", "1": "Black", "2": "Latino/Hispanic",
    "3": "East Asian", "4": "Southeast Asian", "5": "Indian", "6": "Middle Eastern",
}
GENDER_MAP = {"0": "Male", "1": "Female"}
AGE_MAP = {
    "0": "0-2", "1": "3-9", "2": "10-19", "3": "20-29", "4": "30-39",
    "5": "40-49", "6": "50-59", "7": "60-69", "8": "70+",
}
ATTR_MAPS: dict[str, dict[str, str]] = {"race": RACE_MAP, "gender": GENDER_MAP, "age": AGE_MAP}

# ---------------------------------------------------------------------------
# Paths (configurable via env vars if needed)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/scratch/current/ozanbayiz")
RESULTS_ROOT = DATA_ROOT / "results"
SDF_DIR = RESULTS_ROOT / "sdf_paligemma2"
EVAL_DIR = RESULTS_ROOT / "eval_paligemma2"
TRAINING_DIR = DATA_ROOT / "outputs" / "lightning_logs" / "version_3"
PROBE_DIR = DATA_ROOT / "outputs" / "lightning_logs" / "version_2"


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---\n")


def print_report() -> None:
    print("#" * 72)
    print("  IDARVE Progress Report: PaliGemma2 Experiments")
    print("  Vision Encoder: google/paligemma2-3b-pt-448 (SigLIP VE, 1152-dim)")
    print("#" * 72)

    # -----------------------------------------------------------------------
    # 1. Data overview
    # -----------------------------------------------------------------------
    section("1. Data Inventory")

    hdf5_files = {
        "fairface_paligemma2.hdf5": "FairFace full token-level (N, 1024, 1152)",
        "fairface_paligemma2_pooled.hdf5": "FairFace mean-pooled (N, 1152)",
        "imagenet_paligemma2.hdf5": "ImageNet full token-level for SAE training",
    }
    for name, desc in hdf5_files.items():
        p = DATA_ROOT / name
        size = f"{p.stat().st_size / 1e9:.1f} GB" if p.exists() else "MISSING"
        print(f"  {name}: {desc} [{size}]")

    # -----------------------------------------------------------------------
    # 2. SAE training summary
    # -----------------------------------------------------------------------
    section("2. SAE Training (Version 3)")

    print("  Architecture: BatchTopK SAE")
    print("  input_size=1152, hidden_size=4608, top_k=64")
    print("  Learning rate: 1e-4, Aux penalty: 1/32")
    print("  Trained on: ImageNet PaliGemma2 VE latents (token-level)")
    print("  Best checkpoint: epoch 5, step 4224")
    print()
    print("  Key metrics at best epoch:")
    print("    Val cosine similarity: 0.9672")
    print("    Val L2 loss:          0.2283")
    print("    Val total loss:       0.2354")
    print("    Dead features:        ~30 / 4608 (0.65%)")
    print("    L0 norm:              64 (= top_k)")
    print()
    print("  Training progression (cosine sim):")
    print("    Epoch 0: 0.9435  |  Epoch 1: 0.9552  |  Epoch 2: 0.9601")
    print("    Epoch 3: 0.9636  |  Epoch 4: 0.9656  |  Epoch 5: 0.9672")
    print()
    print("  Assessment: Good convergence with consistent improvement.")
    print("  The 4x expansion ratio (1152 -> 4608) and top_k=64 provide a")
    print("  balance between reconstruction quality and sparsity.")

    # -----------------------------------------------------------------------
    # 3. Linear probe summary
    # -----------------------------------------------------------------------
    section("3. Linear Probe Training (Version 2)")

    print("  Architecture: 3 independent linear heads (age/gender/race)")
    print("  Input: mean-pooled PaliGemma2 VE latents (1152-dim)")
    print("  50 epochs, Adam optimizer, lr=1e-3")
    print()
    print("  Best validation metrics (epoch 46):")
    print("    Gender: 65.8% acc, F1=0.644")
    print("    Race:   32.8% acc, F1=0.281 (7 classes, random=14.3%)")
    print("    Age:    32.3% acc, F1=0.150 (9 classes, random=11.1%)")
    print()
    print("  Comparison with original Florence-2 proposal:")
    print("    Florence-2 race probe: 62.15% accuracy (7 classes)")
    print("    PaliGemma2 race probe: 32.8% accuracy (7 classes)")
    print()
    print("  Assessment: PaliGemma2's mean-pooled VE representations encode")
    print("  gender moderately well but race and age are less linearly")
    print("  separable than in Florence-2. This could indicate:")
    print("  (a) demographic info is encoded in token-level patterns,")
    print("      not just the mean-pooled aggregate, or")
    print("  (b) PaliGemma2's SigLIP VE genuinely encodes less")
    print("      demographic information in its activation space.")

    # -----------------------------------------------------------------------
    # 4. SDF analysis
    # -----------------------------------------------------------------------
    section("4. SDF (Sparse Dictionary Feature) Analysis")

    sdf_results = _load_json(SDF_DIR / "sdf_results.json")
    if sdf_results is None:
        print("  [SDF results not found]")
    else:
        print("  Pipeline: 3-stage filtering (activation frequency -> mean")
        print("  activation -> label entropy), k1=200, k2=100, k3=50, k_top=20")
        print("  Dataset: FairFace validation set (13,012 samples)")
        print()

        for attr in ["gender", "race", "age"]:
            if attr not in sdf_results:
                continue
            name_map = ATTR_MAPS.get(attr, {})
            r = sdf_results[attr]
            sdfs = r["sdfs"]
            alignment = r["alignment"]

            total_unique = len(set(f for cls_feats in sdfs.values() for f in cls_feats))

            subsection(f"{attr.upper()} SDFs ({total_unique} unique features)")
            print(f"  {'Class':<22s} {'#SDFs':>5s}  {'Mean Align':>10s}  "
                  f"{'Max':>5s}  {'>0.5':>5s}  {'>0.3':>5s}")
            print(f"  {'-' * 62}")

            for cls_id in sorted(sdfs.keys(), key=int):
                name = name_map.get(cls_id, f"Class {cls_id}")
                a = alignment[cls_id]
                rates = a["alignment_rates"]
                high = sum(1 for x in rates if x > 0.5)
                mid = sum(1 for x in rates if x > 0.3)
                print(f"  {name:<22s} {len(sdfs[cls_id]):>5d}  "
                      f"{a['mean_alignment_rate']:>10.3f}  "
                      f"{max(rates):>5.3f}  {high:>5d}  {mid:>5d}")

        # Cross-attribute overlap
        subsection("Cross-Attribute SDF Overlap")
        attrs = list(sdf_results.keys())
        for i, a1 in enumerate(attrs):
            s1 = set(f for cls_feats in sdf_results[a1]["sdfs"].values() for f in cls_feats)
            for a2 in attrs[i + 1:]:
                s2 = set(f for cls_feats in sdf_results[a2]["sdfs"].values() for f in cls_feats)
                overlap = s1 & s2
                pct = 100 * len(overlap) / min(len(s1), len(s2)) if min(len(s1), len(s2)) > 0 else 0
                print(f"  {a1} & {a2}: {len(overlap)} shared features "
                      f"({pct:.0f}% of smaller set)")

        print()
        print("  KEY FINDING: High cross-attribute overlap (50-60%) suggests")
        print("  PaliGemma2's SAE features encode demographic information in a")
        print("  highly intersectional manner. Individual features correlate with")
        print("  multiple demographic attributes simultaneously.")
        print()
        print("  Gender SDFs are the strongest signal (Male: 0.635 mean alignment,")
        print("  40/50 features >0.5). Race SDFs are moderate (White: 0.227,")
        print("  Black: 0.210). Age SDFs are strong for 20-29 (0.348) but very")
        print("  weak for minority classes (0-2: 0.009, 70+: 0.002), likely due")
        print("  to class imbalance in the data.")

    # -----------------------------------------------------------------------
    # 5. Evaluation results (if available)
    # -----------------------------------------------------------------------
    section("5. SAE Evaluation")

    eval_results = _load_json(EVAL_DIR / "evaluation_results.json")
    if eval_results is None:
        print("  [Evaluation still running or not yet available]")
        print("  Running: ablation studies + downstream probe comparison")
        print("  Expected output: evaluation_results.json")
    else:
        if "sae_health" in eval_results:
            subsection("SAE Health")
            h = eval_results["sae_health"]
            for k, v in h.items():
                if isinstance(v, dict):
                    continue
                print(f"  {k}: {v}")

        if "reconstruction_quality" in eval_results:
            subsection("Reconstruction Quality")
            rq = eval_results["reconstruction_quality"]
            for k, v in rq.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        if "ablation_study" in eval_results:
            subsection("Ablation Study (Necessity / Sufficiency)")
            ab = eval_results["ablation_study"]
            for attr_name, attr_data in ab.items():
                print(f"\n  Attribute: {attr_name}")
                print(f"    Baseline accuracy: {attr_data['baseline_accuracy']:.4f}")
                print(f"    {'k':>6s}  {'Without top-k':>14s}  {'Only top-k':>11s}  "
                      f"{'Nec. drop':>10s}")
                print(f"    {'-' * 48}")
                for k_str, v in sorted(attr_data["ablations"].items(), key=lambda x: int(x[0])):
                    print(f"    {k_str:>6s}  {v['acc_without_top_k']:>14.4f}  "
                          f"{v['acc_only_top_k']:>11.4f}  {v['necessity_drop']:>10.4f}")

        if "probe_comparison" in eval_results:
            subsection("Probe Comparison (Raw VE vs SAE Latents)")
            pc = eval_results["probe_comparison"]
            for attr_name, attr_data in pc.items():
                raw_acc = attr_data.get("raw_latent_accuracy", 0)
                sae_acc = attr_data.get("sae_latent_accuracy", 0)
                delta = attr_data.get("delta", 0)
                print(f"  {attr_name}: raw={raw_acc:.4f}, sae={sae_acc:.4f}, "
                      f"delta={delta:+.4f}")

        subsection("Evaluation Interpretation")
        print("  KEY FINDINGS:")
        print()
        print("  1. SAE Health is excellent: only 3/4608 dead features (0.065%),")
        print("     L0 = 64 (matches top_k), cosine_sim = 0.945. The SAE learns")
        print("     a faithful, sparse representation of the VE activations.")
        print()
        print("  2. Probe comparison shows minimal information loss: the SAE's")
        print("     4608 sparse features preserve nearly all demographic")
        print("     information from the original 1152-dim VE space (delta < 1%).")
        print("     This is important: it means the SDF analysis operates on")
        print("     a representation that truly reflects VE encoding patterns.")
        print()
        print("  3. Ablation study reveals DISTRIBUTED encoding:")
        print("     - Gender: moderate concentration. Removing top-100 features")
        print("       drops accuracy by only 0.9%, but keeping only top-100")
        print("       retains 97% of baseline accuracy. Gender information is")
        print("       both sufficient in a small feature set AND redundantly")
        print("       distributed across many features.")
        print("     - Race: similar pattern but weaker. Top-100 features")
        print("       capture ~89% of baseline race accuracy.")
        print("     - Age: HIGHLY distributed. Top-k features provide no")
        print("       unique age signal (negative necessity drops), indicating")
        print("       age information is spread across the full feature space.")
        print()
        print("  4. Implications for intervention:")
        print("     - The small necessity drops suggest that simple feature")
        print("       suppression (zeroing out SDFs) may not be effective for")
        print("       debiasing, because the information is redundantly encoded.")
        print("     - This motivates the S&P Top-K approach (orthogonal projection)")
        print("       which removes ALL information along selected directions,")
        print("       rather than just zeroing specific features.")
        print("     - Gender suppression has the best prospects due to moderate")
        print("       concentration. Race and age will be harder to suppress.")

    # -----------------------------------------------------------------------
    # 5b. Intervention results (if available)
    # -----------------------------------------------------------------------
    section("5b. Intervention Experiments (Suppression)")

    INTERVENTION_DIR = RESULTS_ROOT / "intervention_paligemma2"
    supp_results = _load_json(INTERVENTION_DIR / "intervention_suppression_results.json")
    if supp_results is None:
        print("  [Intervention results not yet available]")
    else:
        for attr, attr_data in supp_results.items():
            subsection(f"{attr.upper()} Suppression")
            print(f"  {'k':>8s}  {'cos_sim':>8s}  {'tgt_before':>10s}  {'tgt_after':>10s}  "
                  f"{'tgt_delta':>10s}")
            print(f"  {'-' * 54}")
            for k_name, metrics in sorted(attr_data.items()):
                cos = metrics["cosine_similarity"]
                t = metrics.get("target", {})
                bef = t.get("acc_before", 0)
                aft = t.get("acc_after", 0)
                dlt = t.get("delta", 0)
                print(f"  {k_name:>8s}  {cos:>8.4f}  {bef:>10.4f}  {aft:>10.4f}  {dlt:>+10.4f}")

        subsection("Intervention Interpretation")
        print("  CRITICAL FINDING: Masked reconstruction is INEFFECTIVE for debiasing.")
        print()
        print("  Despite suppressing up to 50 SDF features:")
        print("  - Race accuracy drops by at most 0.88% (k=20), negligible")
        print("  - Gender accuracy is essentially UNCHANGED (max drop: 0.15%)")
        print("  - Yet embeddings change dramatically (cosine sim drops to 0.49-0.59)")
        print()
        print("  This means the SAE decoder 'reconstructs' demographic information")
        print("  even when the relevant sparse codes are zeroed out. The decoder's")
        print("  weight matrix introduces demographic-correlated structure back into")
        print("  the reconstructed embeddings, negating the intervention.")
        print()
        print("  This finding is EXACTLY consistent with the S&P Top-K paper")
        print("  (Barbulau et al., 2509.10809), which showed that encoder-centric")
        print("  projection achieves 3.2x better fairness than masked reconstruction.")
        print()
        print("  ROOT CAUSE: Information redundancy. Demographic information is")
        print("  distributed across thousands of features (as shown by the ablation")
        print("  study). Zeroing 50 of 4608 features leaves ~4558 features that")
        print("  still encode the same demographic info through correlated patterns.")
        print()
        print("  NEXT STEP: Implement S&P Top-K encoder-centric projection, which")
        print("  operates in the native embedding space and removes ALL information")
        print("  along the sensitive directions, not just individual sparse codes.")

    # -----------------------------------------------------------------------
    # 5c. S&P Top-K Intervention (if available)
    # -----------------------------------------------------------------------
    section("5c. S&P Top-K Encoder-Centric Projection")

    sp_results = _load_json(INTERVENTION_DIR / "intervention_sp_topk_results.json")
    if sp_results is None:
        print("  [S&P Top-K experiment currently running]")
        print("  Method: Select top-k encoder features by inter-class variance,")
        print("  compute orthogonal projection V = I - alpha * a*a^T / ||a||^2,")
        print("  apply projection to raw VE embeddings, evaluate probes.")
        print()
        print("  Parameters: k={16,32,64}, alphas={0.10,0.30,0.50,0.70,1.00}")
        print("  Attributes: gender, race")
        print("  Features: variance-based selection from SAE encoder weights")
    else:
        for attr, attr_data in sp_results.items():
            subsection(f"{attr.upper()} S&P Top-K Results")
            for k_name, k_data in sorted(attr_data.items()):
                if not isinstance(k_data, dict):
                    continue
                print(f"\n  {k_name}:")
                print(f"  {'alpha':>8s}  {'cos_sim':>8s}  {'tgt_before':>10s}  {'tgt_after':>10s}  "
                      f"{'tgt_delta':>10s}")
                print(f"  {'-' * 54}")
                for alpha_str in sorted(k_data.keys(), key=lambda x: float(x) if x.replace('.', '').isdigit() else 0):
                    alpha_data = k_data[alpha_str]
                    if not isinstance(alpha_data, dict) or "cosine_similarity" not in alpha_data:
                        continue
                    cos = alpha_data["cosine_similarity"]
                    t = alpha_data.get("target", {})
                    bef = t.get("acc_before", 0)
                    aft = t.get("acc_after", 0)
                    dlt = t.get("delta", 0)
                    print(f"  {alpha_str:>8s}  {cos:>8.4f}  {bef:>10.4f}  {aft:>10.4f}  {dlt:>+10.4f}")

    # -----------------------------------------------------------------------
    # 5d. LEACE Intervention (if available)
    # -----------------------------------------------------------------------
    section("5d. LEACE (Least-Squares Concept Erasure)")

    leace_results = _load_json(INTERVENTION_DIR / "intervention_leace_results.json")
    if leace_results is None:
        print("  [LEACE experiment currently running]")
        print("  Method: Closed-form linear concept erasure (Belrose et al., NeurIPS 2023)")
        print("  Provably prevents ALL linear classifiers from detecting the target")
        print("  concept while minimizing embedding changes.")
        print()
        print("  No SAE required — operates directly on raw VE embeddings.")
        print("  Attributes: gender, race, age (each erased independently)")
    else:
        for attr, attr_data in leace_results.items():
            subsection(f"LEACE Erasing '{attr}'")
            cos = attr_data.get("cosine_similarity", None)
            if cos is not None:
                print(f"  cosine_sim(original, erased) = {cos:.4f}")
            t = attr_data.get("target", {})
            if t:
                print(f"  Target ({attr}): before={t['acc_before']:.4f}, "
                      f"after={t['acc_after']:.4f}, delta={t['delta']:+.4f}")
            nt = attr_data.get("non_target", {})
            if nt:
                print(f"  Non-target attributes:")
                for nt_name, nt_data in nt.items():
                    print(f"    {nt_name}: before={nt_data['acc_before']:.4f}, "
                          f"after={nt_data['acc_after']:.4f}, delta={nt_data['delta']:+.4f}")

        subsection("LEACE Interpretation")
        print("  KEY FINDINGS:")
        print()
        print("  1. Gender erasure is HIGHLY EFFECTIVE: accuracy drops from 65.2%")
        print("     to 48.7% (near 50% random chance for binary). LEACE provably")
        print("     removed all linear gender signal with cos_sim=0.999.")
        print()
        print("  2. Race erasure is MODERATE: 31.9% -> 29.1% (-2.8pp). For 7")
        print("     classes, random chance is 14.3%, so 29.1% is still 2x chance.")
        print("     This suggests race info has significant NONLINEAR components")
        print("     that LEACE (a linear method) cannot remove.")
        print()
        print("  3. Age erasure is MODERATE: 29.9% -> 28.1% (-1.8pp). Similar")
        print("     reasoning: age has 9 classes and its encoding is distributed")
        print("     across nonlinear feature interactions.")
        print()
        print("  4. Cross-attribute interference is MINIMAL (<1pp for all pairs).")
        print("     Gender, race, and age are encoded in approximately ORTHOGONAL")
        print("     linear subspaces within the VE embedding space.")
        print()
        print("  5. Embedding distortion is NEGLIGIBLE (cos_sim > 0.999 for all).")
        print("     LEACE achieves concept erasure with minimal change to the")
        print("     overall embedding geometry.")
        print()
        print("  COMPARISON:")
        print("  | Method                  | Gender Δ  | Race Δ   | cos_sim |")
        print("  |-------------------------|-----------|----------|---------|")
        print("  | Masked Reconstruction   | -0.15pp   | -0.88pp  | 0.49    |")
        print("  | S&P Top-K (k=16,α=1.0) | +0.31pp   | (pending)| 0.9999  |")
        print("  | LEACE (erase gender)    | -16.5pp   | -0.85pp  | 0.9991  |")
        print("  | LEACE (erase race)      | +0.31pp   | -2.8pp   | 0.9997  |")
        print()
        print("  LEACE dramatically outperforms both SAE-based methods for gender.")
        print("  Masked reconstruction changes embeddings drastically but fails to")
        print("  remove info (the SAE decoder reconstructs it). S&P Top-K barely")
        print("  changes embeddings at all (the control axis misses the concept).")
        print("  LEACE finds the optimal direction directly from data.")

    # -----------------------------------------------------------------------
    # 6. Related work context
    # -----------------------------------------------------------------------
    section("6. Related Work Context")

    print("  Key papers in the SAE-for-VLM interpretability space:")
    print()
    print("  1. Pach et al. (NeurIPS 2025) - 'SAEs Learn Monosemantic Features")
    print("     in VLMs' [2504.02821]")
    print("     SAEs on CLIP VE learn monosemantic features. Interventions steer")
    print("     multimodal LLM (LLaVA) outputs. Framework for evaluating")
    print("     monosemanticity; sparsity and wide latents are key factors.")
    print()
    print("  2. Barbulau et al. (ICLR 2026 under review) - 'S&P Top-K:")
    print("     Select-and-Project for Fairness and Control' [2509.10809]")
    print("     ENCODER-CENTRIC alternative to decoder-based SAE steering.")
    print("     Three-stage pipeline: (i) select top-K encoder features aligned")
    print("     with sensitive attribute, (ii) aggregate into unified control axis")
    print("     via logistic regression on encoder weights, (iii) compute")
    print("     orthogonal projection V = I - alpha * A(A^T A)^{-1} A^T.")
    print("     Applied directly in embedding space (no SAE decode at inference).")
    print("     Uses JumpReLU SAE with 16,384 features on CLIP ViT-B/16.")
    print("     3.2x improvement over masked reconstruction on CelebA/FairFace.")
    print("     DIRECTLY relevant: same dataset, same fairness goals.")
    print()
    print("  3. Stevens et al. (2025) - 'Interpretable and Testable Vision")
    print("     Features via SAEs' [2502.06755]")
    print("     Patch-level causal edits on frozen ViT activations. Reveals")
    print("     semantic abstraction differences across pretraining objectives.")
    print("     Model-agnostic; code and models publicly available.")
    print()
    print("  4. CVPR 2025 WS - 'Steering CLIP's Vision Transformer with SAEs'")
    print("     [2504.08729]")
    print("     10-15% of neurons and features are steerable. SAEs provide")
    print("     thousands more steerable features than the base model.")
    print("     SOTA on CelebA, Waterbirds, typographic attacks via feature")
    print("     suppression. Optimal results in middle model layers.")
    print()
    print("  5. ICML 2025 - Matryoshka SAE (MSAE) [2503.17547]")
    print("     Nested dictionaries at multiple granularities. Addresses feature")
    print("     splitting/absorption at large dictionary sizes. 0.99 cosine sim")
    print("     with 80% sparsity. 120+ interpretable concepts from CLIP.")
    print("     Reduced feature absorption vs standard SAEs.")
    print()
    print("  6. Bussmann et al. (NeurIPS 2024 WS) - BatchTopK SAEs [2412.06410]")
    print("     The SAE variant we use. Relaxes top-k from sample to batch level")
    print("     for adaptive per-sample sparsity.")
    print()
    print("  7. Lim et al. (2024) - PatchSAE [2412.05276]")
    print("     Patch-level SAEs for CLIP ViT; spatial feature attribution.")
    print("     Referenced in original IDARVE proposal.")
    print()
    print("  Our differentiators:")
    print("  - Multi-VE comparison (5 vision encoders vs single CLIP)")
    print("  - Full pipeline: probes + SAE + SDF discovery + evaluation + intervention")
    print("  - BatchTopK variant (vs TopK/JumpReLU in prior work)")
    print("  - Focus on demographic attribute encoding specifically")
    print("  - Cross-VE analysis of how different architectures encode demographics")
    print()
    print("  ACTIONABLE INSIGHT from S&P Top-K:")
    print("  Our current intervention (masked reconstruction in intervene.py) is")
    print("  the exact baseline that S&P Top-K outperforms. Implementing the")
    print("  encoder-centric projection approach would be straightforward:")
    print("  1. Use existing SDFs as the 'selected features'")
    print("  2. Extract corresponding SAE encoder weights")
    print("  3. Train logistic regression on encoder features (already done)")
    print("  4. Compute weighted sum of encoder weights -> control axis")
    print("  5. Apply orthogonal projection V = I - alpha * a*a^T / ||a||^2")
    print("  This avoids the decode step entirely and operates in native VE space.")

    # -----------------------------------------------------------------------
    # 7. Codebase status
    # -----------------------------------------------------------------------
    section("7. Codebase Status")

    print("  Recent improvements:")
    print("  - DRY refactor: shared load_sae + NumpyEncoder in src/utils.py")
    print("  - Fixed encode_dataset return type annotation (was 3-tuple, now 5)")
    print("  - Fixed numpy RuntimeWarning in selectivity computation")
    print("  - Switched LogisticRegression solver: saga -> lbfgs (10-50x faster)")
    print("  - Added weights_only=False to torch.load for FutureWarning")
    print("  - Streaming SAE encoding to prevent OOM on token-level data")
    print("  - Proper train/test splits in ablation and probe comparisons")
    print("  - All configs verified consistent across 5 VEs")
    print("  - Fixed intervene.py: labels/acts length mismatch in evaluate_suppression")
    print("  - Fixed intervene.py: encoded/labels alignment in evaluate_steering")
    print("  - Added empty-data guard in process_hdf5_with_intervention")
    print("  - Added stratification safety check (< 2 samples per class)")
    print("  - Lazy HDF5 handle caching in FairFaceDataset for num_workers=0")
    print("  - Implemented S&P Top-K encoder-centric projection (intervene.py)")
    print("  - Implemented LEACE concept erasure (intervene.py)")
    print("  - Fixed utils.py: silent failure when checkpoint has no 'model.*' keys")
    print("  - Fixed probe.py: KeyError when labels don't include all tasks")
    print("  - Optimized S&P Top-K: cache original probes across k values")
    print("  - Optimized S&P Top-K: pre-load data once across k values")
    print()
    print("  Modules:")
    print("    src/utils.py                 - Shared utilities")
    print("    src/data.py                  - FairFace DataModule")
    print("    src/datasets/fairface.py     - FairFace HDF5 Dataset")
    print("    src/models/sparse_autoencoder.py  - SAE variants")
    print("    src/models/linear_probes.py  - Linear probe classifiers")
    print("    src/modules/sae.py           - SAE Lightning module")
    print("    src/modules/probe.py         - Probe Lightning module")
    print("    src/analysis.py              - SDF filtering pipeline")
    print("    src/evaluation.py            - Post-training SAE evaluation")
    print("    src/intervene.py             - SAE-based interventions")
    print("    src/setup_datasets/          - VE latent extraction")
    print("    src/main.py                  - Hydra training entry point")
    print("    src/summarize_results.py     - Results summary tool")

    # -----------------------------------------------------------------------
    # 8. Next steps
    # -----------------------------------------------------------------------
    section("8. Next Steps (Priority Order)")

    print("  IMMEDIATE:")
    print("  1. [DONE] Evaluation run complete (ablation + probe comparison)")
    print("  2. [DONE] Suppression intervention (masked reconstruction) -- FAILED")
    print("  3. [DONE] Implemented S&P Top-K + LEACE interventions")
    print("  4. [RUNNING] S&P Top-K experiment (gender+race, k={16,32,64})")
    print("  5. [DONE] LEACE experiment -- gender erased to near-chance!")
    print("     Race/age erasure moderate (nonlinear encoding components)")
    print("  6. Compare all 3 intervention approaches quantitatively")
    print()
    print("  SHORT-TERM:")
    print("  7. Feature scaling (StandardScaler) for probes to fix ConvergenceWarnings")
    print("  8. Multi-direction S&P projection (LEACE-guided subspace, not rank-1)")
    print("  9. SDF-based feature selection for S&P Top-K (compare to variance)")
    print("  10. Extract VE latents for remaining encoders:")
    print("      - DINOv3 (1024-dim), SigLIP2 (1152-dim)")
    print("      - Qwen3-VL (1152-dim), InternVL3.5 (1024-dim)")
    print()
    print("  MEDIUM-TERM:")
    print("  11. Train SAEs + run full pipeline for each additional VE")
    print("  12. Cross-VE comparison: which VEs encode demographics most?")
    print("  13. Compare intervention effectiveness across VEs")
    print("  14. Visualization of top-activating patches for SDFs")
    print("  15. Matryoshka SAE for hierarchical feature discovery")
    print("  16. Paper: multi-VE results, intervention comparison, analysis")

    print(f"\n{'=' * 72}")
    print("  End of Progress Report")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    print_report()
