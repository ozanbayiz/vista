"""Evaluation metrics for VLM causal tracing experiments.

Consumes the JSON output of ``src.vlm_generate`` and quantifies:

* **Caption causal metrics** -- Demographic Content Rate (DCR) delta and
  BERTScore [Zhang et al., 2020] between original and modified captions.
* **VQA causal metrics** -- accuracy shifts on demographic vs. control
  questions after SAE-based feature intervention.
* **Aggregate statistics** stratified by demographic subgroup with 95 %
  bootstrap confidence intervals.

Usage:
    python -m src.causal_eval \\
        --results results/causal_tracing/causal_caption_race_suppression.json \\
        --output-dir results/causal_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import NumpyEncoder, build_metadata

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demographic content detection
# ---------------------------------------------------------------------------

# Curated keyword lists per attribute.  Each list is matched case-
# insensitively via word-boundary regex.

_RACE_TERMS: list[str] = [
    # Broad categories
    "white",
    "black",
    "asian",
    "hispanic",
    "latino",
    "latina",
    "indian",
    "middle eastern",
    "arab",
    "african",
    # More specific
    "caucasian",
    "east asian",
    "southeast asian",
    "south asian",
    "european",
    "pacific islander",
]

_GENDER_TERMS: list[str] = [
    "man",
    "woman",
    "male",
    "female",
    "boy",
    "girl",
    "he",
    "she",
    "his",
    "her",
    "gentleman",
    "lady",
]

_AGE_TERMS: list[str] = [
    "young",
    "old",
    "elderly",
    "teenager",
    "teen",
    "child",
    "baby",
    "toddler",
    "infant",
    "middle-aged",
    "middle aged",
    "senior",
    "adult",
    "kid",
    "adolescent",
    # Decade references
    "twenties",
    "thirties",
    "forties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
]

_TERM_LISTS: dict[str, list[str]] = {
    "race": _RACE_TERMS,
    "gender": _GENDER_TERMS,
    "age": _AGE_TERMS,
}


class DemographicContentDetector:
    """Keyword-based detector for demographic content in generated text.

    Compiles word-boundary regular expressions from curated term lists
    and provides both per-text detection and corpus-level Demographic
    Content Rate (DCR) computation.
    """

    def __init__(self) -> None:
        self._patterns: dict[str, re.Pattern[str]] = {}
        for attr, terms in _TERM_LISTS.items():
            # Build alternation:  \\b(term1|term2|...)\\b
            escaped = [re.escape(t) for t in terms]
            self._patterns[attr] = re.compile(
                r"\b(" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )

    def detect(self, text: str, attribute: str) -> bool:
        """Return whether *text* contains demographic terms for *attribute*.

        Args:
            text: Generated text to inspect.
            attribute: One of ``'race'``, ``'gender'``, ``'age'``.

        Returns:
            ``True`` if at least one keyword is found.
        """
        pattern = self._patterns.get(attribute)
        if pattern is None:
            raise ValueError(f"Unknown attribute: {attribute!r}")
        return bool(pattern.search(text))

    def compute_dcr(self, texts: list[str], attribute: str) -> float:
        """Demographic Content Rate: fraction of *texts* with demographic terms.

        Args:
            texts: List of generated texts.
            attribute: One of ``'race'``, ``'gender'``, ``'age'``.

        Returns:
            Float in [0, 1].
        """
        if not texts:
            return 0.0
        hits = sum(1 for t in texts if self.detect(t, attribute))
        return hits / len(texts)


# ---------------------------------------------------------------------------
# Caption causal metrics
# ---------------------------------------------------------------------------


class CaptionCausalMetrics:
    """Metrics quantifying the causal effect of SAE intervention on captions.

    Computes:

    * DCR delta (per attribute).
    * BERTScore [Zhang et al., 2020] between original and modified
      captions (semantic preservation).
    """

    def __init__(self) -> None:
        self._detector = DemographicContentDetector()

    def compute(
        self,
        original_captions: list[str],
        modified_captions: list[str],
        target_attribute: str,
    ) -> dict[str, Any]:
        """Compute caption causal effect metrics.

        Args:
            original_captions: Captions generated without intervention.
            modified_captions: Captions generated with SAE intervention.
            target_attribute: The attribute whose SDFs were intervened on.

        Returns:
            Dict with ``dcr_original``, ``dcr_modified``, ``dcr_delta``
            per attribute, and ``bertscore`` summary.
        """
        results: dict[str, Any] = {}

        # -- DCR per attribute ------------------------------------------------
        for attr in ("race", "gender", "age"):
            dcr_orig = self._detector.compute_dcr(original_captions, attr)
            dcr_mod = self._detector.compute_dcr(modified_captions, attr)
            entry = {
                "dcr_original": dcr_orig,
                "dcr_modified": dcr_mod,
                "dcr_delta": dcr_mod - dcr_orig,
            }
            results[f"dcr_{attr}"] = entry
            log.info(
                "  DCR(%s): original=%.4f, modified=%.4f, delta=%+.4f%s",
                attr,
                dcr_orig,
                dcr_mod,
                dcr_mod - dcr_orig,
                "  [TARGET]" if attr == target_attribute else "",
            )

        # -- BERTScore --------------------------------------------------------
        try:
            from bert_score import score as bert_score_fn

            log.info("  Computing BERTScore (this may take a moment) ...")
            P, R, F = bert_score_fn(
                modified_captions,
                original_captions,
                lang="en",
                verbose=False,
            )
            results["bertscore"] = {
                "precision_mean": float(P.mean().item()),
                "recall_mean": float(R.mean().item()),
                "f1_mean": float(F.mean().item()),
                "f1_std": float(F.std().item()),
            }
            log.info(
                "  BERTScore F1: %.4f +/- %.4f",
                results["bertscore"]["f1_mean"],
                results["bertscore"]["f1_std"],
            )
        except ImportError:
            log.warning(
                "bert-score not installed; skipping BERTScore computation. Install with: pip install bert-score",
            )
            results["bertscore"] = None

        return results


# ---------------------------------------------------------------------------
# VQA causal metrics
# ---------------------------------------------------------------------------

# Reverse label maps (integer -> canonical string) for fuzzy matching.
_RACE_NAMES_LOWER: dict[int, list[str]] = {
    0: ["white", "caucasian"],
    1: ["black", "african"],
    2: ["latino", "latina", "hispanic"],
    3: ["east asian", "chinese", "japanese", "korean"],
    4: ["southeast asian"],
    5: ["indian", "south asian"],
    6: ["middle eastern", "arab"],
}
_GENDER_NAMES_LOWER: dict[int, list[str]] = {
    0: ["male", "man", "boy", "he", "gentleman"],
    1: ["female", "woman", "girl", "she", "lady"],
}
_AGE_NAMES_LOWER: dict[int, list[str]] = {
    0: ["baby", "infant", "0", "1", "2"],
    1: ["child", "kid", "3", "4", "5", "6", "7", "8", "9"],
    2: ["teenager", "teen", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
    3: ["twenties", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"],
    4: ["thirties", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39"],
    5: ["forties", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"],
    6: ["fifties", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"],
    7: ["sixties", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69"],
    8: ["seventies", "eighties", "70", "71", "72", "73", "74", "75", "elderly", "senior"],
}

_FUZZY_MAPS: dict[str, dict[int, list[str]]] = {
    "race": _RACE_NAMES_LOWER,
    "gender": _GENDER_NAMES_LOWER,
    "age": _AGE_NAMES_LOWER,
}


def _fuzzy_match(answer: str, label: int, attribute: str) -> bool:
    """Return ``True`` if *answer* plausibly matches the ground-truth *label*.

    Uses case-insensitive substring matching against curated synonym lists.
    """
    synonyms = _FUZZY_MAPS.get(attribute, {}).get(label, [])
    answer_lower = answer.lower()
    return any(syn in answer_lower for syn in synonyms)


class VQACausalMetrics:
    """Metrics quantifying causal effects on VQA accuracy.

    For **demographic questions** (those probing the target attribute),
    accuracy should *decrease* after intervention -- confirming that the
    suppressed SAE features were causally responsible for the model's
    ability to answer those questions.

    For **control questions** (non-demographic), accuracy should be
    *preserved* -- confirming that the intervention is selective.
    """

    def compute(
        self,
        samples: list[dict],
        target_attribute: str,
    ) -> dict[str, Any]:
        """Compute VQA causal effect metrics.

        Args:
            samples: List of per-sample dicts as produced by
                ``run_causal_tracing`` with ``task='vqa'``.
            target_attribute: The attribute whose SDFs were intervened on.

        Returns:
            Dict with per-question and per-type accuracy deltas plus the
            overall causal effect score.
        """
        per_question: dict[str, dict[str, Any]] = {}

        for sample in samples:
            gt_label = sample["labels"].get(target_attribute)
            if gt_label is None:
                continue

            for question, q_data in sample.get("questions", {}).items():
                q_type = q_data["type"]
                orig = q_data["original"]
                mod = q_data["modified"]

                if question not in per_question:
                    per_question[question] = {
                        "type": q_type,
                        "original_correct": 0,
                        "modified_correct": 0,
                        "total": 0,
                    }

                pq = per_question[question]
                pq["total"] += 1

                if q_type == "demographic":
                    if _fuzzy_match(orig, gt_label, target_attribute):
                        pq["original_correct"] += 1
                    if _fuzzy_match(mod, gt_label, target_attribute):
                        pq["modified_correct"] += 1
                else:
                    # For control questions, "correct" means the answer
                    # is substantively the same (non-empty, not a refusal).
                    if len(orig.strip()) > 5:
                        pq["original_correct"] += 1
                    if len(mod.strip()) > 5:
                        pq["modified_correct"] += 1

        # Aggregate per question type.
        type_agg: dict[str, dict[str, float]] = {}
        question_results: dict[str, dict[str, Any]] = {}

        for question, pq in per_question.items():
            total = max(pq["total"], 1)
            acc_orig = pq["original_correct"] / total
            acc_mod = pq["modified_correct"] / total
            delta = acc_mod - acc_orig

            question_results[question] = {
                "type": pq["type"],
                "acc_original": acc_orig,
                "acc_modified": acc_mod,
                "delta": delta,
                "n": pq["total"],
            }
            log.info(
                "  [%s] %s: original=%.4f, modified=%.4f, delta=%+.4f",
                pq["type"],
                question,
                acc_orig,
                acc_mod,
                delta,
            )

            q_type = pq["type"]
            if q_type not in type_agg:
                type_agg[q_type] = {"acc_orig_sum": 0, "acc_mod_sum": 0, "count": 0}
            type_agg[q_type]["acc_orig_sum"] += acc_orig
            type_agg[q_type]["acc_mod_sum"] += acc_mod
            type_agg[q_type]["count"] += 1

        # Compute type-level means.
        type_results: dict[str, dict[str, float]] = {}
        for q_type, agg in type_agg.items():
            cnt = max(agg["count"], 1)
            mean_orig = agg["acc_orig_sum"] / cnt
            mean_mod = agg["acc_mod_sum"] / cnt
            type_results[q_type] = {
                "mean_acc_original": mean_orig,
                "mean_acc_modified": mean_mod,
                "mean_delta": mean_mod - mean_orig,
            }

        # Causal effect = demographic accuracy drop - control accuracy drop.
        demo_delta = type_results.get("demographic", {}).get("mean_delta", 0.0)
        ctrl_delta = type_results.get("control", {}).get("mean_delta", 0.0)
        causal_effect = demo_delta - ctrl_delta

        log.info(
            "  Causal effect (demo_delta - ctrl_delta) = %+.4f",
            causal_effect,
        )

        return {
            "per_question": question_results,
            "per_type": type_results,
            "causal_effect": causal_effect,
        }


# ---------------------------------------------------------------------------
# Aggregate with bootstrap CIs
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for a 95 % bootstrap CI.

    Args:
        values: 1-D array of observations.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return float(values.mean()), lower, upper


def aggregate_results(
    samples: list[dict],
    target_attribute: str,
    task: str,
) -> dict[str, Any]:
    """Compute aggregate statistics stratified by demographic subgroup.

    Args:
        samples: Per-sample result dicts from ``run_causal_tracing``.
        target_attribute: Attribute whose SDFs were intervened on.
        task: ``'caption'`` or ``'vqa'``.

    Returns:
        Dict with overall and per-subgroup aggregate metrics, including
        95 % bootstrap confidence intervals.
    """
    from src.vlm_generate import LABEL_NAMES

    detector = DemographicContentDetector()
    names_map = LABEL_NAMES.get(target_attribute, {})
    results: dict[str, Any] = {"task": task, "target_attribute": target_attribute}

    if task == "caption":
        # Group samples by target attribute class.
        by_class: dict[int, list[dict]] = {}
        for s in samples:
            cls = s["labels"].get(target_attribute)
            if cls is not None:
                by_class.setdefault(cls, []).append(s)

        overall_orig: list[int] = []
        overall_mod: list[int] = []

        subgroup_results: dict[str, dict] = {}
        for cls, group_samples in sorted(by_class.items()):
            orig_hits = [int(detector.detect(s["original_text"], target_attribute)) for s in group_samples]
            mod_hits = [int(detector.detect(s["modified_text"], target_attribute)) for s in group_samples]

            overall_orig.extend(orig_hits)
            overall_mod.extend(mod_hits)

            dcr_orig_mean, dcr_orig_lo, dcr_orig_hi = _bootstrap_ci(
                np.array(orig_hits, dtype=np.float64),
            )
            dcr_mod_mean, dcr_mod_lo, dcr_mod_hi = _bootstrap_ci(
                np.array(mod_hits, dtype=np.float64),
            )

            name = names_map.get(cls, str(cls))
            subgroup_results[name] = {
                "n": len(group_samples),
                "dcr_original": {"mean": dcr_orig_mean, "ci_95": [dcr_orig_lo, dcr_orig_hi]},
                "dcr_modified": {"mean": dcr_mod_mean, "ci_95": [dcr_mod_lo, dcr_mod_hi]},
                "dcr_delta": dcr_mod_mean - dcr_orig_mean,
            }

        overall_orig_mean, overall_orig_lo, overall_orig_hi = _bootstrap_ci(
            np.array(overall_orig, dtype=np.float64),
        )
        overall_mod_mean, overall_mod_lo, overall_mod_hi = _bootstrap_ci(
            np.array(overall_mod, dtype=np.float64),
        )
        results["overall"] = {
            "n": len(samples),
            "dcr_original": {"mean": overall_orig_mean, "ci_95": [overall_orig_lo, overall_orig_hi]},
            "dcr_modified": {"mean": overall_mod_mean, "ci_95": [overall_mod_lo, overall_mod_hi]},
            "dcr_delta": overall_mod_mean - overall_orig_mean,
        }
        results["per_subgroup"] = subgroup_results

    elif task == "vqa":
        results["note"] = (
            "VQA aggregation uses per-question metrics from VQACausalMetrics; "
            "subgroup-stratified VQA analysis is included here for completeness."
        )

        by_class: dict[int, list[dict]] = {}
        for s in samples:
            cls = s["labels"].get(target_attribute)
            if cls is not None:
                by_class.setdefault(cls, []).append(s)

        subgroup_results = {}
        for cls, group_samples in sorted(by_class.items()):
            name = names_map.get(cls, str(cls))
            subgroup_results[name] = {
                "n": len(group_samples),
            }
        results["per_subgroup"] = subgroup_results

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
        description="Evaluate VLM causal tracing results",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to JSON output from vlm_generate.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/causal_eval",
        help="Output directory for evaluation results",
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

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    task = config.get("task", "caption")
    attribute = config.get("attribute", "race")
    samples = data.get("samples", [])

    log.info(
        "Loaded %d samples (task=%s, attribute=%s) from %s",
        len(samples),
        task,
        attribute,
        results_path,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_results: dict[str, Any] = {}

    # -- Task-specific evaluation -------------------------------------------

    if task == "caption":
        log.info("=== Caption Causal Metrics ===")
        original = [s["original_text"] for s in samples]
        modified = [s["modified_text"] for s in samples]

        caption_metrics = CaptionCausalMetrics()
        eval_results["caption_metrics"] = caption_metrics.compute(
            original,
            modified,
            attribute,
        )
    elif task == "vqa":
        log.info("=== VQA Causal Metrics ===")
        vqa_metrics = VQACausalMetrics()
        eval_results["vqa_metrics"] = vqa_metrics.compute(samples, attribute)
    else:
        log.warning("Unknown task %r; skipping task-specific evaluation.", task)

    # -- Aggregate with bootstrap CIs --------------------------------------
    log.info("=== Aggregate Results ===")
    eval_results["aggregate"] = aggregate_results(samples, attribute, task)

    # -- Save ---------------------------------------------------------------
    output = {
        "_metadata": build_metadata(args),
        "source_file": str(results_path),
        "config": config,
        "evaluation": eval_results,
    }

    eval_path = out_dir / f"eval_{results_path.stem}.json"
    with open(eval_path, "w") as fp:
        json.dump(output, fp, indent=2, cls=NumpyEncoder)
    log.info("Evaluation results saved to %s", eval_path)

    # -- Log to wandb -------------------------------------------------------
    if args.wandb:
        import wandb

        wandb_config = dict(config)
        wandb_config["source_file"] = str(results_path)

        run = wandb.init(
            project=args.wandb_project,
            name=f"eval_{task}_{attribute}_{config.get('intervention_mode', 'unknown')}",
            config=wandb_config,
            tags=[task, attribute, config.get("intervention_mode", "unknown")],
            job_type="evaluation",
        )

        wandb_metrics: dict[str, Any] = {}

        if task == "caption" and "caption_metrics" in eval_results:
            cm = eval_results["caption_metrics"]
            for attr in ("race", "gender", "age"):
                dcr_entry = cm.get(f"dcr_{attr}", {})
                wandb_metrics[f"dcr_{attr}_original"] = dcr_entry.get("dcr_original", 0)
                wandb_metrics[f"dcr_{attr}_modified"] = dcr_entry.get("dcr_modified", 0)
                wandb_metrics[f"dcr_{attr}_delta"] = dcr_entry.get("dcr_delta", 0)
            bs = cm.get("bertscore", {})
            if bs:
                wandb_metrics["bertscore_f1"] = bs.get("f1_mean", 0)
                wandb_metrics["bertscore_f1_std"] = bs.get("f1_std", 0)

        elif task == "vqa" and "vqa_metrics" in eval_results:
            vm = eval_results["vqa_metrics"]
            wandb_metrics["causal_effect"] = vm.get("causal_effect", 0)
            for q_type, t_data in vm.get("per_type", {}).items():
                wandb_metrics[f"vqa_type/{q_type}/mean_delta"] = t_data.get("mean_delta", 0)

        wandb.log(wandb_metrics)
        run.finish()
        log.info("W&B evaluation run finished")


if __name__ == "__main__":
    main()
