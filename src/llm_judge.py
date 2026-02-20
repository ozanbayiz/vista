#!/usr/bin/env python3
"""LLM-as-judge evaluation for demographic content in VLM captions.

Evaluates whether captions mention demographic attributes using an LLM,
providing a more semantically robust alternative to keyword-based DCR
metrics.

Supports two evaluation paradigms:

* **binary** -- Per-caption yes/no classification of demographic mention.
  Computes LLM-DCR = fraction of "yes" verdicts.  Directly comparable to
  keyword-DCR.
* **pairwise** -- Side-by-side comparison of original vs. modified caption
  asking which contains more demographic content.  Validates the *direction*
  of the intervention effect and is robust to absolute calibration.

Usage:
    # Binary DCR with Qwen3-8B (use qwen3 venv for transformers>=5.0)
    python -m src.llm_judge \\
        --model-path /scratch/current/ozanbayiz/models/Qwen3-8B \\
        --caption-files results/*.json \\
        --output-dir results/llm_judge \\
        --mode binary --batch-size 32

    # Pairwise comparison with JudgeLRM-7B
    python -m src.llm_judge \\
        --model-path /scratch/current/ozanbayiz/models/JudgeLRM-7B \\
        --caption-files results/*.json \\
        --output-dir results/llm_judge \\
        --mode pairwise --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute descriptions for prompts
# ---------------------------------------------------------------------------

ATTRIBUTE_DESCRIPTIONS: dict[str, str] = {
    "race": (
        "race, ethnicity, or skin color (e.g., Black, White, Asian, "
        "Hispanic, dark-skinned, light-skinned, African, Caucasian)"
    ),
    "gender": (
        "gender or sex (e.g., man, woman, male, female, boy, girl, "
        "he, she, his, her, gentleman, lady)"
    ),
    "age": (
        "age or age-related characteristics (e.g., young, old, elderly, "
        "child, teenager, middle-aged, baby, or specific age numbers)"
    ),
}


# ---------------------------------------------------------------------------
# Keyword-based DCR  (mirrors src/causal_eval.py for side-by-side comparison)
# ---------------------------------------------------------------------------

_RACE_TERMS: list[str] = [
    "white", "black", "asian", "hispanic", "latino", "latina",
    "indian", "middle eastern", "arab", "african",
    "caucasian", "east asian", "southeast asian", "south asian",
    "european", "pacific islander",
]

_GENDER_TERMS: list[str] = [
    "man", "woman", "male", "female", "boy", "girl",
    "he", "she", "his", "her", "gentleman", "lady",
]

_AGE_TERMS: list[str] = [
    "young", "old", "elderly", "teenager", "teen", "child",
    "baby", "toddler", "infant", "middle-aged", "middle aged",
    "senior", "adult", "kid", "adolescent",
    "twenties", "thirties", "forties", "fifties",
    "sixties", "seventies", "eighties",
]

_TERM_LISTS: dict[str, list[str]] = {
    "race": _RACE_TERMS,
    "gender": _GENDER_TERMS,
    "age": _AGE_TERMS,
}

# Pre-compiled patterns
_KW_PATTERNS: dict[str, re.Pattern[str]] = {}
for _attr, _terms in _TERM_LISTS.items():
    _escaped = [re.escape(t) for t in _terms]
    _KW_PATTERNS[_attr] = re.compile(
        r"\b(" + "|".join(_escaped) + r")\b", re.IGNORECASE,
    )


def keyword_detect(text: str, attribute: str) -> bool:
    """Return whether *text* contains demographic keywords for *attribute*."""
    return bool(_KW_PATTERNS[attribute].search(text))


def compute_keyword_dcr(
    samples: list[dict], attributes: list[str],
) -> dict[str, dict[str, float]]:
    """Compute keyword-based DCR (original, modified, delta) per attribute."""
    kw_dcr: dict[str, dict[str, float]] = {}
    n = len(samples)
    for attr in attributes:
        orig_hits = sum(1 for s in samples if keyword_detect(s["original_text"], attr))
        mod_hits = sum(1 for s in samples if keyword_detect(s["modified_text"], attr))
        kw_dcr[attr] = {
            "dcr_original": orig_hits / max(n, 1),
            "dcr_modified": mod_hits / max(n, 1),
            "dcr_delta": (mod_hits - orig_hits) / max(n, 1),
        }
    return kw_dcr


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def make_binary_prompt(caption: str, attribute: str) -> str:
    """Create a binary yes/no prompt for demographic content detection."""
    desc = ATTRIBUTE_DESCRIPTIONS[attribute]
    return (
        f"Read the following image caption carefully.\n\n"
        f'Caption: "{caption}"\n\n'
        f"Question: Does this caption explicitly mention or describe "
        f"the person's {desc}?\n\n"
        f"Answer with exactly one word: Yes or No"
    )


def make_pairwise_prompt(
    caption_a: str, caption_b: str, attribute: str,
) -> str:
    """Create a pairwise comparison prompt for two captions."""
    desc = ATTRIBUTE_DESCRIPTIONS[attribute]
    return (
        f"You are evaluating two image captions for demographic content.\n\n"
        f'Caption A: "{caption_a}"\n'
        f'Caption B: "{caption_b}"\n\n'
        f"Question: Which caption contains more explicit references "
        f"to the person's {desc}?\n\n"
        f"If Caption A has more, answer A.\n"
        f"If Caption B has more, answer B.\n"
        f"If roughly equal, answer tie.\n\n"
        f"Answer with exactly one word: A, B, or tie"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_binary(response: str) -> bool | None:
    """Parse yes/no.  Returns True (yes), False (no), or None (ambiguous)."""
    # Strip Qwen3 thinking tags if present
    if "</think>" in response:
        response = response.split("</think>")[-1]
    cleaned = response.strip().lower()
    # Take the first token
    first = cleaned.split()[0].rstrip(".,!;:") if cleaned.split() else ""
    if first == "yes":
        return True
    if first == "no":
        return False
    # Fallback: search entire response
    has_yes = "yes" in cleaned
    has_no = "no" in cleaned
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None  # truly ambiguous


def parse_pairwise(response: str) -> str | None:
    """Parse A / B / tie.  Returns canonical string or None."""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    cleaned = response.strip().lower()
    first = cleaned.split()[0].rstrip(".,!;:") if cleaned.split() else ""
    if first == "a":
        return "A"
    if first == "b":
        return "B"
    if first in ("tie", "equal", "neither", "same"):
        return "tie"
    return None


# ---------------------------------------------------------------------------
# Model loading & batched generation
# ---------------------------------------------------------------------------


def load_model(
    model_path: str, device: str = "cuda",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load an HF causal-LM and its tokenizer."""
    log.info("Loading tokenizer from %s …", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading model from %s …", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded on %s", device)
    return model, tokenizer


def format_chat(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Wrap *prompt* in the model's chat template."""
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    # Qwen3 supports disabling its thinking mode
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.no_grad()
def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 10,
) -> list[str]:
    """Greedy-decode a batch of formatted prompt strings."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the generated portion
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------


def evaluate_binary(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict],
    attributes: list[str],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Binary yes/no DCR evaluation over all (sample, text_type, attribute)."""
    # Pre-format every prompt
    tasks: list[tuple[str, str, int, str]] = []  # (attr, text_type, idx, formatted)
    for attr in attributes:
        for text_type in ("original", "modified"):
            key = f"{text_type}_text"
            for i, sample in enumerate(samples):
                raw_prompt = make_binary_prompt(sample[key], attr)
                tasks.append((attr, text_type, i, format_chat(tokenizer, raw_prompt)))

    log.info("Binary prompts: %d  (batch_size=%d)", len(tasks), batch_size)

    responses: list[str] = []
    for start in tqdm(range(0, len(tasks), batch_size), desc="Binary eval"):
        batch = tasks[start : start + batch_size]
        responses.extend(batch_generate(model, tokenizer, [t[3] for t in batch]))

    # Parse
    verdicts: dict[str, dict[str, list[bool]]] = {
        attr: {"original": [], "modified": []} for attr in attributes
    }
    parse_failures = 0
    for (attr, text_type, _, _), resp in zip(tasks, responses):
        v = parse_binary(resp)
        if v is None:
            parse_failures += 1
            log.debug("Parse failure: %r", resp)
            v = False  # conservative default
        verdicts[attr][text_type].append(v)

    # Aggregate LLM-DCR
    llm_dcr: dict[str, dict[str, float]] = {}
    per_sample: dict[str, dict[str, list[bool]]] = {}
    for attr in attributes:
        o = verdicts[attr]["original"]
        m = verdicts[attr]["modified"]
        orig_rate = sum(o) / max(len(o), 1)
        mod_rate = sum(m) / max(len(m), 1)
        llm_dcr[attr] = {
            "dcr_original": round(orig_rate, 4),
            "dcr_modified": round(mod_rate, 4),
            "dcr_delta": round(mod_rate - orig_rate, 4),
        }
        per_sample[attr] = {"original": o, "modified": m}

    return {
        "llm_dcr": llm_dcr,
        "per_sample_verdicts": per_sample,
        "parse_failures": parse_failures,
        "total_prompts": len(tasks),
    }


def evaluate_pairwise(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict],
    attributes: list[str],
    batch_size: int = 16,
) -> dict[str, Any]:
    """Pairwise comparison: which caption has more demographic content?"""
    import random

    rng = random.Random(42)

    tasks: list[tuple[str, int, bool, str]] = []  # (attr, idx, orig_is_A, formatted)
    for attr in attributes:
        for i, sample in enumerate(samples):
            orig_is_A = rng.random() > 0.5
            if orig_is_A:
                cap_a, cap_b = sample["original_text"], sample["modified_text"]
            else:
                cap_a, cap_b = sample["modified_text"], sample["original_text"]
            raw = make_pairwise_prompt(cap_a, cap_b, attr)
            tasks.append((attr, i, orig_is_A, format_chat(tokenizer, raw)))

    log.info("Pairwise prompts: %d  (batch_size=%d)", len(tasks), batch_size)

    responses: list[str] = []
    for start in tqdm(range(0, len(tasks), batch_size), desc="Pairwise eval"):
        batch = tasks[start : start + batch_size]
        responses.extend(batch_generate(model, tokenizer, [t[3] for t in batch]))

    # Parse and un-randomise
    winners: dict[str, list[str]] = {attr: [] for attr in attributes}
    parse_failures = 0
    for (attr, _, orig_is_A, _), resp in zip(tasks, responses):
        v = parse_pairwise(resp)
        if v is None:
            parse_failures += 1
            v = "tie"
        # Map positional label back to semantic label
        if v == "A":
            winners[attr].append("original" if orig_is_A else "modified")
        elif v == "B":
            winners[attr].append("modified" if orig_is_A else "original")
        else:
            winners[attr].append("tie")

    # Aggregate
    pairwise: dict[str, dict[str, Any]] = {}
    per_sample_winners: dict[str, list[str]] = {}
    for attr in attributes:
        ws = winners[attr]
        n = len(ws)
        pairwise[attr] = {
            "original_more": round(ws.count("original") / max(n, 1), 4),
            "modified_more": round(ws.count("modified") / max(n, 1), 4),
            "tie": round(ws.count("tie") / max(n, 1), 4),
            "n_comparisons": n,
        }
        per_sample_winners[attr] = ws

    return {
        "pairwise": pairwise,
        "per_sample_winners": per_sample_winners,
        "parse_failures": parse_failures,
        "total_prompts": len(tasks),
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def load_caption_file(path: str) -> tuple[list[dict], dict]:
    """Load a caption JSON file produced by ``src.intervene``."""
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", data.get("results", []))
    meta = {**data.get("_metadata", {}), **data.get("config", {})}
    return samples, meta


def infer_condition(filepath: str) -> tuple[str, str, str]:
    """Infer (vlm, target_attribute, intervention_label) from file path.

    Returns a clean condition key suitable for logging and result dicts.
    """
    path = Path(filepath)
    name = path.stem  # e.g. causal_caption_race_suppression

    # VLM
    full = str(path)
    if "paligemma" in full:
        vlm = "paligemma2"
    elif "qwen3vl" in full:
        vlm = "qwen3vl"
    elif "qwen2vl" in full:
        vlm = "qwen2vl"
    else:
        vlm = "unknown"

    # Optional subdirectory qualifier (quality_topk16, control_noise, …)
    parent = path.parent.name
    subdir_prefix = "" if parent.startswith("causal_") else f"{parent}/"

    # Parse attribute and mode from filename
    stripped = name.replace("causal_caption_", "")
    parts = stripped.split("_", 1)
    attribute = parts[0] if parts else "unknown"
    mode = parts[1] if len(parts) > 1 else "unknown"

    intervention = f"{subdir_prefix}{mode}"
    return vlm, attribute, intervention


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="LLM-as-judge caption evaluation")
    ap.add_argument("--model-path", required=True, help="HF model directory")
    ap.add_argument(
        "--caption-files", nargs="+", required=True,
        help="Caption JSON files to evaluate",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["binary", "pairwise"], default="binary")
    ap.add_argument(
        "--attributes", nargs="+", default=["race", "gender", "age"],
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap samples per file (for quick smoke tests)",
    )
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model_path, device=device)
    model_name = Path(args.model_path).name

    all_results: dict[str, Any] = {
        "_metadata": {
            "model": model_name,
            "mode": args.mode,
            "attributes": args.attributes,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "conditions": {},
    }

    for caption_file in args.caption_files:
        log.info("=" * 60)
        log.info("Processing: %s", caption_file)

        samples, _file_meta = load_caption_file(caption_file)
        if args.max_samples:
            samples = samples[: args.max_samples]

        vlm, target_attr, intervention = infer_condition(caption_file)
        cond_key = f"{vlm}/{target_attr}_{intervention}"
        log.info("Condition: %s  (%d samples)", cond_key, len(samples))

        # -- Keyword DCR (for comparison) --
        kw_dcr = compute_keyword_dcr(samples, args.attributes)
        log.info(
            "Keyword DCR Δ: %s",
            {a: f"{v['dcr_delta']:+.3f}" for a, v in kw_dcr.items()},
        )

        # -- LLM evaluation --
        t0 = time.time()
        if args.mode == "binary":
            eval_result = evaluate_binary(
                model, tokenizer, samples, args.attributes, args.batch_size,
            )
        else:
            eval_result = evaluate_pairwise(
                model, tokenizer, samples, args.attributes, args.batch_size,
            )
        elapsed = time.time() - t0

        # -- Log summary --
        if args.mode == "binary":
            log.info(
                "LLM  DCR Δ: %s",
                {a: f"{v['dcr_delta']:+.3f}" for a, v in eval_result["llm_dcr"].items()},
            )
        else:
            for attr, pw in eval_result["pairwise"].items():
                log.info(
                    "  %s: orig_more=%.2f mod_more=%.2f tie=%.2f",
                    attr, pw["original_more"], pw["modified_more"], pw["tie"],
                )

        log.info(
            "Parse failures: %d / %d  (%.1f%%)",
            eval_result["parse_failures"],
            eval_result["total_prompts"],
            100 * eval_result["parse_failures"] / max(eval_result["total_prompts"], 1),
        )
        log.info("Elapsed: %.1fs", elapsed)

        all_results["conditions"][cond_key] = {
            "source_file": str(caption_file),
            "vlm": vlm,
            "target_attribute": target_attr,
            "intervention": intervention,
            "n_samples": len(samples),
            "keyword_dcr": kw_dcr,
            "elapsed_seconds": round(elapsed, 1),
            **eval_result,
        }

    # -- Save --
    out_file = out_dir / f"llm_judge_{args.mode}_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_file)

    # -- Pretty comparison table --
    print("\n" + "=" * 90)
    print(f"  LLM-AS-JUDGE RESULTS  |  model={model_name}  mode={args.mode}")
    print("=" * 90)

    if args.mode == "binary":
        hdr = (
            f"{'Condition':<40} {'Attr':<7} "
            f"{'KW-Δ':>7} {'LLM-Δ':>7} {'KW-orig':>8} {'LLM-orig':>9} "
            f"{'KW-mod':>7} {'LLM-mod':>8}  {'Match':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        for cond_key, cond in all_results["conditions"].items():
            for attr in args.attributes:
                kw = cond["keyword_dcr"][attr]
                llm = cond["llm_dcr"][attr]
                # Direction agreement
                kd, ld = kw["dcr_delta"], llm["dcr_delta"]
                if abs(kd) < 0.01 and abs(ld) < 0.01:
                    match = "~0"
                elif (kd > 0) == (ld > 0):
                    match = "OK"
                else:
                    match = "DIFF"
                star = "*" if attr == cond["target_attribute"] else " "
                print(
                    f"  {cond_key:<38}{star} {attr:<7} "
                    f"{kd:>+7.3f} {ld:>+7.3f} "
                    f"{kw['dcr_original']:>8.3f} {llm['dcr_original']:>9.3f} "
                    f"{kw['dcr_modified']:>7.3f} {llm['dcr_modified']:>8.3f}  "
                    f"{match:>6}"
                )
    else:
        hdr = (
            f"{'Condition':<40} {'Attr':<7} "
            f"{'Orig>':>6} {'Mod>':>6} {'Tie':>6}  {'KW-Δ':>7}"
        )
        print(hdr)
        print("-" * len(hdr))
        for cond_key, cond in all_results["conditions"].items():
            for attr in args.attributes:
                pw = cond["pairwise"][attr]
                kd = cond["keyword_dcr"][attr]["dcr_delta"]
                star = "*" if attr == cond["target_attribute"] else " "
                print(
                    f"  {cond_key:<38}{star} {attr:<7} "
                    f"{pw['original_more']:>6.2f} {pw['modified_more']:>6.2f} "
                    f"{pw['tie']:>6.2f}  {kd:>+7.3f}"
                )

    print()


if __name__ == "__main__":
    main()
