#!/usr/bin/env python3
"""Mechanistic test: VLM captioning with blank (no-information) images.

Tests whether the VLM's language model defaults to gendered or neutral
phrasing when the vision encoder provides no meaningful content.  If
blank-image gender DCR > real-image gender DCR, vision features actively
suppress demographic language, and removing them (via SDF suppression)
releases the LM's gendered prior.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m src.mechanistic_test \
        --vlm qwen2vl --n-samples 200 \
        --output /scratch/current/ozanbayiz/results/mechanistic_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword DCR (same lists as causal_eval.py)
# ---------------------------------------------------------------------------

_GENDER_TERMS = [
    "man", "woman", "male", "female", "boy", "girl",
    "he", "she", "his", "her", "gentleman", "lady",
]
_RACE_TERMS = [
    "white", "black", "asian", "hispanic", "latino", "latina",
    "indian", "middle eastern", "arab", "african",
    "caucasian", "east asian", "southeast asian", "south asian",
    "european", "pacific islander",
]
_AGE_TERMS = [
    "young", "old", "elderly", "teenager", "teen", "child",
    "baby", "toddler", "infant", "middle-aged", "middle aged",
    "senior", "adult", "kid", "adolescent",
    "twenties", "thirties", "forties", "fifties",
    "sixties", "seventies", "eighties",
]

_PATTERNS: dict[str, re.Pattern[str]] = {}
for _attr, _terms in {"race": _RACE_TERMS, "gender": _GENDER_TERMS, "age": _AGE_TERMS}.items():
    _PATTERNS[_attr] = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in _terms) + r")\b",
        re.IGNORECASE,
    )


def keyword_dcr(texts: list[str], attribute: str) -> float:
    return sum(1 for t in texts if _PATTERNS[attribute].search(t)) / max(len(texts), 1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="Mechanistic blank-image test")
    ap.add_argument("--vlm", default="qwen2vl", choices=["paligemma2", "qwen2vl"])
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VLM
    from src.vlm_generate import FULL_VLM_ADAPTERS
    adapter = FULL_VLM_ADAPTERS[args.vlm]()
    adapter.load(device)

    # Create blank images in different conditions
    conditions = {
        "black": Image.new("RGB", (448, 448), (0, 0, 0)),
        "white": Image.new("RGB", (448, 448), (255, 255, 255)),
        "gray": Image.new("RGB", (448, 448), (128, 128, 128)),
    }

    results: dict = {
        "_metadata": {
            "vlm": args.vlm,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "conditions": {},
    }

    caption_prompt = "caption en" if args.vlm == "paligemma2" else "Describe this image."

    for cond_name, blank_img in conditions.items():
        log.info("=== Condition: %s ===", cond_name)
        captions: list[str] = []

        for i in tqdm(range(args.n_samples), desc=cond_name):
            texts = adapter.generate(
                images=[blank_img],
                prompts=[caption_prompt],
                max_new_tokens=args.max_new_tokens,
            )
            captions.append(texts[0])

        # Compute DCR
        dcr = {attr: keyword_dcr(captions, attr) for attr in ["race", "gender", "age"]}
        log.info("  DCR: %s", {a: f"{v:.3f}" for a, v in dcr.items()})

        # Sample captions for qualitative analysis
        results["conditions"][cond_name] = {
            "dcr": dcr,
            "sample_captions": captions[:20],
            "all_captions": captions,
        }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Summary
    print("\n" + "=" * 70)
    print(f"MECHANISTIC TEST: {args.vlm} blank-image captioning")
    print("=" * 70)
    print(f"{'Condition':<12} {'Race DCR':>10} {'Gender DCR':>12} {'Age DCR':>10}")
    print("-" * 44)
    for cond_name, cond in results["conditions"].items():
        d = cond["dcr"]
        print(f"  {cond_name:<10} {d['race']:>10.3f} {d['gender']:>12.3f} {d['age']:>10.3f}")
    print()


if __name__ == "__main__":
    main()
