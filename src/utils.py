"""Shared utilities used by analysis, evaluation, and intervention scripts."""

from __future__ import annotations

import datetime
import json
import subprocess
from typing import Any

import numpy as np
import torch

from src.models.sparse_autoencoder import BaseSAE, build_sae


def load_sae_from_checkpoint(checkpoint_path: str, device: torch.device) -> BaseSAE:
    """Load SAE model from a Lightning checkpoint.

    Extracts hyperparameters and state dict from the checkpoint, builds the
    corresponding SAE variant, loads weights, and moves to *device* in eval
    mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = dict(checkpoint["hyper_parameters"])
    variant = hp.pop("variant")
    hp.pop("learning_rate", None)
    model = build_sae(variant, **hp)
    state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    if not state:
        raise RuntimeError(
            f"No 'model.*' keys found in checkpoint state_dict. "
            f"Available keys: {list(checkpoint['state_dict'].keys())[:10]}"
        )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def get_git_sha() -> str:
    """Return the current git commit SHA, or ``'unknown'`` if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_metadata(args: Any = None, **extra: Any) -> dict:
    """Build a metadata dict for JSON output files.

    Args:
        args: CLI argument namespace (optional).  If provided, all attributes
            are serialised as strings under the ``"args"`` key.
        **extra: Additional key-value pairs to include.

    Returns:
        Dict with ``timestamp``, ``git_sha``, and optionally ``args``.
    """
    meta: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
    }
    if args is not None:
        meta["args"] = {k: str(v) for k, v in vars(args).items()}
    meta.update(extra)
    return meta


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types transparently."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
