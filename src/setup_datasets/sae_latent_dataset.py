"""Generate SAE-encoded latent datasets (raw per-token and aggregated) from vision-encoder HDF5.

Usage:
    python -m src.setup_datasets.sae_latent_dataset \
        --input data/ve_latent_fairface.hdf5 \
        --checkpoint checkpoints/best.ckpt \
        --output-raw data/sae_latent_fairface.hdf5 \
        --output-agg data/agg_sae_latent_fairface.hdf5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from src.models.sparse_autoencoder import BaseSAE, build_sae

log = logging.getLogger(__name__)

RAW_STORAGE_DTYPE = np.float16
AGG_STORAGE_DTYPE = np.float16
ZSTD_LEVEL = 3


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BaseSAE, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" in ckpt:
        hp = dict(ckpt["hyper_parameters"])
        variant = hp.pop("variant")
        hp.pop("learning_rate", None)
        model = build_sae(variant, **hp)
        state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state)
    elif "config" in ckpt and "model_state_dict" in ckpt:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(ckpt["config"])
        import hydra

        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError("Unrecognised checkpoint format")

    model.to(device).eval()
    d_latent = model.hidden_size if hasattr(model, "hidden_size") else model.W_enc.shape[1]
    return model, d_latent


def process_split(
    f_in: h5py.File,
    f_out_raw: h5py.File,
    f_out_agg: h5py.File,
    split: str,
    model: BaseSAE,
    d_latent: int,
    batch_size: int,
    device: torch.device,
) -> None:
    if split not in f_in:
        log.warning("Split '%s' not in input. Skipping.", split)
        return

    input_group = f_in[split]
    out_raw = f_out_raw.create_group(split)
    out_agg = f_out_agg.create_group(split)

    for k, v in input_group.attrs.items():
        out_raw.attrs[k] = v
        out_agg.attrs[k] = v

    if "encoded" not in input_group or input_group["encoded"].shape[0] == 0:
        log.warning("No encoded data in split '%s'. Copying metadata only.", split)
        for name in ("labels", "original_indices"):
            if name in input_group:
                input_group.copy(input_group[name], out_raw, name=name)
                input_group.copy(input_group[name], out_agg, name=name)
        return

    encoded = input_group["encoded"]
    n_samples, seq_len, embed_dim = encoded.shape
    n_tokens = seq_len
    zstd = hdf5plugin.Zstd(clevel=ZSTD_LEVEL)

    out_raw.create_dataset(
        "data",
        shape=(n_samples, n_tokens, d_latent),
        dtype=RAW_STORAGE_DTYPE,
        chunks=(min(batch_size, n_samples), n_tokens, d_latent),
        compression=zstd,
    )
    agg_group = out_agg.create_group("data")
    agg_group.create_dataset(
        "mean",
        shape=(n_samples, d_latent),
        dtype=AGG_STORAGE_DTYPE,
        chunks=(min(batch_size * 4, n_samples), d_latent),
        compression=zstd,
    )
    agg_group.create_dataset(
        "max",
        shape=(n_samples, d_latent),
        dtype=AGG_STORAGE_DTYPE,
        chunks=(min(batch_size * 4, n_samples), d_latent),
        compression=zstd,
    )

    for name in ("labels", "original_indices"):
        if name in input_group:
            input_group.copy(input_group[name], out_raw, name=name)
            input_group.copy(input_group[name], out_agg, name=name)

    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Encoding {split}"):
            end = min(i + batch_size, n_samples)
            batch_np = encoded[i:end].reshape(-1, embed_dim)
            batch_t = torch.from_numpy(batch_np).to(device=device, dtype=model_dtype)
            latents = model.encode(batch_t).cpu().numpy()
            bs = end - i
            reshaped = latents.reshape(bs, n_tokens, d_latent)
            out_raw["data"][i:end] = reshaped.astype(RAW_STORAGE_DTYPE)
            agg_f32 = reshaped.astype(np.float32)
            agg_group["mean"][i:end] = np.mean(agg_f32, axis=1).astype(AGG_STORAGE_DTYPE)
            agg_group["max"][i:end] = np.max(agg_f32, axis=1).astype(AGG_STORAGE_DTYPE)

    log.info("Finished split '%s': %d samples.", split, n_samples)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate SAE latent HDF5 datasets")
    parser.add_argument("--input", type=Path, default=Path("data/ve_latent_fairface.hdf5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.ckpt"))
    parser.add_argument("--output-raw", type=Path, default=Path("data/sae_latent_fairface.hdf5"))
    parser.add_argument("--output-agg", type=Path, default=Path("data/agg_sae_latent_fairface.hdf5"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not args.input.is_file():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)
    if not args.checkpoint.is_file():
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    args.output_agg.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, d_latent = load_model(args.checkpoint, device)
    log.info("Model loaded. Latent dim = %d, device = %s", d_latent, device)

    try:
        with (
            h5py.File(args.input, "r") as f_in,
            h5py.File(args.output_raw, "w") as f_raw,
            h5py.File(args.output_agg, "w") as f_agg,
        ):
            for k, v in f_in.attrs.items():
                f_raw.attrs[k] = v
                f_agg.attrs[k] = v

            for split in ("training", "validation"):
                process_split(f_in, f_raw, f_agg, split, model, d_latent, args.batch_size, device)

        log.info("Done. Raw: %s | Aggregated: %s", args.output_raw, args.output_agg)

    except Exception:
        log.exception("Processing failed.")
        for p in (args.output_raw, args.output_agg):
            if p.exists():
                os.remove(p)
        sys.exit(1)


if __name__ == "__main__":
    main()
