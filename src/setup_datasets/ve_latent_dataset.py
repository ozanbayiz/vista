"""Extract vision-encoder latents from VLMs and save to HDF5.

Supports five VLMs (PaliGemma 2, SigLIP 2, DINOv3, Qwen3-VL, InternVL3.5)
and two datasets (ImageNet for SAE training, FairFace for demographic analysis).

Usage:
    python -m src.setup_datasets.ve_latent_dataset \
        --model-name paligemma2 \
        --dataset fairface \
        --output data/fairface_paligemma2.hdf5 \
        --batch-size 64
"""

from __future__ import annotations

import abc
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

ZSTD_LEVEL = 3
STORAGE_DTYPE = np.float16


# ---------------------------------------------------------------------------
# Abstract VE adapter
# ---------------------------------------------------------------------------


class VEAdapter(abc.ABC):
    """Thin adapter: loads a VLM vision encoder and preprocesses images."""

    # When set, encode() should return activations from this layer index
    # instead of the final layer. None means final layer (default).
    _extract_layer: int | None = None

    @abc.abstractmethod
    def load(self, device: torch.device) -> None:
        """Load model and processor onto *device*."""

    @abc.abstractmethod
    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Return tensors ready for the vision encoder."""

    @abc.abstractmethod
    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return patch activations: ``(batch, seq_len, hidden)``.

        If ``_extract_layer`` is set, return that layer's hidden states
        instead of the final layer output.
        """

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """Dimensionality of the VE output."""


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class PaliGemma2Adapter(VEAdapter):
    """PaliGemma 2 3B -- extracts SigLIP VE embeddings.

    Falls back to a community bf16 copy if the official gated repo
    is inaccessible.
    """

    MODEL_ID = "google/paligemma2-3b-pt-448"
    _FALLBACK_ID = "mlx-community/paligemma2-3b-mix-448-bf16"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")
        self._loaded_id: str = ""

    def load(self, device: torch.device) -> None:
        from transformers import PaliGemmaForConditionalGeneration, AutoImageProcessor

        self._device = device
        for model_id in (self.MODEL_ID, self._FALLBACK_ID):
            try:
                self._processor = AutoImageProcessor.from_pretrained(model_id)
                full_model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    ignore_mismatched_sizes=True,
                )
                # Vision tower location varies: top-level or nested in .model
                if hasattr(full_model, "vision_tower"):
                    self._model = full_model.vision_tower.to(device).eval()
                else:
                    self._model = full_model.model.vision_tower.to(device).eval()
                del full_model
                self._loaded_id = model_id
                log.info("Loaded PaliGemma 2 vision tower (%s)", model_id)
                return
            except (OSError, Exception) as e:
                log.warning("Cannot load %s: %s", model_id, e)
        raise RuntimeError("Could not load any PaliGemma 2 variant")

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self._processor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].to(
                self._device, dtype=torch.float16
            )
        }

    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._model(
            preprocessed["pixel_values"], output_hidden_states=True
        )
        if self._extract_layer is not None:
            return out.hidden_states[self._extract_layer].float()
        return out.last_hidden_state.float()

    @property
    def hidden_size(self) -> int:
        return 1152


class SigLIP2Adapter(VEAdapter):
    """SigLIP 2 So400m standalone encoder."""

    MODEL_ID = "google/siglip2-so400m-patch14-384"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import AutoModel, AutoImageProcessor

        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self._model = (
            AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )
        log.info("Loaded SigLIP 2 (%s)", self.MODEL_ID)

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self._processor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].to(
                self._device, dtype=torch.float16
            )
        }

    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._model.vision_model(
            preprocessed["pixel_values"], output_hidden_states=True
        )
        return out.last_hidden_state.float()

    @property
    def hidden_size(self) -> int:
        return 1152


class DINOv3Adapter(VEAdapter):
    """DINOv2/v3 ViT-L/14 self-supervised encoder.

    Falls back to DINOv2-large if DINOv3 is gated/inaccessible.
    """

    MODEL_ID = "facebook/dinov2-large"
    _PREFERRED_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import AutoModel, AutoImageProcessor

        self._device = device
        # Try DINOv3 first, fall back to DINOv2-large if gated
        model_id = self.MODEL_ID
        try:
            self._processor = AutoImageProcessor.from_pretrained(self._PREFERRED_ID)
            self._model = (
                AutoModel.from_pretrained(self._PREFERRED_ID, torch_dtype=torch.float16)
                .to(device)
                .eval()
            )
            model_id = self._PREFERRED_ID
        except (OSError, Exception) as e:
            log.warning("DINOv3 unavailable (%s), falling back to %s", e, self.MODEL_ID)
            self._processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self._model = (
                AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=torch.float16)
                .to(device)
                .eval()
            )
        log.info("Loaded DINO encoder (%s)", model_id)

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self._processor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].to(
                self._device, dtype=torch.float16
            )
        }

    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._model(
            preprocessed["pixel_values"], output_hidden_states=True
        )
        return out.last_hidden_state.float()

    @property
    def hidden_size(self) -> int:
        return 1024


class Qwen3VLAdapter(VEAdapter):
    """Qwen3-VL-2B -- extracts vision encoder embeddings."""

    MODEL_ID = "/scratch/current/ozanbayiz/models/Qwen3-VL-2B-Instruct"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        # Force fixed resolution: 448x448 = 200704 pixels
        # This ensures all images produce the same number of tokens
        fixed_pixels = 448 * 448
        self._processor.image_processor.min_pixels = fixed_pixels
        self._processor.image_processor.max_pixels = fixed_pixels
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
        )
        # In Qwen3VL, visual encoder is at model.model.visual
        self._model = full_model.model.visual.to(device).eval()
        del full_model.model.language_model
        del full_model.lm_head
        log.info("Loaded Qwen3-VL visual encoder (%s)", self.MODEL_ID)

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        # Force all images to exact 448x448 square to guarantee consistent
        # token counts across all images (different aspect ratios would
        # otherwise produce different grid configurations in Qwen3-VL).
        resized = [img.convert("RGB").resize((448, 448)) for img in images]
        inputs = self._processor.image_processor(
            images=resized, return_tensors="pt"
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_values = preprocessed["pixel_values"].to(dtype=torch.float16)
        grid_thw = preprocessed["image_grid_thw"]
        out = self._model(pixel_values, grid_thw=grid_thw)
        # Returns BaseModelOutputWithDeepstackFeatures.
        # Use pooler_output which is the post-PatchMerger output
        # (total_merged_tokens, out_hidden_size=2048) -- this is what feeds
        # into the language model and what the SAE should model.
        hidden = out.pooler_output.float()
        total_tokens = hidden.shape[0]
        n_images = grid_thw.shape[0]
        d = hidden.shape[-1]

        # Check if tokens divide evenly among images
        if total_tokens % n_images == 0:
            tokens_per_image = total_tokens // n_images
            return hidden.view(n_images, tokens_per_image, d)

        # Uneven: use stored target seq_len if available (from _probe_seq_len),
        # otherwise use the average (rounded up)
        target_len = getattr(self, "_target_seq_len", -(-total_tokens // n_images))
        result = hidden.new_zeros(n_images, target_len, d)
        # Distribute tokens evenly, truncating/padding as needed
        base = total_tokens // n_images
        offset = 0
        for i in range(n_images):
            # Last image gets remaining tokens
            ntok = total_tokens - offset if i == n_images - 1 else base
            use = min(ntok, target_len)
            result[i, :use, :] = hidden[offset : offset + use]
            offset += ntok
        return result

    @property
    def hidden_size(self) -> int:
        return 2048  # Qwen3-VL-2B visual encoder out_hidden_size after PatchMerger


class InternVL35Adapter(VEAdapter):
    """InternVL3.5-2B -- extracts InternViT-300M embeddings."""

    MODEL_ID = "OpenGVLab/InternVL3_5-2B"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoConfig

        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )

        # Load vision model directly to avoid full InternVLChatModel
        # incompatibilities with transformers 5.x
        config = AutoConfig.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        vision_config = config.vision_config

        # Use transformers' own dynamic import to get InternVisionModel
        # (avoids loading full InternVLChatModel which has compat issues with transformers 5.x)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        InternVisionModel = get_class_from_dynamic_module(
            "modeling_intern_vit.InternVisionModel",
            self.MODEL_ID,
        )

        # Create vision model on CPU
        vision_model = InternVisionModel(vision_config)

        # Load only vision model weights from the full checkpoint
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        # Try sharded index first, then single file
        state_dict = {}
        try:
            index_path = hf_hub_download(
                self.MODEL_ID, "model.safetensors.index.json"
            )
            import json
            with open(index_path) as f:
                index = json.load(f)
            vision_shards = set()
            for key, shard in index["weight_map"].items():
                if key.startswith("vision_model."):
                    vision_shards.add(shard)
            for shard_name in vision_shards:
                shard_path = hf_hub_download(self.MODEL_ID, shard_name)
                shard_dict = safetensors.torch.load_file(shard_path)
                for k, v in shard_dict.items():
                    if k.startswith("vision_model."):
                        state_dict[k.replace("vision_model.", "")] = v
        except Exception:
            # Single safetensors file (no index)
            try:
                st_path = hf_hub_download(self.MODEL_ID, "model.safetensors")
                all_weights = safetensors.torch.load_file(st_path)
                state_dict = {
                    k.replace("vision_model.", ""): v
                    for k, v in all_weights.items()
                    if k.startswith("vision_model.")
                }
                del all_weights
            except Exception as e:
                raise RuntimeError(
                    f"Could not load vision weights for {self.MODEL_ID}"
                ) from e

        vision_model.load_state_dict(state_dict, strict=False)
        self._model = vision_model.half().to(device).eval()
        del state_dict
        log.info("Loaded InternVL3.5 vision model (%s)", self.MODEL_ID)

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self._processor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].to(
                self._device, dtype=torch.float16
            )
        }

    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._model(
            preprocessed["pixel_values"], output_hidden_states=True
        )
        return out.last_hidden_state.float()

    @property
    def hidden_size(self) -> int:
        return 1024


class Qwen2VLAdapter(VEAdapter):
    """Qwen2-VL-2B -- extracts vision encoder embeddings.

    Uses ``Qwen2VLForConditionalGeneration`` from HF Transformers, extracts
    ``model.visual`` (a ``Qwen2VisionTransformerPretrainedModel``).  The visual
    model output is a flat ``(total_tokens, hidden_dim)`` tensor; we reshape to
    ``(batch, tokens_per_image, hidden_dim)`` assuming fixed image resolution.
    """

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        # Force fixed resolution: 224x224 to ensure uniform token counts.
        # Qwen2-VL: 224px / 14 patch = 16x16 = 256 patches, merged 2x2 → 64 tokens.
        fixed_pixels = 224 * 224
        self._processor.image_processor.min_pixels = fixed_pixels
        self._processor.image_processor.max_pixels = fixed_pixels

        full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
        )
        # Extract only the visual encoder, discard LLM to save memory
        self._model = full_model.visual.to(device).eval()
        del full_model.model
        del full_model.lm_head
        log.info("Loaded Qwen2-VL visual encoder (%s)", self.MODEL_ID)

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        # Force all images to 224x224 for consistent token counts
        resized = [img.convert("RGB").resize((224, 224)) for img in images]
        inputs = self._processor.image_processor(
            images=resized, return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode(self, preprocessed: dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_values = preprocessed["pixel_values"].to(dtype=torch.float16)
        grid_thw = preprocessed["image_grid_thw"]
        # Qwen2VisionTransformerPretrainedModel.forward returns a raw tensor
        hidden = self._model(pixel_values, grid_thw=grid_thw).float()
        total_tokens = hidden.shape[0]
        n_images = grid_thw.shape[0]
        d = hidden.shape[-1]

        if total_tokens % n_images == 0:
            tokens_per_image = total_tokens // n_images
            return hidden.view(n_images, tokens_per_image, d)

        # Fallback: pad/truncate (shouldn't happen with fixed resolution)
        target_len = -(-total_tokens // n_images)  # ceil division
        result = hidden.new_zeros(n_images, target_len, d)
        base = total_tokens // n_images
        offset = 0
        for i in range(n_images):
            ntok = total_tokens - offset if i == n_images - 1 else base
            use = min(ntok, target_len)
            result[i, :use, :] = hidden[offset : offset + use]
            offset += ntok
        return result

    @property
    def hidden_size(self) -> int:
        return 1536


ADAPTERS: dict[str, type[VEAdapter]] = {
    "paligemma2": PaliGemma2Adapter,
    "siglip2": SigLIP2Adapter,
    "dinov3": DINOv3Adapter,
    "qwen3vl": Qwen3VLAdapter,
    "qwen2vl": Qwen2VLAdapter,
    "internvl35": InternVL35Adapter,
}


# ---------------------------------------------------------------------------
# Dataset wrappers (return PIL images + metadata)
# ---------------------------------------------------------------------------


class FairFaceImageDataset(Dataset):
    """Wraps HuggingFace FairFace split, returns PIL images and label indices."""

    RACE_MAP = {
        "White": 0,
        "Black": 1,
        "Latino_Hispanic": 2,
        "East Asian": 3,
        "Southeast Asian": 4,
        "Indian": 5,
        "Middle Eastern": 6,
    }
    GENDER_MAP = {"Male": 0, "Female": 1}
    AGE_MAP = {
        "0-2": 0,
        "3-9": 1,
        "10-19": 2,
        "20-29": 3,
        "30-39": 4,
        "40-49": 5,
        "50-59": 6,
        "60-69": 7,
        "more than 70": 8,
    }

    _LABEL_MAPS: dict[str, dict[str, int]] = {
        "age": AGE_MAP,
        "gender": GENDER_MAP,
        "race": RACE_MAP,
    }

    def __init__(self, hf_dataset: Any) -> None:
        self._ds = hf_dataset
        # HuggingFace ClassLabel features return integers, not strings.
        # Detect this once so __getitem__ handles both formats correctly.
        sample = self._ds[0]
        self._int_labels: bool = isinstance(sample["age"], (int, np.integer))
        # Remapping tables: HF int -> our int. Identity if ordering matches.
        self._remap: dict[str, np.ndarray | None] = {"age": None, "gender": None, "race": None}
        if self._int_labels:
            self._build_remap()
            log.info("FairFace labels are integer ClassLabels (remapping built).")
        else:
            log.info("FairFace labels are strings; using string-to-int maps.")

    def _build_remap(self) -> None:
        """Build HF-int -> our-int remapping arrays for each label column."""
        features = getattr(self._ds, "features", None)
        if features is None:
            return
        for key, expected_map in self._LABEL_MAPS.items():
            feat = features.get(key)
            if feat is None or not hasattr(feat, "names"):
                continue
            hf_names = feat.names  # e.g. ['East Asian', 'Indian', ...]
            # Build remap: remap[hf_idx] = our_idx
            remap = np.arange(len(hf_names), dtype=np.int64)
            needs_remap = False
            for hf_idx, name in enumerate(hf_names):
                our_idx = expected_map.get(name)
                if our_idx is None:
                    log.warning("Unknown label '%s' for '%s'; keeping HF index %d", name, key, hf_idx)
                    continue
                remap[hf_idx] = our_idx
                if hf_idx != our_idx:
                    needs_remap = True
            if needs_remap:
                log.info("Label '%s' needs remapping: HF order %s", key, hf_names)
                self._remap[key] = remap

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        row = self._ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        if self._int_labels:
            # Apply remapping if HF ordering differs from ours
            age = int(self._remap["age"][row["age"]]) if self._remap["age"] is not None else row["age"]
            gender = int(self._remap["gender"][row["gender"]]) if self._remap["gender"] is not None else row["gender"]
            race = int(self._remap["race"][row["race"]]) if self._remap["race"] is not None else row["race"]
            return {
                "image": img,
                "age": age,
                "gender": gender,
                "race": race,
            }

        return {
            "image": img,
            "age": self.AGE_MAP[row["age"]],
            "gender": self.GENDER_MAP[row["gender"]],
            "race": self.RACE_MAP[row["race"]],
        }


class ImageNetImageDataset(Dataset):
    """Wraps HuggingFace ImageNet, returns PIL images and class labels."""

    def __init__(self, hf_dataset: Any) -> None:
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        row = self._ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        return {"image": img, "label": row["label"]}


def _pil_collate(batch: list[dict]) -> dict[str, Any]:
    """Collate that keeps PIL images as a list (no tensor stacking)."""
    return {k: [b[k] for b in batch] for k in batch[0]}


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _create_hdf5_datasets(
    group: h5py.Group,
    n_samples: int,
    seq_len: int,
    hidden: int,
    batch_size: int,
    label_names: list[str],
) -> tuple[h5py.Dataset, dict[str, h5py.Dataset]]:
    """Create chunked, Zstd-compressed HDF5 datasets within *group*."""
    zstd = hdf5plugin.Zstd(clevel=ZSTD_LEVEL)

    encoded_ds = group.create_dataset(
        "encoded",
        shape=(n_samples, seq_len, hidden),
        dtype=STORAGE_DTYPE,
        chunks=(min(batch_size, n_samples), seq_len, hidden),
        compression=zstd,
    )

    label_group = group.create_group("labels")
    label_datasets: dict[str, h5py.Dataset] = {}
    for name in label_names:
        label_datasets[name] = label_group.create_dataset(
            name,
            shape=(n_samples,),
            dtype=np.int64,
            chunks=(min(batch_size * 4, n_samples),),
            compression=zstd,
        )

    return encoded_ds, label_datasets


def _probe_seq_len(adapter: VEAdapter) -> int:
    """Send a dummy image through the adapter to discover the output seq length."""
    dummy = Image.new("RGB", (448, 448))
    preprocessed = adapter.preprocess([dummy])
    out = adapter.encode(preprocessed)
    seq_len = out.shape[1]
    # Store on the adapter so encode() can pad to a consistent length
    adapter._target_seq_len = seq_len  # type: ignore[attr-defined]
    return seq_len


# ---------------------------------------------------------------------------
# FairFace extraction (stratified train/val split)
# ---------------------------------------------------------------------------


def extract_fairface(
    adapter: VEAdapter,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float = 0.15,
    max_samples: int | None = None,
) -> None:
    """Extract VE latents from FairFace and write to HDF5 with stratified split."""
    from datasets import load_dataset
    from sklearn.model_selection import StratifiedShuffleSplit

    log.info("Loading FairFace from HuggingFace ...")
    hf_ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train")

    if max_samples is not None and max_samples < len(hf_ds):
        log.info("Truncating FairFace from %d to %d samples", len(hf_ds), max_samples)
        hf_ds = hf_ds.select(range(max_samples))

    ff_dataset = FairFaceImageDataset(hf_ds)
    n_total = len(ff_dataset)

    log.info("Gathering labels for stratified split ...")
    all_races = np.array([hf_ds[i]["race"] for i in range(n_total)])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=42
    )
    train_idx, val_idx = next(splitter.split(np.zeros(n_total), all_races))

    log.info("Split: %d training, %d validation", len(train_idx), len(val_idx))

    seq_len = _probe_seq_len(adapter)
    hidden = adapter.hidden_size
    log.info("VE output: seq_len=%d, hidden=%d", seq_len, hidden)

    label_names = ["age", "gender", "race"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f_out:
        f_out.attrs["model"] = type(adapter).__name__
        f_out.attrs["dataset"] = "fairface"

        for split_name, indices in [
            ("training", train_idx),
            ("validation", val_idx),
        ]:
            subset = torch.utils.data.Subset(ff_dataset, indices.tolist())
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_pil_collate,
                pin_memory=False,
            )

            group = f_out.create_group(split_name)
            n = len(indices)

            encoded_ds, label_ds = _create_hdf5_datasets(
                group, n, seq_len, hidden, batch_size, label_names
            )

            zstd = hdf5plugin.Zstd(clevel=ZSTD_LEVEL)
            group.create_dataset(
                "original_indices",
                data=np.array(indices, dtype=np.int64),
                compression=zstd,
            )

            write_idx = 0
            for batch in tqdm(loader, desc=f"FairFace {split_name}"):
                images: list[Image.Image] = batch["image"]
                preprocessed = adapter.preprocess(images)
                hidden_states = adapter.encode(preprocessed)

                bs = hidden_states.shape[0]
                arr = hidden_states.cpu().numpy().astype(STORAGE_DTYPE)
                encoded_ds[write_idx : write_idx + bs] = arr

                for name in label_names:
                    labs = np.array(batch[name], dtype=np.int64)
                    label_ds[name][write_idx : write_idx + bs] = labs

                write_idx += bs

            log.info(
                "Wrote %d samples to %s/%s", write_idx, output_path, split_name
            )

    log.info("FairFace extraction complete: %s", output_path)


# ---------------------------------------------------------------------------
# ImageNet extraction (train split for SAE training)
# ---------------------------------------------------------------------------


def extract_imagenet(
    adapter: VEAdapter,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    imagenet_path: str | None = None,
    max_samples: int | None = None,
) -> None:
    """Extract VE latents from ImageNet training set and write to HDF5."""
    log.info("Loading ImageNet ...")
    if imagenet_path is not None:
        # Use torchvision ImageFolder for fast local loading
        from torchvision.datasets import ImageFolder

        class _LocalImageNetDataset(Dataset):
            """Thin wrapper around torchvision ImageFolder to match our dict API."""

            def __init__(self, root: str) -> None:
                self._folder = ImageFolder(root)

            def __len__(self) -> int:
                return len(self._folder)

            def __getitem__(self, idx: int) -> dict:
                img, label = self._folder[idx]
                return {"image": img.convert("RGB"), "label": label}

        in_dataset = _LocalImageNetDataset(imagenet_path)
    else:
        from datasets import load_dataset
        hf_ds = load_dataset(
            "ILSVRC/imagenet-1k", split="train", trust_remote_code=True
        )
        in_dataset = ImageNetImageDataset(hf_ds)

    n_total = len(in_dataset)
    if max_samples is not None and max_samples < n_total:
        log.info("Truncating ImageNet from %d to %d samples", n_total, max_samples)
        n_total = max_samples

    log.info("ImageNet: %d images", n_total)

    seq_len = _probe_seq_len(adapter)
    hidden = adapter.hidden_size
    log.info("VE output: seq_len=%d, hidden=%d", seq_len, hidden)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split into training/validation (90/10)
    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_size = max(1, n_total // 10)
    val_indices = set(indices[:val_size].tolist())

    train_indices = [i for i in range(n_total) if i not in val_indices]
    val_idx_list = sorted(val_indices)

    log.info("Split: %d training, %d validation", len(train_indices), len(val_idx_list))

    with h5py.File(output_path, "w") as f_out:
        f_out.attrs["model"] = type(adapter).__name__
        f_out.attrs["dataset"] = "imagenet"

        for split_name, split_indices in [
            ("training", train_indices),
            ("validation", val_idx_list),
        ]:
            subset = torch.utils.data.Subset(in_dataset, split_indices)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_pil_collate,
                pin_memory=False,
            )

            group = f_out.create_group(split_name)
            n = len(split_indices)

            encoded_ds, label_ds = _create_hdf5_datasets(
                group, n, seq_len, hidden, batch_size, ["class"]
            )

            write_idx = 0
            for batch in tqdm(loader, desc=f"ImageNet {split_name}"):
                images = batch["image"]
                preprocessed = adapter.preprocess(images)
                hidden_states = adapter.encode(preprocessed)

                bs = hidden_states.shape[0]
                arr = hidden_states.cpu().numpy().astype(STORAGE_DTYPE)
                encoded_ds[write_idx : write_idx + bs] = arr
                label_ds["class"][write_idx : write_idx + bs] = np.array(
                    batch["label"], dtype=np.int64
                )
                write_idx += bs

            log.info("Wrote %d samples to %s/%s", write_idx, output_path, split_name)

    log.info("ImageNet extraction complete: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract VE latents to HDF5")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=list(ADAPTERS.keys()),
        help="VLM to extract from",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["fairface", "imagenet"],
        help="Dataset to extract from",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 path",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--imagenet-path",
        type=str,
        default=None,
        help="Local path to ImageNet (if not using HuggingFace download)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples for fast small-scale testing",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Extract activations from this intermediate layer index "
        "(0-indexed, includes embedding layer at index 0). "
        "If not specified, uses the final layer output.",
    )
    args = parser.parse_args()

    if args.output.exists():
        log.warning("Output file already exists: %s. Overwriting.", args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    adapter = ADAPTERS[args.model_name]()
    if args.layer is not None:
        adapter._extract_layer = args.layer
        log.info("Extracting intermediate layer %d", args.layer)
    adapter.load(device)

    if args.dataset == "fairface":
        extract_fairface(adapter, args.output, args.batch_size, args.num_workers, max_samples=args.max_samples)
    elif args.dataset == "imagenet":
        extract_imagenet(
            adapter,
            args.output,
            args.batch_size,
            args.num_workers,
            args.imagenet_path,
            max_samples=args.max_samples,
        )
    else:
        log.error("Unknown dataset: %s", args.dataset)
        sys.exit(1)


if __name__ == "__main__":
    main()
