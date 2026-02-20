"""VLM causal tracing: measure how SAE-identified demographic features
in vision encoders causally influence downstream language model outputs.

Loads the full PaliGemma 2 VLM (vision encoder **and** language model),
applies SAE-based demographic feature interventions via a PyTorch forward
hook on the vision tower, and generates text (captions or VQA answers) with
and without the intervention active.

The hook-based intervention loop::

    vision_tower(x) -> hook intercepts output ->
        SAE.encode -> modify SDFs -> SAE.decode ->
    modified output flows to projector + language model

Usage:
    # Captioning with race SDF suppression
    python -m src.vlm_generate \\
        --sae-checkpoint checkpoints/sae_paligemma2.ckpt \\
        --sdf-dir results/sdf_paligemma2 \\
        --output-dir results/causal_tracing \\
        --task caption --attribute race

    # VQA with gender SDF suppression
    python -m src.vlm_generate \\
        --sae-checkpoint checkpoints/sae_paligemma2.ckpt \\
        --sdf-dir results/sdf_paligemma2 \\
        --output-dir results/causal_tracing \\
        --task vqa --attribute gender
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

from src.intervene import intervene_on_activations
from src.models.sparse_autoencoder import BaseSAE
from src.utils import NumpyEncoder, build_metadata, load_sae_from_checkpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VQA prompt design
# ---------------------------------------------------------------------------

DEMOGRAPHIC_PROBES: dict[str, list[str]] = {
    "age": ["What is this person's approximate age?"],
    "gender": ["Describe this person's gender."],
    "race": ["What is this person's race or ethnicity?"],
}

CONTROL_PROBES: list[str] = [
    "What is this person wearing?",
    "Describe the background of this image.",
    "What emotion is this person showing?",
]

# Human-readable label names (mirrors FairFaceImageDataset label maps)
RACE_NAMES: dict[int, str] = {
    0: "White",
    1: "Black",
    2: "Latino/Hispanic",
    3: "East Asian",
    4: "Southeast Asian",
    5: "Indian",
    6: "Middle Eastern",
}
GENDER_NAMES: dict[int, str] = {0: "Male", 1: "Female"}
AGE_NAMES: dict[int, str] = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+",
}
LABEL_NAMES: dict[str, dict[int, str]] = {
    "age": AGE_NAMES,
    "gender": GENDER_NAMES,
    "race": RACE_NAMES,
}


# ---------------------------------------------------------------------------
# Abstract VLM adapter
# ---------------------------------------------------------------------------


class FullVLMAdapter(abc.ABC):
    """Abstract adapter for loading a full VLM and generating text.

    Subclasses must implement:
      - ``load(device)`` -- load the full VLM (vision encoder + language model).
      - ``generate(images, prompts)`` -- generate text from image-prompt pairs.
      - ``get_vision_module()`` -- return the hookable vision sub-module.
      - ``hidden_size`` -- dimensionality of the vision encoder output.
    """

    @abc.abstractmethod
    def load(self, device: torch.device) -> None:
        """Load the full VLM onto *device*."""

    @abc.abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Generate text for each (image, prompt) pair.

        Args:
            images: Batch of PIL images.
            prompts: One text prompt per image.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            List of generated strings (one per image).
        """

    @abc.abstractmethod
    def get_vision_module(self) -> nn.Module:
        """Return the ``nn.Module`` whose output the hook should intercept."""

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """Dimensionality of the vision encoder hidden states."""


# ---------------------------------------------------------------------------
# PaliGemma 2 adapter
# ---------------------------------------------------------------------------


class PaliGemma2FullAdapter(FullVLMAdapter):
    """Full PaliGemma 2 VLM adapter for causal tracing experiments.

    Loads ``PaliGemmaForConditionalGeneration`` in full (vision tower +
    multi-modal projector + Gemma 2 language model) so that generated text
    reflects changes made to vision encoder activations.

    Falls back to a community bf16 copy when the official gated repo is
    inaccessible, mirroring the fallback strategy in
    ``PaliGemma2Adapter`` (``src/setup_datasets/ve_latent_dataset.py``).
    """

    MODEL_ID = "google/paligemma2-3b-pt-448"
    _FALLBACK_ID = "mlx-community/paligemma2-3b-mix-448-bf16"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")
        self._loaded_id: str = ""

    def load(self, device: torch.device) -> None:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self._device = device
        for model_id in (self.MODEL_ID, self._FALLBACK_ID):
            # Community mirrors may have minor weight-shape mismatches
            # (e.g., MLX-converted models store conv weights in
            # channels-last [O, H, W, C] vs PyTorch's [O, C, H, W]).
            is_fallback = model_id != self.MODEL_ID
            try:
                log.info("Attempting to load PaliGemma 2 (%s) ...", model_id)
                self._processor = AutoProcessor.from_pretrained(model_id)
                self._model = (
                    PaliGemmaForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        ignore_mismatched_sizes=is_fallback,
                    )
                    .to(device)
                    .eval()
                )
                # MLX models store patch_embedding.weight as (O, H, W, C)
                # but PyTorch Conv2d expects (O, C, H, W).  Fix the layout.
                if is_fallback:
                    self._fix_mlx_patch_embedding()
                self._loaded_id = model_id
                log.info("Loaded PaliGemma 2 full model (%s)", model_id)
                return
            except (OSError, Exception) as exc:
                log.warning("Cannot load %s: %s", model_id, exc)
        raise RuntimeError("Could not load any PaliGemma 2 variant")

    # ------------------------------------------------------------------
    def _fix_mlx_patch_embedding(self) -> None:
        """Transpose the SigLIP patch-embedding weight from MLX layout.

        MLX stores ``Conv2d`` weights as ``(O, H, W, C)`` (channels-last)
        while PyTorch expects ``(O, C, H, W)`` (channels-first).  When
        ``ignore_mismatched_sizes=True``, HF Transformers randomly re-inits
        the weight instead of transposing it, so we load the raw tensor
        from the safetensors file and permute it ourselves.
        """
        from safetensors import safe_open

        vt_key = "vision_tower.vision_model.embeddings.patch_embedding.weight"
        # Find the vision tower module
        vm = self.get_vision_module()
        patch_emb = vm.vision_model.embeddings.patch_embedding

        # Load the original weight from the checkpoint file
        import os
        from huggingface_hub import hf_hub_download, scan_cache_dir

        cache = scan_cache_dir()
        safetensor_paths: list[str] = []
        for repo in cache.repos:
            if repo.repo_id == self._FALLBACK_ID:
                for rev in repo.revisions:
                    for f in rev.files:
                        if f.file_name.endswith(".safetensors"):
                            safetensor_paths.append(str(f.file_path))
                break

        fixed = False
        for st_path in safetensor_paths:
            with safe_open(st_path, framework="pt", device="cpu") as st:
                if vt_key in st.keys():
                    raw_w = st.get_tensor(vt_key)  # (O, H, W, C) in MLX
                    if raw_w.shape != patch_emb.weight.shape:
                        raw_w = raw_w.permute(0, 3, 1, 2).contiguous()
                    raw_w = raw_w.to(
                        dtype=patch_emb.weight.dtype,
                        device=patch_emb.weight.device,
                    )
                    patch_emb.weight.data.copy_(raw_w)
                    log.info(
                        "Fixed MLX patch_embedding weight: %s -> %s",
                        "OHWC",
                        "OIHW",
                    )
                    fixed = True
                    break
        if not fixed:
            log.warning(
                "Could not find %s in cached safetensors; patch embedding may be randomly initialised.",
                vt_key,
            )

    # ------------------------------------------------------------------

    def generate(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, input_len:]
        texts = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return [t.strip() for t in texts]

    def get_vision_module(self) -> nn.Module:
        """Return the SigLIP vision tower used by PaliGemma 2."""
        if hasattr(self._model, "vision_tower"):
            return self._model.vision_tower
        if hasattr(self._model, "model") and hasattr(self._model.model, "vision_tower"):
            return self._model.model.vision_tower
        raise AttributeError("Cannot locate vision_tower in the loaded PaliGemma 2 model")

    @property
    def hidden_size(self) -> int:
        return 1152


# ---------------------------------------------------------------------------
# Qwen2-VL adapter
# ---------------------------------------------------------------------------


class Qwen2VLFullAdapter(FullVLMAdapter):
    """Full Qwen2-VL-2B adapter for causal tracing experiments.

    Loads ``Qwen2VLForConditionalGeneration`` (Qwen2-VL-2B-Instruct) with its
    ViT-based vision encoder + Qwen2 language model.  The vision module
    (``model.visual``) is hookable for intervention experiments.

    Qwen2-VL uses dynamic resolution and variable token counts.  The vision
    encoder output is a flat ``(n_tokens, hidden_dim)`` tensor (not batched),
    which is compatible with the noise intervention hook.
    """

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._device = device
        log.info("Loading Qwen2-VL-2B (%s) ...", self.MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
            )
            .to(device)
            .eval()
        )
        log.info("Loaded Qwen2-VL-2B full model (%s)", self.MODEL_ID)

    def generate(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        # Qwen2-VL uses a chat template with <|vision_start|> / <|vision_end|>
        # markers.  Build one conversation per image.
        texts: list[str] = []
        for prompt in prompts:
            # Translate PaliGemma2-style "caption en" to a natural prompt
            user_text = prompt
            if prompt == "caption en":
                user_text = "Describe this image."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)

        inputs = self._processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, input_len:]
        decoded = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )
        return [t.strip() for t in decoded]

    def get_vision_module(self) -> nn.Module:
        """Return the Qwen2 vision transformer (``model.visual``)."""
        return self._model.visual

    @property
    def hidden_size(self) -> int:
        # Qwen2-VL-2B vision encoder output after PatchMerger: 1536
        return 1536


class Qwen3VLFullAdapter(FullVLMAdapter):
    """Full Qwen3-VL-2B adapter for causal tracing experiments.

    Loads ``Qwen3VLForConditionalGeneration`` (Qwen3-VL-2B-Instruct) with its
    DeepStack-enhanced ViT vision encoder + Qwen3 language model.  The vision
    module (``model.model.visual``) is hookable for intervention experiments.

    Qwen3-VL uses dynamic resolution and variable token counts.  The vision
    encoder output is a flat ``(n_tokens, hidden_dim)`` tensor (not batched),
    which is compatible with the noise intervention hook.
    """

    MODEL_ID = "/scratch/current/ozanbayiz/models/Qwen3-VL-2B-Instruct"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._device = device
        log.info("Loading Qwen3-VL-2B (%s) ...", self.MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
            )
            .to(device)
            .eval()
        )
        log.info("Loaded Qwen3-VL-2B full model (%s)", self.MODEL_ID)

    def generate(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        # Qwen3-VL uses a chat template similar to Qwen2-VL.
        texts: list[str] = []
        for prompt in prompts:
            user_text = prompt
            if prompt == "caption en":
                user_text = "Describe this image."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)

        inputs = self._processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, input_len:]
        decoded = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )
        return [t.strip() for t in decoded]

    def get_vision_module(self) -> nn.Module:
        """Return the Qwen3 vision transformer (``model.model.visual``)."""
        return self._model.model.visual

    @property
    def hidden_size(self) -> int:
        # Qwen3-VL-2B vision encoder output after PatchMerger: 2048
        return 2048


FULL_VLM_ADAPTERS: dict[str, type[FullVLMAdapter]] = {
    "paligemma2": PaliGemma2FullAdapter,
    "qwen2vl": Qwen2VLFullAdapter,
    "qwen3vl": Qwen3VLFullAdapter,
}


# ---------------------------------------------------------------------------
# SAE intervention hook
# ---------------------------------------------------------------------------


class SAEInterventionHook:
    """Context manager that installs a forward hook on the vision encoder.

    While the hook is active, every forward pass through the vision module
    is intercepted: the output hidden states are encoded through the SAE,
    the target SDF features are modified (suppressed / amplified /
    attenuated), and the SAE decoder reconstructs the modified embeddings
    which then flow into the multi-modal projector and language model.

    Usage::

        with SAEInterventionHook(adapter, sae, features, mode="suppression"):
            modified_texts = adapter.generate(images, prompts)
    """

    def __init__(
        self,
        adapter: FullVLMAdapter,
        sae: BaseSAE,
        target_features: list[int] | np.ndarray,
        mode: str = "suppression",
        alpha: float = 0.0,
    ) -> None:
        """
        Args:
            adapter: A loaded ``FullVLMAdapter``.
            sae: Trained SAE whose ``input_size`` matches the vision
                encoder ``hidden_size``.
            target_features: SDF feature indices to modify.
            mode: One of ``'suppression'``, ``'amplification'``,
                ``'attenuation'``.
            alpha: Scaling factor (ignored for suppression).
        """
        self._adapter = adapter
        self._sae = sae
        self._target_features = list(target_features)
        self._mode = mode
        self._alpha = alpha
        self._hook_handle: Any = None

    # -- hook callback ------------------------------------------------------

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """Intercept vision encoder output and apply SAE intervention."""
        # Extract the main hidden-state tensor from the output.
        # Prefer pooler_output (post-merger) when available: Qwen3-VL's
        # DeepStack architecture returns last_hidden_state at pre-merger
        # dimension (1024) while pooler_output is the post-PatchMerger
        # representation (2048) that feeds into the language model and
        # that the SAE was trained on.
        use_pooler = (
            hasattr(output, "pooler_output")
            and output.pooler_output is not None
        )
        if use_pooler:
            hidden = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        original_dtype = hidden.dtype
        original_shape = hidden.shape

        # Flatten all patch tokens for SAE processing.
        # Handles both 3-D (batch, seq_len, hidden_dim) -- e.g. PaliGemma2 --
        # and 2-D (total_tokens, hidden_dim) -- e.g. Qwen2-VL.
        hidden_dim = hidden.shape[-1]
        flat = hidden.reshape(-1, hidden_dim)

        # Match SAE parameter dtype (the SAE may be float32 while the vision
        # tower runs in float16).
        sae_dtype = next(self._sae.parameters()).dtype
        flat = flat.to(dtype=sae_dtype)

        # Encode -> modify target SDFs -> decode.
        with torch.no_grad():
            acts = self._sae.encode(flat)
            modified_acts = intervene_on_activations(
                acts,
                self._target_features,
                self._mode,
                self._alpha,
            )
            modified_flat = self._sae.decode(modified_acts)

        modified_hidden = modified_flat.to(dtype=original_dtype).reshape(original_shape)

        log.debug(
            "Hook fired: shape=%s, features_targeted=%d, L2_change=%.6f",
            original_shape,
            len(self._target_features),
            (modified_hidden.float() - hidden.float()).pow(2).mean().item(),
        )

        # Return the modified output in its original container type.
        if use_pooler:
            output.pooler_output = modified_hidden
            return output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = modified_hidden
            return output
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "SAEInterventionHook":
        vision_module = self._adapter.get_vision_module()
        self._hook_handle = vision_module.register_forward_hook(
            self._hook_fn,
        )
        log.info(
            "Registered SAE intervention hook (mode=%s, %d target features)",
            self._mode,
            len(self._target_features),
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            log.info("Removed SAE intervention hook")
        return False


# ---------------------------------------------------------------------------
# Noise (matched-magnitude perturbation) intervention hook
# ---------------------------------------------------------------------------


class NoiseInterventionHook:
    """Context manager that adds Gaussian noise directly to vision encoder output.

    The noise magnitude is calibrated to match the L2 perturbation introduced
    by SAE passthrough on the same model, enabling a controlled comparison:
    if noise achieves similar DCR changes as SAE passthrough, then the SAE
    bottleneck effect is simply a generic perturbation artefact.

    Args:
        adapter: A loaded ``FullVLMAdapter``.
        noise_std: Standard deviation of Gaussian noise to add per-element.
            If ``None``, a default matched to typical SAE error is used.
        sae: Optional SAE used to *calibrate* noise magnitude. When provided,
            one forward pass through the SAE is performed on the first batch
            to measure the reconstruction error L2, and subsequent batches
            use Gaussian noise with the same per-element std.
    """

    def __init__(
        self,
        adapter: FullVLMAdapter,
        noise_std: float | None = None,
        sae: BaseSAE | None = None,
    ) -> None:
        self._adapter = adapter
        self._noise_std = noise_std
        self._sae = sae
        self._calibrated = noise_std is not None
        self._hook_handle: Any = None

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        use_pooler = (
            hasattr(output, "pooler_output")
            and output.pooler_output is not None
        )
        if use_pooler:
            hidden = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        original_dtype = hidden.dtype

        # Auto-calibrate noise from SAE reconstruction error on first batch
        if not self._calibrated and self._sae is not None:
            flat = hidden.reshape(-1, hidden.shape[-1])
            sae_dtype = next(self._sae.parameters()).dtype
            with torch.no_grad():
                acts = self._sae.encode(flat.to(dtype=sae_dtype))
                recon = self._sae.decode(acts)
            error = (recon.float() - flat.float())
            self._noise_std = float(error.std().item())
            self._calibrated = True
            log.info(
                "Noise hook calibrated from SAE error: std=%.6f (L2/elem=%.6f)",
                self._noise_std,
                error.pow(2).mean().sqrt().item(),
            )

        std = self._noise_std if self._noise_std is not None else 0.01
        noise = torch.randn_like(hidden.float()) * std
        modified_hidden = (hidden.float() + noise).to(dtype=original_dtype)

        log.debug(
            "Noise hook fired: batch=%d, tokens=%d, noise_std=%.6f",
            hidden.shape[0],
            hidden.shape[1] if hidden.ndim > 1 else 0,
            std,
        )

        if use_pooler:
            output.pooler_output = modified_hidden
            return output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = modified_hidden
            return output
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden

    def __enter__(self) -> "NoiseInterventionHook":
        vision_module = self._adapter.get_vision_module()
        self._hook_handle = vision_module.register_forward_hook(self._hook_fn)
        log.info("Registered noise intervention hook (std=%s)", self._noise_std)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            log.info("Removed noise intervention hook")
        return False


# ---------------------------------------------------------------------------
# LEACE intervention hook
# ---------------------------------------------------------------------------


class LEACEInterventionHook:
    """Context manager that applies LEACE concept erasure via a forward hook.

    Unlike :class:`SAEInterventionHook`, LEACE operates directly on the
    vision encoder output without an SAE bottleneck.  The eraser projects
    out the linear subspace most predictive of a target demographic concept,
    so no reconstruction error is introduced by an intermediate autoencoder.

    The eraser is a learned linear projection that can be applied per-token
    to the (batch, seq_len, hidden_dim) output of the vision encoder.

    Usage::

        eraser = fit_leace_eraser(hdf5_path, split, attribute, labels, ...)
        with LEACEInterventionHook(adapter, eraser):
            modified_texts = adapter.generate(images, prompts)
    """

    def __init__(
        self,
        adapter: FullVLMAdapter,
        eraser: Any,  # LeaceEraser
    ) -> None:
        self._adapter = adapter
        self._eraser = eraser
        self._hook_handle: Any = None

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """Intercept vision encoder output and apply LEACE erasure per-token."""
        use_pooler = (
            hasattr(output, "pooler_output")
            and output.pooler_output is not None
        )
        if use_pooler:
            hidden = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        original_dtype = hidden.dtype
        original_shape = hidden.shape
        hidden_dim = hidden.shape[-1]

        # Flatten to (N, D) for the eraser, which expects (N, D).
        # Handles both 3D (batch, seq, dim) and 2D (total_tokens, dim).
        flat = hidden.reshape(-1, hidden_dim).float()

        # LeaceEraser is a frozen dataclass -- move data to its device,
        # apply erasure, then move back.
        eraser_device = self._eraser.proj_left.device
        with torch.no_grad():
            erased_flat = self._eraser(flat.to(eraser_device)).to(flat.device)

        modified_hidden = erased_flat.to(dtype=original_dtype).reshape(original_shape)

        log.debug(
            "LEACE hook fired: tokens=%d, L2_change=%.6f",
            flat.shape[0],
            (modified_hidden.float() - hidden.float()).pow(2).mean().item(),
        )

        if use_pooler:
            output.pooler_output = modified_hidden
            return output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = modified_hidden
            return output
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden

    def __enter__(self) -> "LEACEInterventionHook":
        vision_module = self._adapter.get_vision_module()
        self._hook_handle = vision_module.register_forward_hook(self._hook_fn)
        log.info("Registered LEACE intervention hook")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            log.info("Removed LEACE intervention hook")
        return False


# ---------------------------------------------------------------------------
# S&P Top-K intervention hook
# ---------------------------------------------------------------------------


class SPTopKInterventionHook:
    """Context manager that applies S&P Top-K orthogonal projection via a forward hook.

    Like :class:`LEACEInterventionHook`, this operates directly on the vision
    encoder output without an SAE bottleneck at inference time.  The projection
    matrix V = I - alpha * a*a^T / ||a||^2 is pre-computed from SAE encoder
    weights and feature importance, then applied per-token.

    Usage::

        V = fit_sp_projection(sae, hdf5_path, split, attribute, labels, ...)
        with SPTopKInterventionHook(adapter, V):
            modified_texts = adapter.generate(images, prompts)
    """

    def __init__(
        self,
        adapter: FullVLMAdapter,
        projection_matrix: torch.Tensor,
    ) -> None:
        self._adapter = adapter
        self._V = projection_matrix  # (input_size, input_size) on CPU or GPU
        self._hook_handle: Any = None

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """Intercept vision encoder output and apply S&P projection per-token."""
        use_pooler = (
            hasattr(output, "pooler_output")
            and output.pooler_output is not None
        )
        if use_pooler:
            hidden = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        original_dtype = hidden.dtype
        original_shape = hidden.shape
        hidden_dim = hidden.shape[-1]

        flat = hidden.reshape(-1, hidden_dim).float()

        with torch.no_grad():
            V = self._V.to(device=flat.device, dtype=flat.dtype)
            projected_flat = flat @ V.T

        modified_hidden = projected_flat.to(dtype=original_dtype).reshape(original_shape)

        log.debug(
            "S&P hook fired: tokens=%d, L2_change=%.6f",
            flat.shape[0],
            (modified_hidden.float() - hidden.float()).pow(2).mean().item(),
        )

        if use_pooler:
            output.pooler_output = modified_hidden
            return output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = modified_hidden
            return output
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden

    def __enter__(self) -> "SPTopKInterventionHook":
        vision_module = self._adapter.get_vision_module()
        self._hook_handle = vision_module.register_forward_hook(self._hook_fn)
        log.info("Registered S&P Top-K intervention hook")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            log.info("Removed S&P Top-K intervention hook")
        return False


def fit_sp_projection(
    sae: BaseSAE,
    hdf5_path: str,
    split: str,
    attribute: str,
    labels: dict[str, np.ndarray],
    sdf_dir: str | None = None,
    k: int = 64,
    alpha: float = 1.0,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Fit S&P Top-K projection matrix from HDF5 VE embeddings + SAE.

    The matrix V = I - alpha * a*a^T / ||a||^2 projects out the concept
    direction identified by the SAE encoder weights and feature importance.

    Returns:
        V: (input_size, input_size) projection matrix as a torch Tensor.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    from src.intervene import compute_sp_projection, select_features_by_variance

    target_labs = labels[attribute]

    log.info("Fitting S&P projection for '%s' (k=%d, alpha=%.2f) ...", attribute, k, alpha)

    # Stream through HDF5 to get mean-pooled raw embeddings + SAE activations
    raw_chunks = []
    act_chunks = []
    d_hidden = sae.hidden_size
    sae_dtype = next(sae.parameters()).dtype

    with h5py.File(hdf5_path, "r") as f:
        ds = f[split]["encoded"]
        n = min(ds.shape[0], len(target_labs))
        n_tok = ds.shape[1]
        e_dim = ds.shape[2]

        for i in tqdm(range(0, n, batch_size), desc="Loading+encoding for S&P"):
            end = min(i + batch_size, n)
            batch = np.array(ds[i:end])  # (bs, seq, dim)
            raw_chunks.append(batch.mean(axis=1))
            flat = batch.reshape(-1, e_dim)
            x = torch.from_numpy(flat).to(device=device, dtype=sae_dtype)
            with torch.no_grad():
                acts = sae.encode(x)
            bs = end - i
            per_sample = acts.cpu().float().numpy().reshape(bs, n_tok, d_hidden).mean(axis=1)
            act_chunks.append(per_sample)

    sae_acts = np.concatenate(act_chunks)
    del act_chunks

    # Select features: use SDFs if available, otherwise variance-based
    if sdf_dir is not None:
        sdf_path = Path(sdf_dir) / f"sdfs_{attribute}.npz"
        if sdf_path.exists():
            sdf_data = np.load(sdf_path)
            all_sdfs = np.unique(np.concatenate([sdf_data[key] for key in sdf_data.files]))
            feature_indices = all_sdfs[:k] if k <= len(all_sdfs) else all_sdfs
            log.info("  Using %d SDF features (of %d available)", len(feature_indices), len(all_sdfs))
        else:
            log.warning("  SDF file not found: %s, falling back to variance selection", sdf_path)
            feature_indices = select_features_by_variance(sae_acts, target_labs[:n], k)
    else:
        feature_indices = select_features_by_variance(sae_acts, target_labs[:n], k)
        log.info("  Selected %d features by variance", len(feature_indices))

    # Compute the projection matrix
    V_np = compute_sp_projection(
        sae, feature_indices, sae_acts[:n], target_labs[:n], alpha,
    )

    # Sanity check: cosine similarity
    raw_pooled = np.concatenate(raw_chunks)
    projected = raw_pooled[:n] @ V_np.T
    import torch.nn.functional as _F
    cos_sim = _F.cosine_similarity(
        torch.from_numpy(raw_pooled[:n]).float(),
        torch.from_numpy(projected).float(),
        dim=-1,
    ).mean().item()
    log.info("  cosine_sim(original, projected) = %.4f", cos_sim)
    del raw_pooled, raw_chunks, sae_acts, projected

    return torch.from_numpy(V_np).float()


def fit_leace_eraser(
    hdf5_path: str,
    split: str,
    attribute: str,
    labels: dict[str, np.ndarray],
    batch_size: int = 256,
) -> Any:
    """Fit a LEACE eraser on mean-pooled VE embeddings from HDF5.

    The eraser learns a projection that removes the linear subspace most
    predictive of *attribute* from the embedding space.  It can then be
    applied per-token to the vision encoder output during VLM generation.

    Returns:
        Fitted ``LeaceEraser`` instance.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    from concept_erasure import LeaceEraser

    target_labs = labels[attribute]

    log.info("Fitting LEACE eraser for '%s' from %s ...", attribute, hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        ds = f[split]["encoded"]
        n = min(ds.shape[0], len(target_labs))

        raw_chunks = []
        ndim = len(ds.shape)
        for i in tqdm(range(0, n, batch_size), desc="Mean-pooling for LEACE"):
            end = min(i + batch_size, n)
            batch = np.array(ds[i:end])
            if ndim == 3:
                raw_chunks.append(batch.mean(axis=1))  # (bs, seq, dim) -> (bs, dim)
            else:
                raw_chunks.append(batch)  # already (bs, dim)

    raw_pooled = np.concatenate(raw_chunks)
    del raw_chunks

    X_t = torch.from_numpy(raw_pooled).float()
    Z_t = torch.from_numpy(target_labs[:n]).long()

    log.info(
        "  Fitting eraser: %d samples, embed_dim=%d, %d classes",
        n, X_t.shape[1], len(np.unique(target_labs[:n])),
    )
    eraser = LeaceEraser.fit(X_t, Z_t)

    # Quick sanity check
    import torch.nn.functional as _F
    cos_sim = _F.cosine_similarity(X_t, eraser(X_t), dim=-1).mean().item()
    log.info("  cosine_sim(original, erased) = %.4f", cos_sim)

    return eraser


# ---------------------------------------------------------------------------
# FairFace data loading
# ---------------------------------------------------------------------------


def _load_fairface_sample(
    n_samples: int,
    attribute: str,
) -> tuple[list[Image.Image], dict[str, np.ndarray], np.ndarray]:
    """Load a stratified sample of FairFace images.

    Images are loaded from the HuggingFace ``HuggingFaceM4/FairFace``
    dataset.  Stratification is performed on *attribute* to ensure
    balanced representation of all demographic classes.

    Args:
        n_samples: Number of images to sample.
        attribute: Attribute to stratify by (``'race'``, ``'gender'``,
            or ``'age'``).

    Returns:
        images: List of PIL images.
        labels: Dict mapping attribute name to ``(n_samples,)`` int array.
        indices: Original dataset indices for the sampled images.
    """
    from datasets import load_dataset

    from src.setup_datasets.ve_latent_dataset import FairFaceImageDataset

    log.info("Loading FairFace from HuggingFace ...")
    hf_ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train")
    ff_dataset = FairFaceImageDataset(hf_ds)
    n_total = len(ff_dataset)

    # Access column directly for fast stratification.
    strat_labels = np.array(hf_ds[attribute])

    if n_samples >= n_total:
        indices = np.arange(n_total)
        log.info("Using all %d samples (n_samples >= dataset size)", n_total)
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_samples,
            random_state=42,
        )
        indices, _ = next(
            splitter.split(np.zeros(n_total), strat_labels),
        )
        log.info(
            "Sampled %d images (stratified by %s)",
            len(indices),
            attribute,
        )

    # Load images and properly-remapped labels for selected samples.
    images: list[Image.Image] = []
    label_lists: dict[str, list[int]] = {"age": [], "gender": [], "race": []}

    for idx in tqdm(indices, desc="Loading FairFace images"):
        sample = ff_dataset[int(idx)]
        images.append(sample["image"])
        for attr in label_lists:
            label_lists[attr].append(sample[attr])

    label_arrays = {k: np.array(v) for k, v in label_lists.items()}

    # Log class distribution for each attribute.
    for attr, arr in label_arrays.items():
        names_map = LABEL_NAMES.get(attr, {})
        unique, counts = np.unique(arr, return_counts=True)
        dist = ", ".join(f"{names_map.get(int(c), str(c))}={n}" for c, n in zip(unique, counts))
        log.info("  %s distribution: %s", attr, dist)

    return images, label_arrays, indices


# ---------------------------------------------------------------------------
# Main causal tracing pipeline
# ---------------------------------------------------------------------------


def run_causal_tracing(
    adapter: FullVLMAdapter,
    hook_ctx: Any,
    images: list[Image.Image],
    labels: dict[str, np.ndarray],
    *,
    task: str,
    attribute: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Generate text with and without an intervention hook on the vision encoder.

    For each batch of images the function generates text twice:

    1. **Baseline** -- original vision encoder output (no hook).
    2. **Modified** -- the provided ``hook_ctx`` context manager is entered
       to intercept and modify vision encoder outputs before they reach
       the language model.

    The ``hook_ctx`` can be any context manager -- :class:`SAEInterventionHook`
    for SAE-based interventions, :class:`LEACEInterventionHook` for LEACE
    concept erasure, or any future hook type.

    Args:
        adapter: Loaded full VLM adapter.
        hook_ctx: Context manager that installs a forward hook on the vision
            encoder when entered and removes it when exited.
        images: List of PIL images.
        labels: Dict mapping attribute name to ``(N,)`` int array.
        task: ``'caption'`` or ``'vqa'``.
        attribute: Target demographic attribute being intervened on.
        batch_size: Number of images per generation batch.
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        List of per-sample result dicts with original and modified
        generations.
    """
    n = len(images)
    results: list[dict] = []

    def _make_sample_meta(idx: int) -> dict:
        sample_labels = {attr: int(labels[attr][idx]) for attr in labels}
        return {
            "index": int(idx),
            "labels": sample_labels,
            "labels_readable": {attr: LABEL_NAMES.get(attr, {}).get(v, str(v)) for attr, v in sample_labels.items()},
        }

    # ---- Captioning task --------------------------------------------------

    if task == "caption":
        log.info(
            "Captioning causal tracing: %d images, batch_size=%d",
            n,
            batch_size,
        )
        t_start = time.time()

        for i in tqdm(range(0, n, batch_size), desc="Captioning"):
            end = min(i + batch_size, n)
            batch_images = images[i:end]
            prompts = ["caption en"] * len(batch_images)

            original_texts = adapter.generate(
                batch_images,
                prompts,
                max_new_tokens,
            )
            with hook_ctx:
                modified_texts = adapter.generate(
                    batch_images,
                    prompts,
                    max_new_tokens,
                )

            for j, (orig, mod) in enumerate(
                zip(original_texts, modified_texts),
            ):
                rec = _make_sample_meta(i + j)
                rec.update(
                    task="caption",
                    original_text=orig,
                    modified_text=mod,
                )
                results.append(rec)

        elapsed = time.time() - t_start
        log.info(
            "Captioning complete: %d samples in %.1fs (%.2f s/sample)",
            n,
            elapsed,
            elapsed / max(n, 1),
        )

    # ---- VQA task ---------------------------------------------------------

    elif task == "vqa":
        demo_questions = DEMOGRAPHIC_PROBES.get(attribute, [])
        all_questions: list[tuple[str, str]] = [(q, "demographic") for q in demo_questions] + [
            (q, "control") for q in CONTROL_PROBES
        ]

        log.info(
            "VQA causal tracing: %d images, %d questions, batch_size=%d",
            n,
            len(all_questions),
            batch_size,
        )

        # Pre-create result containers for each sample.
        for idx in range(n):
            rec = _make_sample_meta(idx)
            rec.update(task="vqa", questions={})
            results.append(rec)

        t_start = time.time()
        for q_idx, (question, q_type) in enumerate(all_questions):
            log.info(
                "  Question %d/%d [%s]: %s",
                q_idx + 1,
                len(all_questions),
                q_type,
                question,
            )

            all_original: list[str] = []
            all_modified: list[str] = []

            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch_images = images[i:end]
                prompts = [question] * len(batch_images)

                original_texts = adapter.generate(
                    batch_images,
                    prompts,
                    max_new_tokens,
                )
                with hook_ctx:
                    modified_texts = adapter.generate(
                        batch_images,
                        prompts,
                        max_new_tokens,
                    )

                all_original.extend(original_texts)
                all_modified.extend(modified_texts)

            for idx in range(n):
                results[idx]["questions"][question] = {
                    "type": q_type,
                    "original": all_original[idx],
                    "modified": all_modified[idx],
                }

        elapsed = time.time() - t_start
        total_gens = n * len(all_questions) * 2
        log.info(
            "VQA complete: %d total generations in %.1fs (%.2f s/gen)",
            total_gens,
            elapsed,
            elapsed / max(total_gens, 1),
        )

    else:
        raise ValueError(
            f"Unknown task: {task!r}. Expected 'caption' or 'vqa'.",
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
        description="VLM causal tracing with demographic feature intervention",
    )
    parser.add_argument(
        "--sae-checkpoint", default=None,
        help="Path to trained SAE checkpoint (required for SAE modes)",
    )
    parser.add_argument(
        "--sdf-dir", default=None,
        help="Directory with sdfs_<attr>.npz files (required for SAE modes)",
    )
    parser.add_argument(
        "--hdf5", default=None,
        help="VE latent HDF5 file (required for leace/sp_topk mode)",
    )
    parser.add_argument(
        "--sp-k", type=int, default=64,
        help="Number of features for S&P Top-K projection (sp_topk mode)",
    )
    parser.add_argument(
        "--output-dir", default="results/causal_tracing",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--vlm", default="paligemma2",
        choices=list(FULL_VLM_ADAPTERS.keys()),
    )
    parser.add_argument(
        "--task", required=True, choices=["caption", "vqa"],
    )
    parser.add_argument(
        "--attribute", default="race", choices=["race", "gender", "age"],
    )
    parser.add_argument(
        "--mode", default="suppression",
        choices=["suppression", "amplification", "attenuation", "passthrough", "leace", "sp_topk", "random_suppression", "noise"],
        help="Intervention mode. leace/sp_topk = linear projection. random_suppression = SAE with random non-SDF features. noise = matched-magnitude gaussian perturbation.",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--noise-std", type=float, default=None,
        help="Manual noise std for noise mode (bypasses SAE calibration). Use for VLMs without a trained SAE.",
    )
    parser.add_argument(
        "--effective-topk", type=int, default=None,
        help="Override SAE top_k at inference time (for quality ablation). Lower = sparser = worse reconstruction.",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
            "task": args.task,
            "attribute": args.attribute,
            "intervention_mode": args.mode,
            "alpha": args.alpha,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "vlm": args.vlm,
            "sp_k": args.sp_k,
            "effective_topk": args.effective_topk,
        }
        _wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{args.task}_{args.attribute}_{args.mode}",
            config=wandb_config,
            tags=[args.task, args.attribute, args.mode, args.vlm],
            job_type="causal_tracing",
        )
        log.info("W&B run initialised: %s", _wandb_run.url)

    is_linear_proj = args.mode in ("leace", "sp_topk", "noise")

    # -- Load VLM -----------------------------------------------------------
    log.info("Loading VLM: %s", args.vlm)
    adapter = FULL_VLM_ADAPTERS[args.vlm]()
    adapter.load(device)

    # -- Load FairFace sample -----------------------------------------------
    images, labels, sample_indices = _load_fairface_sample(
        args.n_samples, args.attribute,
    )

    # -- Build intervention hook --------------------------------------------
    if args.mode == "leace":
        if not args.hdf5:
            parser.error("--hdf5 is required for leace mode")
        import h5py
        import hdf5plugin  # noqa: F401

        with h5py.File(args.hdf5, "r") as f:
            hdf5_labels = {
                key: np.array(f["validation"]["labels"][key])
                for key in f["validation"]["labels"].keys()
            }
        eraser = fit_leace_eraser(
            args.hdf5, "validation", args.attribute, hdf5_labels,
        )
        hook_ctx = LEACEInterventionHook(adapter, eraser)
        n_target_features = 0
        config_extra: dict = {"hdf5": args.hdf5}

    elif args.mode == "sp_topk":
        if not args.hdf5:
            parser.error("--hdf5 is required for sp_topk mode")
        if not args.sae_checkpoint:
            parser.error("--sae-checkpoint is required for sp_topk mode")
        import h5py
        import hdf5plugin  # noqa: F401

        log.info("Loading SAE from %s", args.sae_checkpoint)
        sae = load_sae_from_checkpoint(args.sae_checkpoint, device)

        with h5py.File(args.hdf5, "r") as f:
            hdf5_labels = {
                key: np.array(f["validation"]["labels"][key])
                for key in f["validation"]["labels"].keys()
            }
        V = fit_sp_projection(
            sae, args.hdf5, "validation", args.attribute, hdf5_labels,
            sdf_dir=args.sdf_dir, k=args.sp_k, alpha=args.alpha if args.alpha > 0 else 1.0,
            batch_size=256, device=device,
        )
        hook_ctx = SPTopKInterventionHook(adapter, V)
        n_target_features = args.sp_k
        config_extra = {
            "hdf5": args.hdf5,
            "sae_checkpoint": args.sae_checkpoint,
            "sp_k": args.sp_k,
        }

    elif args.mode == "noise":
        # P1c: Matched-noise baseline -- add Gaussian noise to VE output
        # calibrated to match SAE reconstruction error magnitude
        sae_for_calib = None
        manual_std = args.noise_std
        if manual_std is not None:
            log.info("Using manual noise std: %.6f", manual_std)
        elif args.sae_checkpoint:
            log.info("Loading SAE for noise calibration from %s", args.sae_checkpoint)
            sae_for_calib = load_sae_from_checkpoint(args.sae_checkpoint, device)
        hook_ctx = NoiseInterventionHook(adapter, noise_std=manual_std, sae=sae_for_calib)
        n_target_features = 0
        calib_mode = "manual" if manual_std else ("sae_matched" if sae_for_calib else "default")
        config_extra = {"noise_calibration": calib_mode, "noise_std": manual_std}

    elif args.mode == "random_suppression":
        # P1b: Suppress random non-SDF features through SAE
        if not args.sae_checkpoint:
            parser.error("--sae-checkpoint is required for random_suppression mode")
        if not args.sdf_dir:
            parser.error("--sdf-dir is required for random_suppression mode")

        log.info("Loading SAE from %s", args.sae_checkpoint)
        sae = load_sae_from_checkpoint(args.sae_checkpoint, device)

        # Load actual SDFs to exclude them
        sdf_path = Path(args.sdf_dir) / f"sdfs_{args.attribute}.npz"
        if not sdf_path.exists():
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        sdf_data = np.load(sdf_path)
        actual_sdfs = set(
            np.unique(np.concatenate([sdf_data[k] for k in sdf_data.files])).tolist()
        )
        n_sdfs = len(actual_sdfs)

        # Select random non-SDF features
        all_features = set(range(sae.hidden_size))
        non_sdf_pool = sorted(all_features - actual_sdfs)
        rng = np.random.default_rng(42)
        random_features = rng.choice(non_sdf_pool, size=min(n_sdfs, len(non_sdf_pool)), replace=False)
        log.info(
            "Selected %d random non-SDF features (excluded %d actual SDFs from %d total)",
            len(random_features), n_sdfs, sae.hidden_size,
        )

        hook_ctx = SAEInterventionHook(
            adapter, sae, random_features, "suppression", 0.0,
        )
        n_target_features = int(len(random_features))
        config_extra = {
            "sae_checkpoint": args.sae_checkpoint,
            "random_seed": 42,
            "excluded_sdfs": n_sdfs,
        }

    else:
        if not args.sae_checkpoint:
            parser.error("--sae-checkpoint is required for SAE modes")
        if not args.sdf_dir:
            parser.error("--sdf-dir is required for SAE modes")

        log.info("Loading SAE from %s", args.sae_checkpoint)
        sae = load_sae_from_checkpoint(args.sae_checkpoint, device)

        sdf_path = Path(args.sdf_dir) / f"sdfs_{args.attribute}.npz"
        if not sdf_path.exists():
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        sdf_data = np.load(sdf_path)
        target_features = np.unique(
            np.concatenate([sdf_data[k] for k in sdf_data.files]),
        )
        log.info("Loaded %d target SDFs for '%s'", len(target_features), args.attribute)

        hook_ctx = SAEInterventionHook(
            adapter, sae, target_features, args.mode, args.alpha,
        )
        n_target_features = int(len(target_features))
        config_extra = {"sae_checkpoint": args.sae_checkpoint}

    # -- Override SAE top_k if requested (quality ablation) -----------------
    if args.effective_topk is not None and hasattr(hook_ctx, '_sae'):
        original_topk = hook_ctx._sae.top_k
        hook_ctx._sae.top_k = args.effective_topk
        log.info(
            "Overriding SAE top_k: %d -> %d (quality ablation)",
            original_topk, args.effective_topk,
        )

    # -- Run causal tracing -------------------------------------------------
    samples = run_causal_tracing(
        adapter=adapter,
        hook_ctx=hook_ctx,
        images=images,
        labels=labels,
        task=args.task,
        attribute=args.attribute,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # -- Save results -------------------------------------------------------
    output = {
        "_metadata": build_metadata(args),
        "config": {
            "task": args.task,
            "attribute": args.attribute,
            "intervention_mode": args.mode,
            "alpha": args.alpha,
            "n_samples": len(images),
            "n_target_features": n_target_features,
            "vlm": args.vlm,
            **config_extra,
        },
        "samples": samples,
    }

    fname = f"causal_{args.task}_{args.attribute}_{args.mode}.json"
    results_path = out_dir / fname
    with open(results_path, "w") as fp:
        json.dump(output, fp, indent=2, cls=NumpyEncoder)
    log.info("Results saved to %s", results_path)

    # -- Inline evaluation & wandb logging ----------------------------------
    from src.causal_eval import (
        CaptionCausalMetrics,
        VQACausalMetrics,
        aggregate_results,
    )

    eval_results: dict[str, Any] = {}

    if args.task == "caption":
        log.info("=== Inline Caption Evaluation ===")
        original = [s["original_text"] for s in samples]
        modified = [s["modified_text"] for s in samples]
        caption_metrics = CaptionCausalMetrics()
        eval_results["caption_metrics"] = caption_metrics.compute(
            original, modified, args.attribute,
        )
    elif args.task == "vqa":
        log.info("=== Inline VQA Evaluation ===")
        vqa_metrics = VQACausalMetrics()
        eval_results["vqa_metrics"] = vqa_metrics.compute(samples, args.attribute)

    eval_results["aggregate"] = aggregate_results(samples, args.attribute, args.task)

    # Save evaluation JSON alongside the raw results
    eval_output = {
        "_metadata": build_metadata(args),
        "source_file": str(results_path),
        "config": output["config"],
        "evaluation": eval_results,
    }
    eval_path = out_dir / f"eval_{results_path.stem}.json"
    with open(eval_path, "w") as fp:
        json.dump(eval_output, fp, indent=2, cls=NumpyEncoder)
    log.info("Evaluation results saved to %s", eval_path)

    # -- Log to wandb -------------------------------------------------------
    if _wandb_run is not None:
        import wandb

        wandb_metrics: dict[str, Any] = {
            "n_samples": len(images),
            "n_target_features": n_target_features,
        }

        if args.task == "caption":
            cm = eval_results.get("caption_metrics", {})
            for attr in ("race", "gender", "age"):
                dcr_entry = cm.get(f"dcr_{attr}", {})
                wandb_metrics[f"dcr_{attr}_original"] = dcr_entry.get("dcr_original", 0)
                wandb_metrics[f"dcr_{attr}_modified"] = dcr_entry.get("dcr_modified", 0)
                wandb_metrics[f"dcr_{attr}_delta"] = dcr_entry.get("dcr_delta", 0)
            bs = cm.get("bertscore", {})
            if bs:
                wandb_metrics["bertscore_f1"] = bs.get("f1_mean", 0)
                wandb_metrics["bertscore_f1_std"] = bs.get("f1_std", 0)

        elif args.task == "vqa":
            vm = eval_results.get("vqa_metrics", {})
            wandb_metrics["causal_effect"] = vm.get("causal_effect", 0)
            for q_name, q_data in vm.get("per_question", {}).items():
                q_key = q_name[:40].replace(" ", "_").replace("?", "").replace("'", "")
                wandb_metrics[f"vqa/{q_key}/acc_original"] = q_data.get("acc_original", 0)
                wandb_metrics[f"vqa/{q_key}/acc_modified"] = q_data.get("acc_modified", 0)
                wandb_metrics[f"vqa/{q_key}/delta"] = q_data.get("delta", 0)
            for q_type, t_data in vm.get("per_type", {}).items():
                wandb_metrics[f"vqa_type/{q_type}/mean_delta"] = t_data.get("mean_delta", 0)

        wandb.log(wandb_metrics)

        # Log result artifacts
        artifact = wandb.Artifact(
            name=f"causal_{args.task}_{args.attribute}_{args.mode}",
            type="results",
        )
        artifact.add_file(str(results_path))
        artifact.add_file(str(eval_path))
        _wandb_run.log_artifact(artifact)

        _wandb_run.finish()
        log.info("W&B run finished")


if __name__ == "__main__":
    main()
