from __future__ import annotations

from typing import Union

import lightning as L
import torch
import torch.nn.functional as F

from src.models.sparse_autoencoder import build_sae

Batch = Union[torch.Tensor, tuple, list]


class SAEModule(L.LightningModule):
    def __init__(
        self,
        variant: str = "batchtopk",
        input_size: int = 768,
        hidden_size: int = 4096,
        learning_rate: float = 1e-4,
        **sae_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_sae(variant, input_size, hidden_size, **sae_kwargs)

    def _process_batch(self, batch: Batch) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if x.ndim == 3:
            # Flatten all patch tokens into one big batch for SAE training.
            # We keep all tokens (including CLS where present) since the SAE
            # should learn to reconstruct any token representation.
            x = x.reshape(-1, self.hparams.input_size)
        return x

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        x = self._process_batch(batch)
        out = self.model(x)
        self.log("train/loss", out["loss"], prog_bar=True)
        self.log("train/l2_loss", out["l2_loss"])
        self.log("train/sparsity_loss", out["sparsity_loss"])
        self.log("train/aux_loss", out["aux_loss"])
        self.log("train/l0_norm", out["l0_norm"], prog_bar=True)
        self.log("train/num_dead_features", out["num_dead_features"])
        return out["loss"]

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        x = self._process_batch(batch)
        out = self.model(x)
        cosine_sim = F.cosine_similarity(x.float(), out["reconstructed"].float(), dim=-1).mean()
        self.log("val/loss", out["loss"], prog_bar=True, sync_dist=True)
        self.log("val/l2_loss", out["l2_loss"], sync_dist=True)
        self.log("val/sparsity_loss", out["sparsity_loss"], sync_dist=True)
        self.log("val/aux_loss", out["aux_loss"], sync_dist=True)
        self.log("val/l0_norm", out["l0_norm"], sync_dist=True)
        self.log("val/num_dead_features", out["num_dead_features"], sync_dist=True)
        self.log("val/cosine_sim", cosine_sim, prog_bar=True, sync_dist=True)

    def on_after_backward(self) -> None:
        self.model.make_decoder_weights_and_grad_unit_norm()

        # Log gradient norm for training stability monitoring
        # (recommended by Google Deep Learning Tuning Playbook)
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm(2).item() ** 2
        total_norm = total_norm**0.5
        self.log("train/grad_norm", total_norm)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
