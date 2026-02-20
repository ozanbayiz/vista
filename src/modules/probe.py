from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from src.models.linear_probes import LinearProbe

TASKS = ("age", "gender", "race")


class ProbeModule(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        age_classes: int = 9,
        gender_classes: int = 2,
        race_classes: int = 7,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearProbe(input_dim, age_classes, gender_classes, race_classes)
        self.criterion = nn.CrossEntropyLoss()

        num_classes = {"age": age_classes, "gender": gender_classes, "race": race_classes}
        self.val_acc = nn.ModuleDict(
            {task: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes[task]) for task in TASKS}
        )
        self.val_f1 = nn.ModuleDict(
            {
                task: torchmetrics.F1Score(task="multiclass", num_classes=num_classes[task], average="macro")
                for task in TASKS
            }
        )

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    def _shared_step(self, batch: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features, labels_dict = batch
        features = self._pool(features)
        age_logits, gender_logits, race_logits = self.model(features)
        logits = {"age": age_logits, "gender": gender_logits, "race": race_logits}
        # Only compute losses for tasks present in the label dict.
        active_tasks = [t for t in TASKS if t in labels_dict]
        losses = {task: self.criterion(logits[task], labels_dict[task]) for task in active_tasks}
        return logits, losses

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits, losses = self._shared_step(batch)
        total_loss = sum(losses.values())
        for task in losses:
            self.log(f"train/{task}_loss", losses[task])
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        logits, losses = self._shared_step(batch)
        _, labels_dict = batch
        total_loss = sum(losses.values())
        for task in losses:
            self.log(f"val/{task}_loss", losses[task], sync_dist=True)
            preds = logits[task].argmax(dim=-1)
            self.val_acc[task].update(preds, labels_dict[task])
            self.val_f1[task].update(preds, labels_dict[task])
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        for task in TASKS:
            acc_metric = self.val_acc[task]
            f1_metric = self.val_f1[task]
            # Only log if the metric was updated during this epoch.
            if acc_metric._update_called if hasattr(acc_metric, "_update_called") else True:
                self.log(f"val/{task}_acc", acc_metric.compute(), prog_bar=("race" in task), sync_dist=True)
                self.log(f"val/{task}_f1", f1_metric.compute(), sync_dist=True)
            acc_metric.reset()
            f1_metric.reset()

    def on_after_backward(self) -> None:
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
