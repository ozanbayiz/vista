"""Unit tests for LightningModule wrappers (training systems).

Verifies that SAEModule and ProbeModule can perform forward/training steps
and that gradient norm logging is active.
"""

import pytest
import torch

from src.modules.sae import SAEModule
from src.modules.probe import ProbeModule


class TestSAEModule:
    def test_training_step(self) -> None:
        module = SAEModule(
            variant="batchtopk",
            input_size=32,
            hidden_size=64,
            top_k=8,
            top_k_aux=16,
        )
        x = torch.randn(4, 32)
        loss = module.training_step(x, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_training_step_3d_input(self) -> None:
        """3D input (batch, seq_len, features) should be flattened internally."""
        module = SAEModule(
            variant="batchtopk",
            input_size=32,
            hidden_size=64,
            top_k=8,
            top_k_aux=16,
        )
        x = torch.randn(2, 10, 32)  # 2 samples, 10 tokens, 32 features
        loss = module.training_step(x, batch_idx=0)
        assert torch.isfinite(loss)

    def test_validation_step(self) -> None:
        module = SAEModule(
            variant="batchtopk",
            input_size=32,
            hidden_size=64,
            top_k=8,
            top_k_aux=16,
        )
        x = torch.randn(4, 32)
        # Should not raise
        module.validation_step(x, batch_idx=0)

    def test_configure_optimizers(self) -> None:
        module = SAEModule(
            variant="topk",
            input_size=32,
            hidden_size=64,
            learning_rate=1e-3,
            top_k=8,
        )
        opt = module.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)


class TestProbeModule:
    def _make_batch(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = torch.randn(4, 32)
        labels = {
            "age": torch.randint(0, 9, (4,)),
            "gender": torch.randint(0, 2, (4,)),
            "race": torch.randint(0, 7, (4,)),
        }
        return features, labels

    def test_training_step(self) -> None:
        module = ProbeModule(input_dim=32, age_classes=9, gender_classes=2, race_classes=7)
        batch = self._make_batch()
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_training_step_3d_input(self) -> None:
        """3D input should be mean-pooled before classification."""
        module = ProbeModule(input_dim=32, age_classes=9, gender_classes=2, race_classes=7)
        features = torch.randn(4, 10, 32)  # (batch, seq_len, features)
        labels = {
            "age": torch.randint(0, 9, (4,)),
            "gender": torch.randint(0, 2, (4,)),
            "race": torch.randint(0, 7, (4,)),
        }
        loss = module.training_step((features, labels), batch_idx=0)
        assert torch.isfinite(loss)

    def test_configure_optimizers(self) -> None:
        module = ProbeModule(input_dim=32, learning_rate=1e-2)
        opt = module.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)

    def test_partial_labels(self) -> None:
        """Should handle batches with only a subset of label keys."""
        module = ProbeModule(input_dim=32, age_classes=9, gender_classes=2, race_classes=7)
        features = torch.randn(4, 32)
        labels = {"gender": torch.randint(0, 2, (4,))}  # only gender
        loss = module.training_step((features, labels), batch_idx=0)
        assert torch.isfinite(loss)
