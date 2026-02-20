"""Unit tests for model architectures (nn.Module classes).

Verifies forward pass shapes, encode/decode consistency, and build_sae factory.
"""

import pytest
import torch

from src.models.linear_probes import LinearProbe
from src.models.sparse_autoencoder import (
    BaseSAE,
    BatchTopKSAE,
    JumpReLUSAE,
    TopKSAE,
    VanillaSAE,
    build_sae,
)


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------


class TestLinearProbe:
    def test_output_shapes(self) -> None:
        probe = LinearProbe(input_dim=64, age_classes=9, gender_classes=2, race_classes=7)
        x = torch.randn(8, 64)
        age, gender, race = probe(x)
        assert age.shape == (8, 9)
        assert gender.shape == (8, 2)
        assert race.shape == (8, 7)

    def test_single_sample(self) -> None:
        probe = LinearProbe(input_dim=32, age_classes=3, gender_classes=2, race_classes=4)
        x = torch.randn(1, 32)
        age, gender, race = probe(x)
        assert age.shape == (1, 3)
        assert gender.shape == (1, 2)
        assert race.shape == (1, 4)


# ---------------------------------------------------------------------------
# SAE variants
# ---------------------------------------------------------------------------


@pytest.fixture(params=["vanilla", "topk", "batchtopk", "jumprelu"])
def sae_variant(request: pytest.FixtureRequest) -> str:
    return request.param


class TestSAEVariants:
    INPUT_SIZE = 32
    HIDDEN_SIZE = 64
    BATCH_SIZE = 4

    def _build(self, variant: str) -> BaseSAE:
        kwargs = {"input_size": self.INPUT_SIZE, "hidden_size": self.HIDDEN_SIZE}
        if variant == "vanilla":
            kwargs["l1_coeff"] = 1e-4
        elif variant in ("topk", "batchtopk"):
            kwargs["top_k"] = 8
            kwargs["top_k_aux"] = 16
        elif variant == "jumprelu":
            kwargs["bandwidth"] = 0.001
            kwargs["l0_coeff"] = 1e-4
        return build_sae(variant, **kwargs)

    def test_forward_output_keys(self, sae_variant: str) -> None:
        model = self._build(sae_variant)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        out = model(x)
        expected_keys = {
            "reconstructed",
            "feature_acts",
            "loss",
            "l2_loss",
            "sparsity_loss",
            "aux_loss",
            "l0_norm",
            "num_dead_features",
        }
        assert set(out.keys()) == expected_keys

    def test_forward_shapes(self, sae_variant: str) -> None:
        model = self._build(sae_variant)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        out = model(x)
        assert out["reconstructed"].shape == x.shape
        assert out["feature_acts"].shape == (self.BATCH_SIZE, self.HIDDEN_SIZE)
        assert out["loss"].ndim == 0  # scalar

    def test_encode_shape(self, sae_variant: str) -> None:
        model = self._build(sae_variant)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        acts = model.encode(x)
        assert acts.shape == (self.BATCH_SIZE, self.HIDDEN_SIZE)

    def test_decode_shape(self, sae_variant: str) -> None:
        model = self._build(sae_variant)
        z = torch.randn(self.BATCH_SIZE, self.HIDDEN_SIZE)
        recon = model.decode(z)
        assert recon.shape == (self.BATCH_SIZE, self.INPUT_SIZE)

    def test_encode_decode_roundtrip(self, sae_variant: str) -> None:
        """Encode then decode should produce the same shape as input."""
        model = self._build(sae_variant)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        acts = model.encode(x)
        recon = model.decode(acts)
        assert recon.shape == x.shape

    def test_loss_is_finite(self, sae_variant: str) -> None:
        model = self._build(sae_variant)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        out = model(x)
        assert torch.isfinite(out["loss"]).all()

    def test_backward_runs(self, sae_variant: str) -> None:
        """Verify gradients flow through the loss."""
        model = self._build(sae_variant)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        out = model(x)
        out["loss"].backward()
        # At minimum, encoder weights should have gradients
        assert model.W_enc.grad is not None
        assert model.W_enc.grad.shape == model.W_enc.shape


class TestBuildSAE:
    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown SAE variant"):
            build_sae("nonexistent", input_size=32, hidden_size=64)

    def test_all_registered_variants(self) -> None:
        from src.models.sparse_autoencoder import SAE_VARIANTS

        for name in SAE_VARIANTS:
            model = build_sae(name, input_size=16, hidden_size=32)
            assert isinstance(model, BaseSAE)
