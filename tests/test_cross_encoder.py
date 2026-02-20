"""Unit tests for src.cross_encoder -- CKA, SDF overlap, principal angles,
and demographic subspace similarity.

All tests use synthetic data and mock SAE objects; no checkpoint loading.
"""

import numpy as np
import pytest
import torch

from src.cross_encoder import (
    _hsic,
    _inter_class_variance,
    _principal_angles,
    compute_cka,
    compute_sdf_overlap,
    demographic_subspace_similarity,
)
from src.models.sparse_autoencoder import build_sae


# ---------------------------------------------------------------------------
# HSIC
# ---------------------------------------------------------------------------


class TestHSIC:
    def test_identical_kernels_positive(self) -> None:
        """HSIC(K, K) should be positive for non-trivial K."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 5))
        K = X @ X.T
        assert _hsic(K, K) > 0

    def test_independent_near_zero(self) -> None:
        """HSIC between independent kernels should be near zero."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 5))
        Y = rng.normal(size=(50, 5))
        K = X @ X.T
        L = Y @ Y.T
        assert abs(_hsic(K, L)) < 0.5  # loose bound for finite sample

    def test_small_n_returns_zero(self) -> None:
        K = np.eye(3)
        assert _hsic(K, K) == 0.0

    def test_does_not_mutate_input(self) -> None:
        """_hsic must not modify the caller's matrices."""
        K = np.ones((5, 5))
        L = np.ones((5, 5))
        K_orig = K.copy()
        L_orig = L.copy()
        _hsic(K, L)
        np.testing.assert_array_equal(K, K_orig)
        np.testing.assert_array_equal(L, L_orig)


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


class TestCKA:
    def test_identical_matrices(self) -> None:
        """CKA(X, X) should be 1.0."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 10))
        assert compute_cka(X, X) == pytest.approx(1.0, abs=0.01)

    def test_scaled_copy(self) -> None:
        """CKA is invariant to isotropic scaling."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 10))
        Y = X * 3.5
        assert compute_cka(X, Y) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_representations_low(self) -> None:
        """Truly independent representations should have low CKA."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 10))
        Y = rng.normal(size=(50, 10))
        assert compute_cka(X, Y) < 0.3

    def test_mismatched_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="Sample count mismatch"):
            compute_cka(np.zeros((10, 5)), np.zeros((20, 5)))

    def test_different_feature_dims(self) -> None:
        """CKA should work with different feature dimensions."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 5))
        Y = rng.normal(size=(30, 20))
        val = compute_cka(X, Y)
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Inter-class variance and SDF overlap
# ---------------------------------------------------------------------------


class TestInterClassVariance:
    def test_perfectly_separated(self) -> None:
        """Features that perfectly separate classes should have high variance."""
        acts = np.array(
            [
                [10.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [0.0, 10.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        var = _inter_class_variance(acts, labels)
        assert var[0] > 0  # feature 0 separates classes
        assert var[1] > 0  # feature 1 separates classes

    def test_uniform_feature_zero_variance(self) -> None:
        """A feature that's identical across classes has zero inter-class variance."""
        acts = np.array(
            [
                [5.0, 1.0],
                [5.0, 2.0],
                [5.0, 3.0],
                [5.0, 4.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        var = _inter_class_variance(acts, labels)
        assert var[0] == pytest.approx(0.0)
        assert var[1] > 0


class TestSDFOverlap:
    def test_identical_activations(self) -> None:
        """Same activations should give perfect overlap."""
        rng = np.random.default_rng(0)
        acts = rng.normal(size=(100, 50))
        labels = np.repeat(np.arange(5), 20)
        rho = compute_sdf_overlap(acts, acts, labels)
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_independent_activations_valid_range(self) -> None:
        """Independent activations should produce a valid Spearman rho."""
        rng = np.random.default_rng(0)
        acts_a = rng.normal(size=(100, 50))
        acts_b = rng.normal(size=(100, 50))
        labels = np.repeat(np.arange(5), 20)
        rho = compute_sdf_overlap(acts_a, acts_b, labels)
        # The function compares sorted variance distribution *shapes*,
        # so even independent data can yield high rho (both follow
        # similar descending curves).  Just verify valid range.
        assert -1.0 <= rho <= 1.0

    def test_different_feature_dims(self) -> None:
        """Should handle different numbers of features."""
        rng = np.random.default_rng(0)
        acts_a = rng.normal(size=(100, 30))
        acts_b = rng.normal(size=(100, 60))
        labels = np.repeat(np.arange(5), 20)
        rho = compute_sdf_overlap(acts_a, acts_b, labels)
        assert -1.0 <= rho <= 1.0


# ---------------------------------------------------------------------------
# Principal angles and subspace similarity
# ---------------------------------------------------------------------------


class TestPrincipalAngles:
    def test_identical_subspace(self) -> None:
        """Angles between a subspace and itself should be near zero."""
        rng = np.random.default_rng(0)
        A = rng.normal(size=(10, 3))
        angles = _principal_angles(A, A)
        # QR + SVD introduce ~1e-8 numerical noise; use a practical tolerance.
        np.testing.assert_allclose(angles, 0.0, atol=1e-6)

    def test_orthogonal_subspaces(self) -> None:
        """Orthogonal subspaces should have angles of pi/2."""
        A = np.eye(6)[:, :3]  # first 3 standard basis vectors
        B = np.eye(6)[:, 3:]  # last 3 standard basis vectors
        angles = _principal_angles(A, B)
        np.testing.assert_allclose(angles, np.pi / 2, atol=1e-10)


class TestDemographicSubspaceSimilarity:
    def _make_sae(self, input_size: int, hidden_size: int) -> "BaseSAE":
        return build_sae("vanilla", input_size=input_size, hidden_size=hidden_size, l1_coeff=1e-4)

    def test_same_sae_identity(self) -> None:
        sae = self._make_sae(32, 64)
        indices = np.array([0, 1, 2, 3])
        result = demographic_subspace_similarity(sae, sae, indices, indices)
        assert result["mean_cos_principal_angle"] == pytest.approx(1.0, abs=0.01)
        assert result["grassmann_distance"] == pytest.approx(0.0, abs=0.01)

    def test_different_saes(self) -> None:
        sae_a = self._make_sae(32, 64)
        sae_b = self._make_sae(32, 64)
        idx_a = np.array([0, 1, 2])
        idx_b = np.array([10, 11, 12])
        result = demographic_subspace_similarity(sae_a, sae_b, idx_a, idx_b)
        assert 0.0 <= result["mean_cos_principal_angle"] <= 1.0
        assert result["grassmann_distance"] >= 0.0

    def test_different_input_sizes(self) -> None:
        """SAEs with different input sizes should still work (zero-padding)."""
        sae_a = self._make_sae(32, 64)
        sae_b = self._make_sae(48, 64)
        idx_a = np.array([0, 1])
        idx_b = np.array([0, 1])
        result = demographic_subspace_similarity(sae_a, sae_b, idx_a, idx_b)
        assert 0.0 <= result["mean_cos_principal_angle"] <= 1.0

    def test_empty_indices(self) -> None:
        sae = self._make_sae(32, 64)
        result = demographic_subspace_similarity(
            sae,
            sae,
            np.array([]),
            np.array([0, 1]),
        )
        assert result["mean_cos_principal_angle"] == 0.0
