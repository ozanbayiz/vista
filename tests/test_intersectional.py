"""Unit tests for src.intersectional -- intersectional label construction,
mutual information, and interaction effect utilities.

All tests use synthetic data; no HDF5 files or checkpoints required.
"""

import numpy as np
import pytest

from src.intersectional import (
    _binned_mi,
    build_intersectional_labels,
    information_decomposition,
)


# ---------------------------------------------------------------------------
# build_intersectional_labels
# ---------------------------------------------------------------------------


class TestBuildIntersectionalLabels:
    def _make_labels(self, n: int = 200) -> dict[str, np.ndarray]:
        """Create synthetic labels with 3 age, 2 gender, 2 race classes."""
        rng = np.random.default_rng(42)
        return {
            "age": rng.integers(0, 3, size=n),
            "gender": rng.integers(0, 2, size=n),
            "race": rng.integers(0, 2, size=n),
        }

    def test_composite_label_values(self) -> None:
        labels = self._make_labels(300)
        composite, mapping, readable = build_intersectional_labels(
            labels,
            min_subgroup_size=5,
        )
        # All valid labels should be >= 0
        assert (composite >= -1).all()
        # Number of unique valid labels matches mapping
        valid = composite[composite >= 0]
        assert len(np.unique(valid)) == len(mapping)

    def test_mapping_consistency(self) -> None:
        labels = self._make_labels(300)
        composite, mapping, readable = build_intersectional_labels(
            labels,
            min_subgroup_size=5,
        )
        for cls, attrs in mapping.items():
            mask = composite == cls
            # All samples with this composite label should share the same triple
            for attr in ("age", "gender", "race"):
                assert np.all(labels[attr][mask] == attrs[attr])

    def test_rare_subgroups_excluded(self) -> None:
        """Subgroups below min_subgroup_size get label -1."""
        labels = {
            "age": np.array([0] * 50 + [1] * 50 + [2] * 2),
            "gender": np.array([0] * 50 + [0] * 50 + [1] * 2),
            "race": np.array([0] * 50 + [0] * 50 + [0] * 2),
        }
        composite, mapping, readable = build_intersectional_labels(
            labels,
            min_subgroup_size=10,
        )
        # The last 2 samples (age=2, gender=1, race=0) should be excluded
        assert composite[-1] == -1
        assert composite[-2] == -1

    def test_readable_names_populated(self) -> None:
        labels = self._make_labels(200)
        composite, mapping, readable = build_intersectional_labels(
            labels,
            min_subgroup_size=5,
        )
        assert len(readable) == len(mapping)
        for name in readable:
            assert len(name) > 0
            assert "_" in name  # format is "age_gender_race"


# ---------------------------------------------------------------------------
# Binned mutual information
# ---------------------------------------------------------------------------


class TestBinnedMI:
    def test_independent_near_zero(self) -> None:
        """MI between independent variables should be near zero."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        y = rng.integers(0, 5, size=1000)
        mi = _binned_mi(x, y, n_bins=20)
        assert mi < 0.1  # should be very small

    def test_deterministic_high_mi(self) -> None:
        """MI should be high when x perfectly predicts y."""
        # Class 0: x ~ N(-5, 0.1), Class 1: x ~ N(+5, 0.1)
        rng = np.random.default_rng(0)
        n = 500
        y = np.repeat([0, 1], n)
        x = np.concatenate(
            [
                rng.normal(-5, 0.1, n),
                rng.normal(5, 0.1, n),
            ]
        )
        mi = _binned_mi(x, y, n_bins=20)
        # Should be close to log2(2) = 1.0 bit
        assert mi > 0.5

    def test_empty_returns_zero(self) -> None:
        assert _binned_mi(np.array([]), np.array([]), n_bins=10) == 0.0

    def test_constant_x_returns_zero(self) -> None:
        """Constant feature has zero MI with everything."""
        x = np.ones(100)
        y = np.repeat([0, 1], 50)
        assert _binned_mi(x, y, n_bins=10) == 0.0

    def test_non_negative(self) -> None:
        """MI should always be >= 0 (bias correction is clipped)."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            x = rng.normal(size=50)
            y = rng.integers(0, 3, size=50)
            assert _binned_mi(x, y) >= 0.0


# ---------------------------------------------------------------------------
# information_decomposition (lightweight, skip ANOVA)
# ---------------------------------------------------------------------------


class TestInformationDecomposition:
    def test_basic_decomposition(self) -> None:
        """Verify MI is computed for all label sets and synergy is reasonable."""
        rng = np.random.default_rng(0)
        n = 200
        acts = rng.normal(size=(n, 10))
        labels = {
            "age": rng.integers(0, 3, size=n),
            "gender": rng.integers(0, 2, size=n),
            "race": rng.integers(0, 2, size=n),
        }
        # Build composite labels for the function
        n_gender = 2
        n_race = 2
        composite = labels["age"] * (n_gender * n_race) + labels["gender"] * n_race + labels["race"]

        results = information_decomposition(
            acts,
            labels,
            composite,
            np.array([0, 1, 2]),
            n_bins=10,
        )
        assert len(results) == 3
        for r in results:
            assert "mi" in r
            assert "synergy_estimate" in r
            assert "age" in r["mi"]
            assert "age_x_gender_x_race" in r["mi"]
