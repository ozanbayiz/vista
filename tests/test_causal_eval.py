"""Unit tests for src.causal_eval -- demographic content detection, fuzzy
matching, caption/VQA metrics, and bootstrap confidence intervals.

All tests run offline without GPU or model weights.
"""

import numpy as np
import pytest

from src.causal_eval import (
    CaptionCausalMetrics,
    DemographicContentDetector,
    VQACausalMetrics,
    _bootstrap_ci,
    _fuzzy_match,
    aggregate_results,
)


# ---------------------------------------------------------------------------
# DemographicContentDetector
# ---------------------------------------------------------------------------


class TestDemographicContentDetector:
    @pytest.fixture()
    def detector(self) -> DemographicContentDetector:
        return DemographicContentDetector()

    # -- race terms --

    def test_detect_race_keyword(self, detector: DemographicContentDetector) -> None:
        assert detector.detect("The person is caucasian.", "race")
        assert detector.detect("An East Asian woman.", "race")

    def test_detect_race_case_insensitive(self, detector: DemographicContentDetector) -> None:
        assert detector.detect("She appears to be AFRICAN.", "race")

    def test_no_false_positive_race(self, detector: DemographicContentDetector) -> None:
        assert not detector.detect("A person wearing a hat.", "race")

    # -- gender terms --

    def test_detect_gender_keyword(self, detector: DemographicContentDetector) -> None:
        assert detector.detect("A young woman smiling.", "gender")
        assert detector.detect("He is wearing a jacket.", "gender")

    def test_gender_word_boundary(self, detector: DemographicContentDetector) -> None:
        """'her' should not match inside 'here' or 'there'."""
        assert not detector.detect("I went there and here.", "gender")
        assert detector.detect("This is her bag.", "gender")

    # -- age terms --

    def test_detect_age_keyword(self, detector: DemographicContentDetector) -> None:
        assert detector.detect("An elderly gentleman.", "age")
        assert detector.detect("A teenager with headphones.", "age")

    def test_no_false_positive_age(self, detector: DemographicContentDetector) -> None:
        assert not detector.detect("A person with glasses.", "age")

    # -- unknown attribute --

    def test_unknown_attribute_raises(self, detector: DemographicContentDetector) -> None:
        with pytest.raises(ValueError, match="Unknown attribute"):
            detector.detect("some text", "height")

    # -- DCR --

    def test_dcr_all_hits(self, detector: DemographicContentDetector) -> None:
        texts = ["A young girl", "An old man", "A teenager"]
        assert detector.compute_dcr(texts, "age") == pytest.approx(1.0)

    def test_dcr_no_hits(self, detector: DemographicContentDetector) -> None:
        texts = ["A person", "Another person"]
        assert detector.compute_dcr(texts, "race") == pytest.approx(0.0)

    def test_dcr_empty_list(self, detector: DemographicContentDetector) -> None:
        assert detector.compute_dcr([], "age") == pytest.approx(0.0)

    def test_dcr_partial(self, detector: DemographicContentDetector) -> None:
        texts = ["A white cat", "A red car"]  # "white" is a race term
        assert detector.compute_dcr(texts, "race") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    def test_race_match(self) -> None:
        assert _fuzzy_match("The person appears to be East Asian", 3, "race")

    def test_race_no_match(self) -> None:
        assert not _fuzzy_match("The person appears happy", 0, "race")

    def test_gender_match(self) -> None:
        assert _fuzzy_match("This is a woman", 1, "gender")

    def test_gender_case_insensitive(self) -> None:
        assert _fuzzy_match("MALE", 0, "gender")

    def test_age_match(self) -> None:
        assert _fuzzy_match("The person looks about 25 years old", 3, "age")

    def test_unknown_attribute(self) -> None:
        """Unknown attribute should return False (no synonyms found)."""
        assert not _fuzzy_match("anything", 0, "height")


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_deterministic_values(self) -> None:
        """All-same values should give zero-width CI."""
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        mean, lo, hi = _bootstrap_ci(values)
        assert mean == pytest.approx(0.5)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(10.0, 1.0, size=200)
        mean, lo, hi = _bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_ci_width_shrinks_with_more_data(self) -> None:
        rng = np.random.default_rng(0)
        small = rng.normal(0, 1, size=20)
        large = rng.normal(0, 1, size=2000)
        _, lo_s, hi_s = _bootstrap_ci(small)
        _, lo_l, hi_l = _bootstrap_ci(large)
        assert (hi_l - lo_l) < (hi_s - lo_s)


# ---------------------------------------------------------------------------
# CaptionCausalMetrics (without BERTScore, since we don't require it in tests)
# ---------------------------------------------------------------------------


class TestCaptionCausalMetrics:
    def test_dcr_delta_computed(self) -> None:
        metrics = CaptionCausalMetrics()
        original = ["A black woman smiling", "A white man walking"]
        modified = ["A person smiling", "A person walking"]  # demographic removed
        result = metrics.compute(original, modified, "race")
        # Race DCR should decrease (negative delta)
        assert result["dcr_race"]["dcr_delta"] < 0

    def test_no_change_zero_delta(self) -> None:
        metrics = CaptionCausalMetrics()
        texts = ["A person sitting", "A person standing"]
        result = metrics.compute(texts, texts, "race")
        assert result["dcr_race"]["dcr_delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VQACausalMetrics
# ---------------------------------------------------------------------------


class TestVQACausalMetrics:
    def _make_samples(
        self,
        n: int = 10,
        *,
        original_acc: float = 0.8,
        modified_acc: float = 0.2,
    ) -> list[dict]:
        """Build synthetic VQA samples where demographic accuracy drops."""
        samples = []
        for i in range(n):
            gt_label = 0  # White
            # Original answers are correct most of the time
            orig_demo = "white person" if (i / n) < original_acc else "unknown"
            mod_demo = "white person" if (i / n) < modified_acc else "unknown"
            samples.append(
                {
                    "labels": {"race": gt_label, "gender": 0, "age": 3},
                    "questions": {
                        "What is this person's race or ethnicity?": {
                            "type": "demographic",
                            "original": orig_demo,
                            "modified": mod_demo,
                        },
                        "What is this person wearing?": {
                            "type": "control",
                            "original": "A blue shirt and jeans",
                            "modified": "A blue shirt and jeans",
                        },
                    },
                }
            )
        return samples

    def test_demographic_accuracy_drop(self) -> None:
        samples = self._make_samples(n=20, original_acc=0.8, modified_acc=0.2)
        metrics = VQACausalMetrics()
        result = metrics.compute(samples, "race")
        # Demographic accuracy should drop
        demo = result["per_type"]["demographic"]
        assert demo["mean_delta"] < 0

    def test_control_preserved(self) -> None:
        samples = self._make_samples(n=20)
        metrics = VQACausalMetrics()
        result = metrics.compute(samples, "race")
        # Control accuracy should be preserved (both answers are long enough)
        ctrl = result["per_type"]["control"]
        assert ctrl["mean_delta"] == pytest.approx(0.0, abs=0.01)

    def test_causal_effect_negative(self) -> None:
        """Causal effect = demo_delta - ctrl_delta; should be negative when
        demographic accuracy drops but control is preserved."""
        samples = self._make_samples(n=20, original_acc=0.8, modified_acc=0.2)
        metrics = VQACausalMetrics()
        result = metrics.compute(samples, "race")
        assert result["causal_effect"] < 0

    def test_missing_labels_skipped(self) -> None:
        samples = [{"labels": {}, "questions": {}}]
        metrics = VQACausalMetrics()
        result = metrics.compute(samples, "race")
        assert result["per_question"] == {}


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------


class TestAggregateResults:
    def test_caption_aggregation(self) -> None:
        samples = [
            {"labels": {"race": 0}, "original_text": "A white person", "modified_text": "A person"},
            {"labels": {"race": 1}, "original_text": "A black person", "modified_text": "A person"},
        ]
        result = aggregate_results(samples, "race", "caption")
        assert "overall" in result
        assert "per_subgroup" in result
        assert result["overall"]["dcr_delta"] < 0  # race terms removed

    def test_vqa_aggregation(self) -> None:
        samples = [{"labels": {"race": 0}}, {"labels": {"race": 1}}]
        result = aggregate_results(samples, "race", "vqa")
        assert "per_subgroup" in result
