"""STEP 6-E3 temporal variation audit의 synthetic 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drowsiness_detection.evaluation_v2.context_temporal_variation_audit import (
    ContextTemporalVariationAuditError,
    audit_backbone_temporal,
    classify_temporal_pattern,
    summarize_split_transitions,
    temporal_statistics,
    transition_type,
)


def _sequence(step: float = 1.0) -> np.ndarray:
    base = np.ones((32, 512), dtype=np.float32)
    direction = np.linspace(0.1, 1.0, 512, dtype=np.float32)
    return base + np.arange(32, dtype=np.float32)[:, None] * direction * step


def test_adjacent_l2_cosine_start_end_and_video_metrics() -> None:
    features = _sequence(0.01)
    mask = np.zeros(32, dtype=np.bool_)
    stats, l2_delta, cosine, kinds = temporal_statistics(features, mask)
    expected_delta = np.linalg.norm(
        features[1].astype(np.float64) - features[0].astype(np.float64))
    assert l2_delta.shape == (31,)
    assert cosine.shape == (31,)
    assert np.allclose(l2_delta, expected_delta)
    assert np.all((-1 <= cosine) & (cosine <= 1))
    assert stats["mean_temporal_l2_delta"] == pytest.approx(expected_delta)
    assert stats["start_end_l2"] == pytest.approx(expected_delta * 31)
    assert len(kinds) == 31
    assert set(kinds) == {"original_to_original"}


def test_transition_types_cover_all_imputation_connections() -> None:
    assert transition_type(False, False) == "original_to_original"
    assert transition_type(False, True) == "original_to_imputed"
    assert transition_type(True, False) == "imputed_to_original"
    assert transition_type(True, True) == "imputed_to_imputed"
    features = _sequence(0.01)
    mask = np.zeros(32, dtype=np.bool_)
    mask[1:3] = True
    _, _, _, kinds = temporal_statistics(features, mask)
    assert kinds[:3] == [
        "original_to_imputed", "imputed_to_imputed", "imputed_to_original"]


def test_split_summary_reports_parallel_redundancy_thresholds() -> None:
    l2_values = [0.1, 0.2, 0.3, 0.4]
    cosine_values = [0.94, 0.97, 0.991, 0.999]
    summary = summarize_split_transitions(l2_values, cosine_values)
    assert summary["transition_count"] == 4
    assert summary["adjacent_l2_delta"]["median"] == pytest.approx(0.25)
    rates = summary["near_duplicate_transition_rate"]
    assert rates["cosine_gte_0.95"] == 0.75
    assert rates["cosine_gte_0.98"] == 0.5
    assert rates["cosine_gte_0.99"] == 0.5
    assert rates["cosine_gte_0.995"] == 0.25


def test_invalid_feature_mask_and_zero_norm_are_blocked() -> None:
    with pytest.raises(ContextTemporalVariationAuditError, match="feature 계약"):
        temporal_statistics(np.ones((31, 512), dtype=np.float32),
                            np.zeros(32, dtype=np.bool_))
    with pytest.raises(ContextTemporalVariationAuditError, match="mask 계약"):
        temporal_statistics(_sequence(), np.zeros(31, dtype=np.bool_))
    with pytest.raises(ContextTemporalVariationAuditError, match="zero-norm"):
        temporal_statistics(np.zeros((32, 512), dtype=np.float32),
                            np.zeros(32, dtype=np.bool_))


def _metric_summary(center: float) -> dict:
    return {
        "count": 10, "mean": center, "median": center,
        "p10": center - 1, "p90": center + 1,
        "p95": center + 1, "p99": center + 1,
        "min": center - 1, "max": center + 1,
    }


def _backbone_summary(backbone: str, redundancy: bool = False) -> dict:
    video_level = {
        metric: _metric_summary(0.0) for metric in (
            "mean_temporal_l2_delta", "median_temporal_l2_delta",
            "mean_adjacent_cosine", "start_end_l2", "start_end_cosine")
    }
    rate = 0.9 if redundancy else 0.2
    return {
        "backbone": backbone,
        "splits": {
            split: {
                "video_level": video_level,
                "near_duplicate_transition_rate": {"cosine_gte_0.99": rate},
            } for split in ("train", "val")
        },
    }


def _label_rows() -> list[dict]:
    return [{
        "backbone": backbone, "split": split, "label": label, "metric": metric,
        "median": 0.0, "p10": -1.0, "p90": 1.0,
    } for backbone in ("resnet18", "vgg16")
    for split in ("train", "val")
    for label in ("not_drowsy", "drowsy")
    for metric in ("mean_temporal_l2_delta", "median_temporal_l2_delta",
                   "mean_adjacent_cosine", "start_end_l2", "start_end_cosine")]


def test_classification_supports_no_shift_and_redundancy() -> None:
    labels = _label_rows()
    classification, evidence = classify_temporal_pattern([
        _backbone_summary("resnet18"), _backbone_summary("vgg16")], labels)
    assert classification == "NO_OBVIOUS_TEMPORAL_SHIFT"
    assert "Not significance" in evidence["criteria_note"]
    classification, _ = classify_temporal_pattern([
        _backbone_summary("resnet18", redundancy=True),
        _backbone_summary("vgg16"),
    ], labels)
    assert classification == "POSSIBLE_TEMPORAL_REDUNDANCY"


def test_unsupported_backbone_is_blocked_before_artifact_access() -> None:
    with pytest.raises(ContextTemporalVariationAuditError, match="unsupported backbone"):
        audit_backbone_temporal(Path("unused"), "test")
