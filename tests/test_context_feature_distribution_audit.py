"""STEP 6-E2 frozen feature distribution audit의 synthetic 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drowsiness_detection.evaluation_v2.context_feature_distribution_audit import (
    ContextFeatureDistributionAuditError,
    _finalize_state,
    _new_state,
    _update_state,
    audit_backbone,
    classify_feature_shift,
    dimension_shift_rows,
    distribution_summary,
    split_mean_vector_comparison,
    video_feature_statistics,
)


def _features(offset: float = 0.0, scale: float = 1.0) -> np.ndarray:
    values = np.arange(32 * 512, dtype=np.float32).reshape(32, 512)
    return values * np.float32(scale / values.size) + np.float32(offset)


def test_video_statistics_shape_dtype_finite_and_norms() -> None:
    features = _features()
    result = video_feature_statistics(features)
    frame_norms = np.linalg.norm(features.astype(np.float64), axis=1)
    assert result["feature_mean"] == pytest.approx(float(features.mean()))
    assert result["feature_std"] == pytest.approx(float(features.std()))
    assert result["mean_frame_l2_norm"] == pytest.approx(float(frame_norms.mean()))
    assert result["median_frame_l2_norm"] == pytest.approx(float(np.median(frame_norms)))
    assert result["min_frame_l2_norm"] == pytest.approx(float(frame_norms.min()))
    assert result["max_frame_l2_norm"] == pytest.approx(float(frame_norms.max()))


def test_minimal_integrity_rejects_bad_shape_dtype_and_nonfinite() -> None:
    with pytest.raises(ContextFeatureDistributionAuditError, match="shape"):
        video_feature_statistics(np.zeros((31, 512), dtype=np.float32))
    with pytest.raises(ContextFeatureDistributionAuditError, match="dtype"):
        video_feature_statistics(np.zeros((32, 512), dtype=np.float64))
    nonfinite = np.zeros((32, 512), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ContextFeatureDistributionAuditError, match="NaN/Inf"):
        video_feature_statistics(nonfinite)


def test_streaming_state_global_dimension_and_zero_counts() -> None:
    state = _new_state()
    zeros = np.zeros((32, 512), dtype=np.float32)
    shifted = np.ones((32, 512), dtype=np.float32)
    _update_state(state, zeros)
    _update_state(state, shifted)
    result = _finalize_state(state)
    assert result["integrity"]["video_count"] == 2
    assert result["integrity"]["zero_element_count"] == 32 * 512
    assert result["integrity"]["all_zero_frame_feature_count"] == 32
    assert result["integrity"]["all_zero_video_count"] == 1
    assert result["global"]["mean"] == 0.5
    assert result["global"]["std"] == 0.5
    assert np.allclose(result["dimension_mean"], 0.5)
    assert np.allclose(result["dimension_std"], 0.5)


def test_dimension_shift_and_mean_vector_comparison() -> None:
    train = np.vstack([_features(), _features(0.1)]).reshape(-1, 512)
    val = np.vstack([_features(0.2), _features(0.3)]).reshape(-1, 512)
    rows = dimension_shift_rows("resnet18", train, val)
    assert len(rows) == 512
    assert rows[0]["absolute_mean_difference"] == pytest.approx(0.2, abs=1e-6)
    assert rows[0]["standardized_shift"] >= 0
    comparison = split_mean_vector_comparison(train.mean(axis=0), val.mean(axis=0))
    assert comparison["cosine_similarity"] is not None
    assert -1.0 <= comparison["cosine_similarity"] <= 1.0
    assert comparison["l2_distance"] > 0


def _video_metric_summary(center: float) -> dict[str, dict[str, float]]:
    result = {}
    for metric in ("feature_mean", "feature_std", "mean_frame_l2_norm",
                   "median_frame_l2_norm", "min_frame_l2_norm", "max_frame_l2_norm"):
        result[metric] = {
            "count": 10, "mean": center, "median": center,
            "p10": center - 1, "p90": center + 1,
            "p95": center + 1, "p99": center + 1,
            "min": center - 1, "max": center + 1,
        }
    return result


def _backbone_summary(backbone: str, shifted: bool) -> dict:
    val_center = 3.0 if shifted else 0.0
    return {
        "backbone": backbone,
        "splits": {
            "train": {"video_level": _video_metric_summary(0.0)},
            "val": {"video_level": _video_metric_summary(val_center)},
        },
        "dimension_standardized_shift": {"p95": 0.8 if shifted else 0.1},
        "split_mean_vector": {"cosine_similarity": 0.95 if shifted else 0.999},
    }


def _label_rows() -> list[dict]:
    rows = []
    for backbone in ("resnet18", "vgg16"):
        for split in ("train", "val"):
            for label in ("not_drowsy", "drowsy"):
                for metric in ("feature_mean", "feature_std", "mean_frame_l2_norm"):
                    rows.append({
                        "backbone": backbone, "split": split, "label": label,
                        "metric": metric, "median": 0.0, "p10": -1.0, "p90": 1.0,
                    })
    return rows


def test_descriptive_classification_supports_none_and_backbone_specific() -> None:
    labels = _label_rows()
    classification, evidence = classify_feature_shift([
        _backbone_summary("resnet18", False),
        _backbone_summary("vgg16", False),
    ], labels)
    assert classification == "NO_OBVIOUS_FEATURE_DISTRIBUTION_SHIFT"
    assert "significance" in evidence["criteria_note"]

    classification, _ = classify_feature_shift([
        _backbone_summary("resnet18", True),
        _backbone_summary("vgg16", False),
    ], labels)
    assert classification == "BACKBONE_SPECIFIC_SHIFT"


def test_distribution_summary_is_video_level_descriptive() -> None:
    summary = distribution_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["median"] == 2.5
    assert summary["p10"] == pytest.approx(1.3)
    assert summary["p90"] == pytest.approx(3.7)


def test_unsupported_backbone_is_blocked_before_artifact_access() -> None:
    with pytest.raises(ContextFeatureDistributionAuditError, match="unsupported backbone"):
        audit_backbone(Path("unused"), "test")
