"""STEP 6-E4 hard-sample audit의 순수 계산 로직 테스트."""

from __future__ import annotations

import torch
import pytest

from drowsiness_detection.evaluation_v2.context_val_hard_sample_audit import (
    ContextValHardSampleAuditError,
    aggregate_inference_metrics,
    backbone_overlap_rows,
    classify_result,
    descriptive_relations,
    hard_sample_rows,
    join_audit_fields,
    loss_concentration_rows,
    per_sample_rows_from_logits,
    spearman_correlation,
    verify_metric_reproduction,
)


def _rows(count: int = 30) -> list[dict]:
    result = []
    for index in range(count):
        result.append({
            "backbone": "resnet18",
            "video_id": f"v{index:03d}",
            "label": "drowsy" if index % 2 else "not_drowsy",
            "label_index": index % 2,
            "predicted_label": "drowsy" if index % 3 else "not_drowsy",
            "predicted_label_index": 1 if index % 3 else 0,
            "correct": (index % 2) == (1 if index % 3 else 0),
            "predicted_class_probability": 0.5 + index / 100,
            "per_sample_ce_loss": float(index + 1),
            "imputed_count": index % 3,
            "mean_frame_l2_norm": float(index + 2),
            "mean_temporal_l2_delta": float(index + 3),
            "mean_adjacent_cosine": 1.0 - index / 100,
        })
    return result


def test_per_sample_unsmoothed_ce_and_metric_shapes() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.long)
    rows = per_sample_rows_from_logits("resnet18", ["a", "b"], labels, logits)
    expected = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert [row["per_sample_ce_loss"] for row in rows] == pytest.approx(expected.tolist())
    assert rows[0]["correct"] is True
    assert rows[1]["correct"] is False
    metrics = aggregate_inference_metrics(rows)
    assert metrics["video_count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [0, 0]]


def test_metric_reproduction_gate_rejects_mismatch() -> None:
    metrics = {
        "val_loss": 0.0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
        "video_count": 295,
    }
    with pytest.raises(ContextValHardSampleAuditError, match="hard-sample 분석을 중단"):
        verify_metric_reproduction("resnet18", metrics)


def test_loss_concentration_and_hard_sample_order() -> None:
    rows = _rows()
    concentration = loss_concentration_rows("resnet18", rows)
    lookup = {row["segment"]: row for row in concentration}
    assert lookup["top_5"]["sample_count"] == 5
    assert lookup["top_10_percent"]["sample_count"] == 3
    assert 0 < lookup["top_5"]["share"] < 1
    candidates = hard_sample_rows("resnet18", rows)
    loss_top = [row for row in candidates if row["candidate_type"] == "loss_top20"]
    assert loss_top[0]["video_id"] == "v029"
    assert loss_top[0]["rank"] == 1


def test_overlap_reports_shared_and_backbone_specific_errors() -> None:
    left = _rows()
    right = [dict(row, backbone="vgg16") for row in left]
    right[0]["correct"] = not left[0]["correct"]
    overlap = backbone_overlap_rows(left, right)
    categories = {row["category"] for row in overlap}
    assert "both_high_loss_top20" in categories
    assert "vgg16_only_incorrect" in categories or "resnet18_only_incorrect" in categories


def test_join_descriptive_relations_and_tie_aware_spearman() -> None:
    source = [{
        "video_id": "a", "correct": True, "label": "not_drowsy",
        "per_sample_ce_loss": 0.1,
    }, {
        "video_id": "b", "correct": False, "label": "drowsy",
        "per_sample_ce_loss": 0.9,
    }]
    e1 = {
        "a": {"imputed_count": "0", "max_imputed_run": "0"},
        "b": {"imputed_count": "2", "max_imputed_run": "1"},
    }
    e2 = {key: {
        "feature_mean": "1", "feature_std": "2", "mean_frame_l2_norm": str(index + 3),
    } for index, key in enumerate(("a", "b"))}
    e3 = {key: {
        "mean_temporal_l2_delta": str(index + 1),
        "mean_adjacent_cosine": str(0.9 - index / 10),
        "start_end_l2": "4", "start_end_cosine": "0.5",
    } for index, key in enumerate(("a", "b"))}
    joined = join_audit_fields(source, e1, e2, e3)
    relation = descriptive_relations(joined)
    assert relation["groups"]["prediction"]["incorrect"]["mean_ce"] == 0.9
    assert spearman_correlation([1, 1, 2], [1, 1, 3]) == pytest.approx(1.0)


def test_classification_uses_predeclared_descriptive_thresholds() -> None:
    concentration = [
        {"backbone": "resnet18", "segment": "top_10_percent", "share": 0.55},
        {"backbone": "vgg16", "segment": "top_10_percent", "share": 0.40},
    ]
    classification, evidence = classify_result(concentration, [])
    assert classification == "LOSS_CONCENTRATED_IN_FEW_SAMPLES"
    assert "criteria" in evidence
