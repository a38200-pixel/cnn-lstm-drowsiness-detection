"""STEP 5-D2R pilot/full feature 회귀 감사 테스트."""

from __future__ import annotations

import numpy as np
import pytest

import drowsiness_detection.sequences_v2.cnn_feature_regression_audit as audit
from drowsiness_detection.sequences_v2.cnn_feature_extractor import full_run_status


def test_difference_stats_rejects_accidental_broadcasting():
    with pytest.raises(ValueError, match="REGRESSION_SHAPE_MISMATCH"):
        audit.difference_stats(np.zeros((32, 512), np.float32),
                               np.zeros((1, 512), np.float32))


def test_difference_stats_reports_element_and_tolerance_counts():
    left = np.zeros((2, 512), dtype=np.float32)
    right = left.copy()
    right[1, 10] = 2e-4
    result = audit.difference_stats(left, right)
    assert result["shape"] == [2, 512]
    assert result["nonzero_diff_element_count"] == 1
    assert result["allclose_failed_element_count"] == 1
    assert not result["allclose"] and not result["exact_equal"]
    assert [(item["rtol"], item["atol"]) for item in result["tolerance_results"]] == [
        (1e-6, 1e-7), (1e-5, 1e-6), (1e-4, 1e-5)]


def test_batch_position_computes_full_and_partial_batches():
    assert audit.batch_position(447, 454) == {
        "source_feature_id": 447, "batch_index": 27, "position_within_batch": 15,
        "batch_actual_size": 16, "partial_batch": False,
        "batch_start_source_feature_id": 432,
        "batch_end_source_feature_id_exclusive": 448,
    }
    partial = audit.batch_position(453, 454)
    assert partial["batch_index"] == 28
    assert partial["position_within_batch"] == 5
    assert partial["batch_actual_size"] == 6 and partial["partial_batch"]
    full_tail = audit.batch_position(52964, 52965)
    assert full_tail["batch_actual_size"] == 5 and full_tail["partial_batch"]


def test_full_run_status_separates_extraction_and_integrity():
    failed = full_run_status([{"pilot_full_regression": {"status": "FAIL"}}])
    assert failed == {
        "status": "STEP_5D2_FULL_EXTRACTION_COMPLETED_REVIEW_REQUIRED",
        "extraction_status": "COMPLETE", "integrity_status": "REVIEW_REQUIRED"}
    passed = full_run_status([{"pilot_full_regression": {"status": "PASS"}}])
    assert passed["extraction_status"] == "COMPLETE"
    assert passed["integrity_status"] == "PASS"


def _classification_item(mapping=0, source_path=0, max_abs=1e-4):
    comparison = {"allclose": True, "exact_equal": True, "max_abs_diff": 0.0}
    batch_comparison = {"allclose": False, "exact_equal": False,
                        "max_abs_diff": max_abs}
    return {"mapping_mismatch": mapping, "source_path_mismatch": source_path,
            "shape_dtype_equal": True, "controlled_rerun": {"sources": [{"comparisons": {
                "d1_stored_vs_d1_batch": comparison,
                "d2_stored_vs_d2_batch": comparison,
                "d1_batch_vs_d2_batch": batch_comparison,
            }}]}}


def test_classification_requires_batch_reproduction_and_small_difference():
    assert audit.classify_cause([_classification_item()]) == (
        "BENIGN FLOATING-POINT / BATCH NUMERICAL DIFFERENCE")
    assert audit.classify_cause([_classification_item(max_abs=0.1)]) == "UNRESOLVED"
    assert audit.classify_cause([_classification_item(mapping=1)]) == "SOURCE MAPPING BUG"


@pytest.mark.parametrize("backbone", ["resnet18", "vgg16"])
def test_actual_stored_regression_identifies_n311_without_mapping_error(backbone):
    result = audit.compare_stored_backbone(audit.Path("."), backbone)
    assert result["mismatched_video_ids"] == ["n_311"]
    assert result["mapping_mismatch"] == 0
    assert result["source_path_mismatch"] == 0
    assert result["shape_dtype_equal"]
    assert result["pilot_unique_sources"] == 454
    assert result["exact_equal_sources"] == 448
    assert result["allclose_mismatch_sources"] == 6
    slots = [slot["target_context_index"]
             for slot in result["mismatched_videos"][0]["slots"] if not slot["allclose"]]
    assert slots == list(range(24, 32))


def test_audit_preserves_test_seal_and_artifacts(monkeypatch, tmp_path):
    item = {"backbone": "resnet18", "mismatched_video_ids": [],
            "mapping_mismatch": 0, "source_path_mismatch": 0,
            "shape_dtype_equal": True}
    monkeypatch.setattr(audit, "BACKBONES", ("resnet18",))
    monkeypatch.setattr(audit, "compare_stored_backbone", lambda *_: item)
    monkeypatch.setattr(audit, "determinism_settings", lambda: {})
    result = audit.run_audit(tmp_path, run_inference=False)
    assert result["test_access"] == 0
    assert not result["artifacts_modified"]
    assert result["cause"] == "UNRESOLVED"
