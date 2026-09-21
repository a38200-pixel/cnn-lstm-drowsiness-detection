"""STEP 5-C branch 관계·구조·mask·hash·불변성 감사 테스트."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drowsiness_detection.sequences_v2.behavior_sequence_builder import build_video_plan as build_behavior
from drowsiness_detection.sequences_v2.context_sequence_builder import build_video_plan as build_context
from drowsiness_detection.sequences_v2.cross_branch_audit import (
    POLICY_HASH, SEQUENCE_POLICY_HASH, _assert_global_manifest, _bundle_ids,
    _cross_summary, _tree_fingerprint, _validate_hash_document,
    behavior_semantic_anomalies, classify_branch, validate_context_pattern,
)


def context_plan(tmp_path, video_id="v1"):
    rows = [{"video_id": video_id, "split": "train", "label": "drowsy",
             "context_selected": "True", "context_slot": str(index),
             "canonical_index": str(index * 3), "actual_timestamp_sec": str(index * 0.3),
             "context_crop_available": "True",
             "context_crop_relpath": f"context_crops/{index}.jpg"} for index in range(32)]
    return build_context(video_id, "train", "drowsy", rows, True,
                         tmp_path / "canonical", tmp_path, check_images=False)


def behavior_plan():
    rows = [{"video_id": "v1", "split": "train", "label": "drowsy",
             "canonical_index": str(index), "actual_timestamp_sec": str(index / 10),
             "yunet_success": "True", "landmark_success": "True", "pose_success": "True",
             "ear": "0.2", "mar": "0.3", "pitch_raw": "180",
             "pitch_centered_candidate": "0", "yaw": "-1", "roll": "1"}
            for index in range(100)]
    return build_behavior("v1", "train", "drowsy", rows, True)


@pytest.mark.parametrize("context,behavior,status", [
    (True, True, "BOTH"), (True, False, "CONTEXT_ONLY"),
    (False, True, "BEHAVIOR_ONLY"), (False, False, "NEITHER"),
])
def test_branch_classification(context, behavior, status):
    assert classify_branch(context, behavior) == status


def test_context_target_pattern_consistent(tmp_path):
    first, second = context_plan(tmp_path, "v1"), context_plan(tmp_path, "v2")
    assert validate_context_pattern([first, second]) == tuple(index * 3 for index in range(32))


def test_context_target_pattern_mismatch(tmp_path):
    first, second = context_plan(tmp_path, "v1"), context_plan(tmp_path, "v2")
    second.rows[5]["target_canonical_index"] = 16
    with pytest.raises(ValueError, match="pattern mismatch"):
        validate_context_pattern([first, second])


@pytest.mark.parametrize("index,value", [(1, 0), (31, 100), (4, -1)])
def test_context_target_pattern_invalid(tmp_path, index, value):
    plan = context_plan(tmp_path)
    plan.rows[index]["target_canonical_index"] = value
    with pytest.raises(ValueError, match="pattern invalid"):
        validate_context_pattern([plan])


def test_behavior_semantics_clean():
    assert behavior_semantic_anomalies(behavior_plan()) == {
        "false_mask_with_required_finite": 0, "required_valid_nan": 0,
        "feature_valid_mask_mismatch": 0, "infinity": 0}


def test_behavior_false_mask_finite_detected():
    plan = behavior_plan()
    plan.behavior_valid_mask[3] = False
    assert behavior_semantic_anomalies(plan)["false_mask_with_required_finite"] == 1


def test_behavior_required_nan_detected():
    plan = behavior_plan()
    plan.features[3, 0] = np.nan
    assert behavior_semantic_anomalies(plan)["required_valid_nan"] == 1


def test_behavior_feature_mask_mismatch_detected():
    plan = behavior_plan()
    plan.feature_valid_mask[3, 0] = False
    assert behavior_semantic_anomalies(plan)["feature_valid_mask_mismatch"] == 1


def test_behavior_infinity_detected():
    plan = behavior_plan()
    plan.features[3, 0] = np.inf
    assert behavior_semantic_anomalies(plan)["infinity"] == 1


@pytest.mark.parametrize("branch,slots", [("Context", 32), ("Behavior", 100)])
def test_global_manifest_valid(tmp_path, branch, slots):
    plan = context_plan(tmp_path) if branch == "Context" else behavior_plan()
    rows = [{key: str(value) for key, value in row.items()} for row in plan.rows]
    _assert_global_manifest(rows, {"v1": plan}, branch, slots)


@pytest.mark.parametrize("branch,slots", [("Context", 32), ("Behavior", 100)])
def test_global_manifest_row_count_rejected(tmp_path, branch, slots):
    plan = context_plan(tmp_path) if branch == "Context" else behavior_plan()
    rows = [{key: str(value) for key, value in row.items()} for row in plan.rows[:-1]]
    with pytest.raises(ValueError, match="global slots"):
        _assert_global_manifest(rows, {"v1": plan}, branch, slots)


@pytest.mark.parametrize("branch,index_name", [
    ("Context", "target_context_index"), ("Behavior", "canonical_index")])
def test_global_manifest_duplicate_rejected(tmp_path, branch, index_name):
    plan = context_plan(tmp_path) if branch == "Context" else behavior_plan()
    rows = [{key: str(value) for key, value in row.items()} for row in plan.rows]
    rows[1][index_name] = rows[0][index_name]
    with pytest.raises(ValueError, match="global slots"):
        _assert_global_manifest(rows, {"v1": plan}, branch, len(rows))


@pytest.mark.parametrize("branch,slots", [("Context", 32), ("Behavior", 100)])
def test_global_manifest_content_rejected(tmp_path, branch, slots):
    plan = context_plan(tmp_path) if branch == "Context" else behavior_plan()
    rows = [{key: str(value) for key, value in row.items()} for row in plan.rows]
    rows[0]["label"] = "not_drowsy"
    with pytest.raises(ValueError, match="global content"):
        _assert_global_manifest(rows, {"v1": plan}, branch, slots)


@pytest.mark.parametrize("status", ["BOTH", "CONTEXT_ONLY", "BEHAVIOR_ONLY", "NEITHER"])
def test_cross_summary_contains_each_status(status):
    rows = [{"video_id": "v1", "split": "train", "label": "drowsy",
             "cross_branch_status": status}]
    overall = [row for row in _cross_summary(rows)
               if row["scope"] == "overall" and row["status"] == status][0]
    assert overall["video_count"] == 1 and overall["fraction"] == 1


def test_tree_fingerprint_deterministic(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    assert _tree_fingerprint(tmp_path) == _tree_fingerprint(tmp_path)


def test_tree_fingerprint_detects_mutation(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    before = _tree_fingerprint(tmp_path)
    path.write_text("changed", encoding="utf-8")
    assert before != _tree_fingerprint(tmp_path)


def test_bundle_ids_rejects_test_split(tmp_path):
    (tmp_path / "per_video/test").mkdir(parents=True)
    with pytest.raises(ValueError, match="TEST_LEAKAGE"):
        _bundle_ids(tmp_path, "Context")


def test_bundle_ids_rejects_incomplete(tmp_path):
    (tmp_path / "per_video/train/v1").mkdir(parents=True)
    with pytest.raises(ValueError, match="incomplete"):
        _bundle_ids(tmp_path, "Context")


def test_bundle_ids_rejects_duplicate_across_splits(tmp_path):
    for split in ("train", "val"):
        bundle = tmp_path / "per_video" / split / "v1"
        bundle.mkdir(parents=True)
        (bundle / "COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _bundle_ids(tmp_path, "Context")


def test_hash_document_valid(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"canonical_policy_hash": POLICY_HASH,
                                "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
                                "test_rows": 0}), encoding="utf-8")
    _validate_hash_document(path, "Context")


@pytest.mark.parametrize("field,value", [
    ("canonical_policy_hash", "wrong"), ("sequence_missing_policy_hash", "wrong"),
    ("test_rows", 1),
])
def test_hash_or_test_mismatch_rejected(tmp_path, field, value):
    data = {"canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH, "test_rows": 0}
    data[field] = value
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash/test mismatch"):
        _validate_hash_document(path, "Context")
