"""STEP 5-B Behavior 100-slot NaN·mask·출처·resume 테스트."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drowsiness_detection.sequences_v2.behavior_sequence_builder import (
    FEATURE_NAMES, FEATURE_SCHEMA_HASH, build_video_plan, is_b2_eligible,
    materialize, plan_statistics, validate_bundle, validate_video_plan,
)


def make_rows(missing: set[int] | None = None, *, split: str = "train",
              partial_pose: set[int] | None = None):
    missing, partial_pose = missing or set(), partial_pose or set()
    rows = []
    for index in range(100):
        absent = index in missing
        pose_absent = absent or index in partial_pose
        rows.append({
            "video_id": "v1", "split": split, "label": "drowsy",
            "canonical_index": str(index), "actual_timestamp_sec": str(index / 10),
            "yunet_success": str(not absent), "landmark_success": str(not absent),
            "pose_success": str(not pose_absent),
            "ear": "" if absent else str(0.2 + index / 10000),
            "mar": "" if absent else str(0.3 + index / 10000),
            "pitch_raw": "" if pose_absent else str(180 + index / 100),
            "pitch_centered_candidate": "" if pose_absent else str(index / 100),
            "yaw": "" if pose_absent else str(-10 + index / 100),
            "roll": "" if pose_absent else str(1 + index / 100),
        })
    return rows


def make_plan(missing: set[int] | None = None, *, context_eligible: bool = True,
              partial_pose: set[int] | None = None):
    return build_video_plan("v1", "train", "drowsy", make_rows(missing, partial_pose=partial_pose),
                            context_eligible)


def test_sequence_shapes_and_feature_order():
    plan = make_plan({3, 4})
    assert len(plan.rows) == 100
    assert plan.features.shape == (100, 6)
    assert FEATURE_NAMES == ("ear", "mar", "pitch_raw", "pitch_centered_candidate", "yaw", "roll")
    assert [row["canonical_index"] for row in plan.rows] == list(range(100))


@pytest.mark.parametrize("name", [
    "behavior_valid_mask", "detector_valid_mask", "landmark_valid_mask", "pose_valid_mask",
])
def test_sequence_mask_shape_and_dtype(name):
    mask = getattr(make_plan({8}), name)
    assert mask.shape == (100,)
    assert mask.dtype == np.bool_


@pytest.mark.parametrize("column,name", list(enumerate(FEATURE_NAMES)))
def test_valid_continuous_value_preserved_as_float32(column, name):
    rows = make_rows()
    plan = build_video_plan("v1", "train", "drowsy", rows, True)
    assert plan.features[23, column].tobytes() == np.float32(float(rows[23][name])).tobytes()


@pytest.mark.parametrize("column", range(6))
def test_missing_feature_remains_nan_without_interpolation(column):
    plan = make_plan({50})
    assert np.isnan(plan.features[50, column])
    assert not plan.feature_valid_mask[50, column]
    assert np.isfinite(plan.features[49, column]) and np.isfinite(plan.features[51, column])


@pytest.mark.parametrize("valid_count,longest,expected", [
    (95, 5, True), (100, 0, True), (94, 1, False),
    (99, 6, False), (95, 6, False), (94, 6, False),
])
def test_frozen_b2_boundary(valid_count, longest, expected):
    assert is_b2_eligible(valid_count, longest) is expected


@pytest.mark.parametrize("context_eligible", [True, False])
def test_context_eligibility_does_not_filter_behavior(context_eligible):
    plan = make_plan({2}, context_eligible=context_eligible)
    assert int(plan.behavior_valid_mask.sum()) == 99
    assert plan.context_eligible is context_eligible


def test_partial_pose_validity_preserves_ear_mar():
    plan = make_plan(partial_pose={7})
    assert plan.behavior_valid_mask[7]
    assert not plan.pose_valid_mask[7]
    assert np.isfinite(plan.features[7, :2]).all()
    assert np.isnan(plan.features[7, 2:]).all()


def test_test_split_rejected():
    with pytest.raises(ValueError, match="TEST_LEAKAGE"):
        build_video_plan("v1", "test", "drowsy", make_rows(split="test"), True)


def test_infinity_rejected():
    rows = make_rows()
    rows[3]["ear"] = "inf"
    with pytest.raises(ValueError, match="infinity"):
        build_video_plan("v1", "train", "drowsy", rows, True)


def test_canonical_numeric_mismatch_detected():
    rows = make_rows()
    plan = build_video_plan("v1", "train", "drowsy", rows, True)
    rows[9]["mar"] = "9.0"
    with pytest.raises(ValueError, match="canonical numeric mismatch"):
        validate_video_plan(plan, rows)


@pytest.mark.parametrize("key", ["input_videos", "eligible_videos", "frame_rows", "excluded_videos"])
def test_plan_count_mismatch_rejected(key):
    plan = make_plan({4})
    expected = {"input_videos": 1, "eligible_videos": 1, "excluded_videos": 0,
                "frame_rows": 100}
    expected[key] += 1
    with pytest.raises(ValueError, match=key):
        plan_statistics([plan], [], expected)


def test_atomic_bundle_reports_resume_and_schema_conflict(tmp_path):
    plan = make_plan({4, 5})
    expected = {"input_videos": 1, "eligible_videos": 1, "excluded_videos": 0,
                "frame_rows": 100}
    root, reports = tmp_path / "behavior", tmp_path / "reports"
    result = materialize([plan], [], root, reports, expected=expected)
    assert result["bundles_created_this_run"] == 1
    assert result["interpolation_count"] == 0
    bundle = root / "per_video/train/v1"
    validate_bundle(bundle, plan)
    assert (reports / "behavior_sequence_validity_histogram.csv").is_file()
    resumed = materialize([plan], [], root, reports, resume=True, expected=expected)
    assert resumed["bundles_skipped_this_run"] == 1
    marker_path = bundle / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["behavior_feature_schema_hash"] == FEATURE_SCHEMA_HASH
    marker["behavior_feature_schema_hash"] = "conflict"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="schema conflict"):
        materialize([plan], [], root, reports, resume=True, expected=expected)
