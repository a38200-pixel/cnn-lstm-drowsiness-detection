"""STEP 5-A Context 원본/대체 참조와 bundle 회귀 테스트."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness_detection.sequences_v2.context_sequence_builder import (
    FRAME_COLUMNS, POLICY_HASH, SEQUENCE_POLICY_HASH, build_video_plan,
    materialize, plan_statistics, validate_bundle, validate_video_plan,
)


def make_plan(tmp_path: Path, missing: set[int] | None = None,
              video_id: str = "v1", behavior_eligible: bool = False,
              write_images: bool = False):
    missing = missing or set()
    canonical = tmp_path / "canonical"
    bundle = canonical / "per_video/train" / video_id / "context_crops"
    if write_images:
        bundle.mkdir(parents=True)
    rows = []
    for index in range(32):
        relative = f"context_crops/ctx_{index:02d}.jpg"
        if write_images and index not in missing:
            assert cv2.imwrite(str(bundle / f"ctx_{index:02d}.jpg"),
                               np.zeros((224, 224, 3), dtype=np.uint8))
        rows.append({"video_id": video_id, "split": "train", "label": "drowsy",
                     "context_selected": "True", "context_slot": str(index),
                     "canonical_index": str(index * 3), "actual_timestamp_sec": str(index * 0.3),
                     "context_crop_available": str(index not in missing),
                     "context_crop_relpath": relative if index not in missing else ""})
    return build_video_plan(video_id, "train", "drowsy", rows, behavior_eligible,
                            canonical, tmp_path, check_images=write_images)


@pytest.mark.parametrize("missing,target,source", [
    ({10}, 10, 9), ({0}, 0, 1), ({31}, 31, 30),
    ({0, 1, 2}, 0, 3), ({29, 30, 31}, 31, 28),
    ({10, 11, 12, 13}, 12, 14),
])
def test_nearest_valid_and_edges(tmp_path, missing, target, source):
    plan = make_plan(tmp_path, missing)
    assert plan.rows[target]["source_context_index"] == source
    assert plan.rows[target]["source_image_relpath"] == plan.rows[source]["source_image_relpath"]
    assert plan.original_mask[source]
    assert not plan.original_mask[target]
    assert plan.imputed_mask[target]
    assert plan.rows[target]["target_timestamp_sec"] == target * 0.3
    assert plan.rows[target]["imputation_offset_context_slots"] == source - target


def test_tie_earlier_and_no_chain(tmp_path):
    plan = make_plan(tmp_path, {9, 10, 11})
    assert plan.rows[10]["source_context_index"] == 8
    assert plan.rows[9]["source_context_index"] == 8
    assert plan.rows[11]["source_context_index"] == 12
    assert all(plan.original_mask[row["source_context_index"]] for row in plan.rows)
    assert len(plan.rows) == len(plan.original_mask) == len(plan.imputed_mask) == 32
    assert [row["target_context_index"] for row in plan.rows] == list(range(32))
    assert all(not (valid and filled) and (valid or filled)
               for valid, filled in zip(plan.original_mask, plan.imputed_mask))


def test_self_reference_behavior_independent_and_deterministic(tmp_path):
    a = make_plan(tmp_path, {5}, behavior_eligible=False)
    b = make_plan(tmp_path, {5}, behavior_eligible=False)
    assert a == b
    assert a.behavior_eligible is False
    assert a.rows[4]["source_context_index"] == 4
    assert a.rows[4]["imputation_distance_context_slots"] == 0


@pytest.mark.parametrize("index", [0, 1, 15, 30, 31])
def test_original_valid_slot_self_reference(tmp_path, index):
    plan = make_plan(tmp_path, {7})
    row = plan.rows[index]
    if index == 7:
        pytest.fail("검사 index는 원본-valid여야 함")
    assert row["source_context_index"] == index
    assert row["source_canonical_index"] == row["target_canonical_index"]
    assert row["imputation_distance_sec"] == 0


@pytest.mark.parametrize("missing", [{1}, {2, 3}, {8, 9, 10}, {18, 19, 20, 21}, {30}, {0, 1}])
def test_every_source_is_original_and_same_video(tmp_path, missing):
    plan = make_plan(tmp_path, missing)
    for row in plan.rows:
        source = row["source_context_index"]
        assert plan.original_mask[source]
        assert row["video_id"] == plan.video_id
        assert row["source_image_relpath"].startswith(
            f"canonical/per_video/train/{plan.video_id}/context_crops/")


@pytest.mark.parametrize("key", [
    "eligible_videos", "frame_rows", "original_valid_rows", "imputed_rows",
])
def test_plan_statistics_rejects_count_mismatch(tmp_path, key):
    plan = make_plan(tmp_path, {4})
    expected = {"input_videos": 1, "eligible_videos": 1, "excluded_videos": 0,
                "frame_rows": 32, "original_valid_rows": 31, "imputed_rows": 1,
                "videos_without_imputation": 0, "videos_requiring_imputation": 1}
    expected[key] += 1
    with pytest.raises(ValueError, match=key):
        plan_statistics([plan], [], expected)


def test_reject_invalid_metadata_and_source(tmp_path):
    plan = make_plan(tmp_path, {5})
    with pytest.raises(ValueError, match="nearest source"):
        plan.rows[5]["source_context_index"] = 7
        validate_video_plan(plan, tmp_path, check_images=False)
    plan = make_plan(tmp_path, {5})
    plan.rows[5]["canonical_policy_hash"] = "wrong"
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        validate_video_plan(plan, tmp_path, check_images=False)
    plan = make_plan(tmp_path, {5})
    with pytest.raises(FileNotFoundError, match="source JPEG"):
        validate_video_plan(plan, tmp_path, check_images=True)
    with pytest.raises(ValueError, match="TEST_LEAKAGE"):
        build_video_plan("v1", "test", "drowsy", [], False,
                         tmp_path / "canonical", tmp_path, check_images=False)
    with pytest.raises(ValueError, match="NO_VALID_SOURCE"):
        make_plan(tmp_path, set(range(32)))


def test_materialize_resume_reports_and_conflict(tmp_path):
    plan = make_plan(tmp_path, {4, 5}, write_images=True)
    expected = {"input_videos": 1, "eligible_videos": 1, "excluded_videos": 0,
                "frame_rows": 32, "original_valid_rows": 30, "imputed_rows": 2,
                "videos_without_imputation": 0, "videos_requiring_imputation": 1}
    root, reports = tmp_path / "sequences", tmp_path / "reports"
    result = materialize([plan], [], root, reports, tmp_path, expected=expected, decode_limit=4)
    assert result["imputed_rows"] == 2
    assert result["bundles_created_this_run"] == 1
    bundle = root / "per_video/train/v1"
    validate_bundle(bundle, plan)
    assert not list(root.rglob("*.jpg"))
    with (root / "metadata/context_sequence_frames.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 32
    assert (reports / "context_sequence_imputation_histogram.csv").is_file()
    assert (reports / "context_sequence_split_label_summary.csv").is_file()
    resumed = materialize([plan], [], root, reports, tmp_path, resume=True,
                          expected=expected, decode_limit=4)
    assert resumed["bundles_skipped_this_run"] == 1
    marker_path = bundle / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["source_metadata_fingerprint"] = "conflict"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        materialize([plan], [], root, reports, tmp_path, resume=True,
                    expected=expected, decode_limit=4)
