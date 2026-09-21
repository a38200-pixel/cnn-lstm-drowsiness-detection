"""STEP 4-A의 결측 정의·연속 구간·안전한 출력 계약을 검증한다."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from drowsiness_detection.evaluation_v2.quality_missing_audit import (
    POLICY_HASH, _analyze_video, analyze, choose_candidates, concentration, cumulative_counts,
    dry_run, missing_runs, write_audit,
)


def sample_video(video_id: str, split: str, label: str, misses: set[int]) -> tuple[dict, list[dict]]:
    """원본 영상 없이 canonical 100행과 대응 summary를 합성한다."""
    selected = {index: slot for slot, index in enumerate(range(0, 96, 3))}
    rows = []
    for index in range(100):
        face = index not in misses
        context = index in selected
        rows.append({
            "video_id": video_id, "split": split, "label": label,
            "canonical_index": str(index), "target_timestamp_sec": str(index / 10),
            "source_frame_index": str(index), "source_index_duplicate": "False",
            "source_index_in_range": "True", "decode_ok": "True",
            "yunet_success": str(face), "landmark_success": str(face), "pose_success": str(face),
            "context_selected": str(context), "context_slot": str(selected[index]) if context else "",
            "context_crop_available": str(context and face), "context_padding_fraction": "0" if context and face else "nan",
            "ear": "0.3" if face else "nan", "mar": "0.2" if face else "nan",
            "pitch_raw": "1" if face else "nan", "pitch_centered_candidate": "1" if face else "nan",
            "yaw": "5" if face else "nan", "roll": "0" if face else "nan",
            "bbox_width": "20" if face else "nan", "bbox_height": "30" if face else "nan",
            "frame_width": "100", "frame_height": "100", "yunet_confidence": "0.95" if face else "nan",
        })
    context_miss = sum(index in misses for index in selected)
    video = {"video_id": video_id, "split": split, "label": label,
             "bundle_status": "COMPLETE", "policy_hash": POLICY_HASH,
             "yunet_success_count": str(100 - len(misses)), "yunet_failure_count": str(len(misses)),
             "landmark_success_count": str(100 - len(misses)), "landmark_failure_count": "0",
             "ear_valid_count": str(100 - len(misses)), "mar_valid_count": str(100 - len(misses)),
             "pose_valid_count": str(100 - len(misses)), "context_selected_count": "32",
             "context_crop_available_count": str(32 - context_miss),
             "context_crop_missing_count": str(context_miss), "decoded_slots": "100",
             "duplicate_source_index_count": "0", "multiple_face_frame_count": "0"}
    return video, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """합성 metadata CSV를 생성한다."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fixture_root(tmp_path: Path, misses_train: set[int] | None = None,
                 misses_val: set[int] | None = None) -> Path:
    """작은 train/val canonical metadata를 만든다."""
    root = tmp_path / "canonical"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    train, train_rows = sample_video("d_1", "train", "drowsy", misses_train or set())
    val, val_rows = sample_video("n_1", "val", "not_drowsy", misses_val or set())
    write_csv(metadata / "canonical_videos.csv", [train, val])
    write_csv(metadata / "canonical_frames.csv", train_rows + val_rows)
    (metadata / "run_metadata.json").write_text(json.dumps({"policy_hash": POLICY_HASH,
        "run_config_hash": "synthetic", "test_split_processed": False}), encoding="utf-8")
    return root


def test_missing_run_single_multiple_start_end_and_empty() -> None:
    assert missing_runs([True, True, False, True, False, True]) == [(0, 1), (3, 3), (5, 5)]
    assert missing_runs([False, False]) == []
    assert missing_runs([True, True]) == [(0, 1)]


def test_video_counts_context_runs_temporal_edges_and_neighbors() -> None:
    video, rows = sample_video("d_1", "train", "drowsy", {0, 1, 5, 99})
    result = _analyze_video(video, rows)
    quality = result["quality"]
    assert quality["yunet_missing_count"] == 4 and quality["yunet_missing_rate"] == 0.04
    assert quality["behavior_valid_count"] == quality["pose_valid_count"] == 96
    assert quality["context_missing_count"] == 1 and quality["context_missing_rate"] == 1 / 32
    assert quality["longest_yunet_missing_run"] == 2
    assert quality["number_of_yunet_missing_runs"] == 3
    assert quality["longest_context_missing_run"] == 1
    assert quality["missing_at_start"] and quality["missing_at_end"]
    assert [(run["start_index"], run["end_index"], run["run_length"]) for run in result["runs"]] == [(0, 1, 2), (5, 5, 1), (99, 99, 1)]
    assert result["runs"][0]["touches_clip_start"] and result["runs"][-1]["touches_clip_end"]
    assert result["runs"][0]["context_missing_slots_inside_run"] == 1
    assert result["neighbors"][0]["nearest_valid_before_yaw"] == ""
    assert result["neighbors"][0]["nearest_valid_after_yaw"] == 5.0


def test_cumulative_concentration_and_zero_missing() -> None:
    assert cumulative_counts([0, 1, 2], 2) == [
        {"max_missing_allowed": 0, "videos_satisfying": 1, "fraction": 1 / 3},
        {"max_missing_allowed": 1, "videos_satisfying": 2, "fraction": 2 / 3},
        {"max_missing_allowed": 2, "videos_satisfying": 3, "fraction": 1.0}]
    assert concentration([9, 1, 0])[0]["fraction_of_all_missing"] == 0.9
    assert concentration([0, 0])[0]["fraction_of_all_missing"] == 0
    video, rows = sample_video("n_1", "val", "not_drowsy", set())
    result = _analyze_video(video, rows)
    assert result["quality"]["yunet_missing_count"] == 0
    assert result["quality"]["context_missing_count"] == 0
    assert result["runs"] == []


def test_candidate_determinism_deduplication_and_normal_reference() -> None:
    videos = []
    runs = []
    for index in range(40):
        split = "train" if index % 2 else "val"
        label = "drowsy" if index % 4 < 2 else "not_drowsy"
        misses = {10, 11} if index < 20 else {10} if index < 30 else set()
        video, frames = sample_video(f"v{index:02d}", split, label, misses)
        result = _analyze_video(video, frames)
        videos.append(result["quality"])
        runs.extend(result["runs"])
    first = choose_candidates(videos, runs, target=36)
    assert first == choose_candidates(list(reversed(videos)), list(reversed(runs)), target=36)
    assert len(first) == len({row["video_id"] for row in first}) == 36
    assert {row["split"] for row in first} == {"train", "val"}
    assert {row["label"] for row in first} == {"drowsy", "not_drowsy"}
    assert "zero_missing_reference" in {row["selection_reason"] for row in first}


def test_dry_run_analysis_temporal_and_report_writer(tmp_path: Path) -> None:
    root = fixture_root(tmp_path, {0, 1, 5, 99}, {10})
    planned = dry_run(root, expected_splits={"train": 1, "val": 1})
    assert planned["video_count"] == 2 and planned["frame_count"] == 200
    assert planned["raw_video_opened"] is False and planned["output_written"] is False
    result = analyze(root, expected_splits={"train": 1, "val": 1})
    summary = result["summary"]
    assert summary["overall"]["yunet_missing"] == 5
    assert summary["overall"]["context_missing"] == 1
    assert summary["missing_runs"]["count"] == 4
    assert summary["numeric_anomaly_count"] == 0
    assert result["temporal"][0]["yunet_missing_count"] == 1
    assert result["context_temporal"][0]["missing_count"] == 1
    output = tmp_path / "review"
    write_audit(result, output)
    assert (output / "quality_missing_report.txt").is_file()
    assert "MISSING POLICY: NOT SELECTED" in (output / "quality_missing_report.txt").read_text(encoding="utf-8")
    assert len(list(csv.DictReader((output / "per_video_quality.csv").open(encoding="utf-8-sig")))) == 2
    with pytest.raises(FileExistsError):
        write_audit(result, output)


def test_test_split_and_policy_mismatch_block_before_output(tmp_path: Path) -> None:
    root = fixture_root(tmp_path)
    videos_path = root / "metadata/canonical_videos.csv"
    rows = list(csv.DictReader(videos_path.open(encoding="utf-8")))
    rows[1]["split"] = "test"
    write_csv(videos_path, rows)
    with pytest.raises(ValueError, match="STEP_4A_BLOCKED_TEST_LEAKAGE"):
        dry_run(root, expected_splits={"train": 1, "val": 1})
    assert not (tmp_path / "review").exists()
    rows[1]["split"] = "val"
    rows[1]["policy_hash"] = "wrong"
    write_csv(videos_path, rows)
    with pytest.raises(ValueError, match="policy hash"):
        dry_run(root, expected_splits={"train": 1, "val": 1})


def test_invalid_frame_structure_and_numeric_anomaly(tmp_path: Path) -> None:
    root = fixture_root(tmp_path, {10})
    frames_path = root / "metadata/canonical_frames.csv"
    rows = list(csv.DictReader(frames_path.open(encoding="utf-8")))
    rows[0]["canonical_index"] = "3"
    write_csv(frames_path, rows)
    with pytest.raises(ValueError, match="canonical index"):
        analyze(root, expected_splits={"train": 1, "val": 1})
    video, frames = sample_video("d_1", "train", "drowsy", {10})
    frames[0]["ear"] = "nan"
    video["ear_valid_count"] = "98"
    analyzed = _analyze_video(video, frames)
    assert analyzed["anomalies"][0]["issue"] == "LANDMARK_SUCCESS_NONFINITE_EAR_MAR"
    frames[0]["ear"] = "inf"
    analyzed = _analyze_video(video, frames)
    assert {row["issue"] for row in analyzed["anomalies"]} == {
        "LANDMARK_SUCCESS_NONFINITE_EAR_MAR", "NUMERIC_INFINITY"}


def test_context_slot_requirement_and_empty_anomaly_csv(tmp_path: Path) -> None:
    root = fixture_root(tmp_path)
    frames_path = root / "metadata/canonical_frames.csv"
    rows = list(csv.DictReader(frames_path.open(encoding="utf-8")))
    rows[0]["context_slot"] = "2"
    write_csv(frames_path, rows)
    with pytest.raises(ValueError, match="Context 32 slot"):
        dry_run(root, expected_splits={"train": 1, "val": 1})
    video, frames = sample_video("d_1", "train", "drowsy", set())
    assert _analyze_video(video, frames)["anomalies"] == []
