"""실제 영상을 열지 않고 Context 결측 pack의 범위와 저장을 검증한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness_detection.preprocessing_v2 import missing_context_review as review


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """테스트용 최소 CSV를 만든다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pilot_fixture(root: Path, missing: bool = True) -> Path:
    """결측 한 건 또는 결측 없는 완료 bundle을 합성한다."""

    pilot = root / "canonical_pilot"
    source = root / "source.mp4"
    source.write_bytes(b"synthetic placeholder")
    video = {"video_id": "n_1", "split": "train", "label": "not_drowsy", "source_path": str(source)}
    frame = {"video_id": "n_1", "split": "train", "label": "not_drowsy", "canonical_index": "13",
             "context_slot": "4", "source_frame_index": "39", "target_timestamp_sec": "1.3",
             "actual_timestamp_sec": "1.3", "context_selected": "True",
             "context_crop_available": "False" if missing else "True", "decode_ok": "True",
             "decode_status": "OK", "yunet_success": "False", "yunet_confidence": "nan",
             "yunet_detection_count": "0", "detector_status": "YUNET_FACE_NOT_FOUND",
             "landmark_success": "False", "landmark_status": "NOT_RUN", "pose_success": "False",
             "pose_status": "NOT_RUN", "context_status": "CONTEXT_NO_FACE", "failure_flags": "YUNET_FACE_NOT_FOUND",
             "bbox_x1": "nan", "bbox_y1": "nan", "bbox_x2": "nan", "bbox_y2": "nan",
             "frame_width": "320", "frame_height": "240"}
    _write_rows(pilot / "metadata/canonical_frames.csv", [frame])
    _write_rows(pilot / "metadata/canonical_videos.csv", [video])
    _write_rows(pilot / "metadata/input_manifest.csv", [video])
    bundle = pilot / "per_video/train/n_1"
    _write_rows(bundle / "frames.csv", [frame])
    (bundle / "COMPLETE.json").write_text(json.dumps({"policy_hash": "hash"}), encoding="utf-8")
    return pilot


def test_filter_and_reject_test_split() -> None:
    rows = [
        {"split": "train", "video_id": "a", "canonical_index": "1", "context_selected": "True", "context_crop_available": "False"},
        {"split": "val", "video_id": "b", "canonical_index": "2", "context_selected": "False", "context_crop_available": "False"},
        {"split": "train", "video_id": "c", "canonical_index": "3", "context_selected": "True", "context_crop_available": "True"},
    ]
    assert [row["video_id"] for row in review.select_missing_rows(rows)] == ["a"]
    with pytest.raises(ValueError):
        review.select_missing_rows(rows + [{"split": "test", "context_selected": "True", "context_crop_available": "False"}])


def test_suspected_reason_follows_stage_state() -> None:
    row = {"decode_ok": "True", "yunet_success": "False", "context_selected": "True", "context_crop_available": "False"}
    assert review.suspected_reason(row) == "DETECTOR_MISS"
    row["yunet_success"] = "True"
    assert review.suspected_reason(row) == "PIPELINE_OR_STORAGE_SUSPECT"
    row["landmark_success"] = "True"
    assert review.suspected_reason(row) == "PIPELINE_OR_STORAGE_SUSPECT"
    row["decode_ok"] = "False"
    assert review.suspected_reason(row) == "SOURCE_FRAME_UNAVAILABLE"


def test_generate_reports_images_and_preserve_input(tmp_path: Path, monkeypatch) -> None:
    pilot = _pilot_fixture(tmp_path)
    original = (pilot / "metadata/canonical_frames.csv").read_bytes()
    monkeypatch.setattr(review, "_source_frame", lambda path, indices: {39: np.zeros((240, 320, 3), dtype=np.uint8)})
    output = tmp_path / "reports/missing_context_review"
    summary = review.generate_review(pilot, output, expected_count=1)
    assert summary["total_context_missing"] == 1 and summary["visual_artifacts"]["sheet_count"] == 1
    assert len(list((output / "samples").glob("*.jpg"))) == 1
    sheet = cv2.imread(str(output / "missing_context_sheet_01.jpg"))
    assert sheet is not None and sheet.shape[0] == 438
    with (output / "missing_context_manual_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        manual = list(csv.DictReader(handle))
    assert manual[0]["manual_category"] == manual[0]["review_decision"] == ""
    assert manual[0]["suspected_reason_auto"] == "DETECTOR_MISS"
    for name in ("missing_context_rows.csv", "missing_context_summary.json", "missing_context_report.txt",
                 "MISSING_CONTEXT_REVIEW_GUIDE.md"):
        assert (output / name).is_file()
    assert (pilot / "metadata/canonical_frames.csv").read_bytes() == original
    with pytest.raises(FileExistsError):
        review.generate_review(pilot, output)
    with pytest.raises(ValueError):
        review.generate_review(pilot, pilot / "missing_context_review")


def test_zero_missing_is_graceful(tmp_path: Path, monkeypatch) -> None:
    pilot = _pilot_fixture(tmp_path, missing=False)
    monkeypatch.setattr(review, "_source_frame", lambda *args: (_ for _ in ()).throw(AssertionError("영상 접근 금지")))
    output = tmp_path / "reports/missing_context_review"
    summary = review.generate_review(pilot, output, expected_count=0)
    assert summary["total_context_missing"] == 0
    assert summary["missing_rate"] == 0 and summary["visual_artifacts"]["sheet_count"] == 0
    assert not list((output / "samples").glob("*.jpg"))
    assert "결측이 없어" in (output / "missing_context_report.txt").read_text(encoding="utf-8")
