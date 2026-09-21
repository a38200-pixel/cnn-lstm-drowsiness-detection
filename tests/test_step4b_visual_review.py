"""STEP 4-B 후보·프레임 선택·이미지 출처·미판정 CSV를 검증한다."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness_detection.evaluation_v2 import step4b_visual_review as review


def _candidate(reason: str = "highest_missing_count", review_order: int = 0) -> dict[str, str]:
    return {"review_order": str(review_order), "video_id": f"v{review_order}", "split": "train",
            "label": "drowsy", "selection_reason": reason, "recommended_frame_indices": "1;2;3",
            "yunet_missing_count": "1", "yunet_missing_rate": "0.01", "longest_missing_run": "1",
            "context_missing_count": "0", "behavior_valid_rate": "0.99"}


def _frames(missing: set[int] | None = None, context_missing: set[int] | None = None) -> list[dict[str, str]]:
    missing = missing or set()
    context_missing = context_missing or set()
    return [{"video_id": "v0", "canonical_index": str(index), "yunet_success": str(index not in missing),
             "context_selected": str(index in {0, 3, 50, 99} or index in context_missing),
             "context_crop_available": str(index in {0, 3, 50, 99} and index not in context_missing)}
            for index in range(100)]


def test_candidate_validation_rejects_duplicates_and_test() -> None:
    rows = [_candidate(review_order=index) for index in range(36)]
    assert len(review.validate_candidates(rows)) == 36
    rows[2]["video_id"] = rows[1]["video_id"]
    with pytest.raises(ValueError, match="중복"):
        review.validate_candidates(rows)
    rows[2]["video_id"] = "v2"
    rows[2]["split"] = "test"
    with pytest.raises(ValueError, match="STEP_4B_BLOCKED_TEST_LEAKAGE"):
        review.validate_candidates(rows)


def test_frame_selection_run_isolated_full_normal_and_context() -> None:
    assert review.select_frame_indices(_candidate(), _frames(set(range(20, 31)))) == [19, 20, 25, 30, 31]
    assert review.select_frame_indices(_candidate("longest_consecutive_run"), _frames(set(range(0, 8)))) == [0, 3, 7, 8]
    assert review.select_frame_indices(_candidate(), _frames(set(range(100)))) == [0, 25, 50, 75, 99]
    assert review.select_frame_indices(_candidate("isolated_single_miss"), _frames({42})) == [41, 42, 43]
    assert review.select_frame_indices(_candidate("zero_missing_reference"), _frames()) == [0, 50, 99]
    context = review.select_frame_indices(_candidate("highest_context_missing"), _frames({20}, {10, 40, 90}))
    assert {10, 40, 90} <= set(context)
    assert len(context) <= 5


def test_frame_selection_is_deterministic_and_bounded() -> None:
    candidate = _candidate("longest_consecutive_run")
    frames = _frames(set(range(10, 50)))
    first = review.select_frame_indices(candidate, frames)
    assert first == review.select_frame_indices(candidate, frames)
    assert len(first) <= 5


def test_stored_bbox_only_on_yunet_success() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    row = {"video_id": "v0", "canonical_index": "4", "actual_timestamp_sec": "0.4",
           "source_frame_index": "12", "yunet_success": "True", "context_selected": "False",
           "context_crop_available": "False", "bbox_x1": "20", "bbox_y1": "20",
           "bbox_x2": "80", "bbox_y2": "80", "yunet_confidence": "0.9"}
    success = review.render_panel(row, frame, None)
    assert success[92 + 48, 5 + 48, 1] > 100
    row["yunet_success"] = "False"
    row["bbox_x1"] = row["bbox_y1"] = row["bbox_x2"] = row["bbox_y2"] = ""
    missing = review.render_panel(row, frame, None)
    assert missing[92 + 48, 5 + 48, 1] == 0


def test_existing_context_jpeg_is_read_only(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    crop = bundle / "context_crops/ctx_00_k000.jpg"
    crop.parent.mkdir(parents=True)
    ok, encoded = cv2.imencode(".jpg", np.zeros((224, 224, 3), dtype=np.uint8))
    assert ok
    crop.write_bytes(encoded.tobytes())
    original = crop.read_bytes()
    row = {"context_selected": "True", "context_crop_available": "True",
           "context_crop_relpath": "context_crops/ctx_00_k000.jpg"}
    assert review._crop_image(bundle, row).shape == (224, 224, 3)
    assert crop.read_bytes() == original
    row["context_crop_relpath"] = "context_crops/missing.jpg"
    with pytest.raises(FileNotFoundError, match="PIPELINE REVIEW BLOCKER"):
        review._crop_image(bundle, row)


def test_pack_generation_index_contact_manual_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = []
    for index in range(36):
        candidate = _candidate("zero_missing_reference", index)
        candidate.update({"yunet_missing_count": "0", "yunet_missing_rate": "0",
                          "longest_missing_run": "0", "behavior_valid_rate": "1"})
        frames = _frames()
        for row in frames:
            row.update({"source_frame_index": row["canonical_index"], "frame_width": "8", "frame_height": "8"})
        plan.append({"candidate": candidate, "quality": {"number_of_yunet_missing_runs": "0", "context_missing_rate": "0"},
                     "frames": frames, "selected_indices": [0, 50, 99], "selected_rows": [frames[k] for k in (0, 50, 99)],
                     "source": tmp_path / f"v{index}.mp4", "bundle": tmp_path})
    opened: list[int] = []
    def fake_read(source: Path, rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
        opened.append(len(rows))
        return {int(row["source_frame_index"]): np.zeros((8, 8, 3), dtype=np.uint8) for row in rows}
    monkeypatch.setattr(review, "_read_source_frames", fake_read)
    monkeypatch.setattr(review, "render_individual", lambda item, raw: np.zeros((80, 120, 3), dtype=np.uint8))
    output = tmp_path / "review"
    summary = review.generate_pack(plan, output)
    assert summary["candidate_count"] == summary["individual_sheet_count"] == 36
    assert summary["contact_sheet_count"] == 9
    assert summary["frames_extracted"] == 108
    assert len(opened) == 36
    with (output / "step4b_review_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    with (output / "step4b_manual_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        manual_rows = list(csv.DictReader(handle))
    assert len(index_rows) == len(manual_rows) == 36
    assert len({row["review_id"] for row in index_rows}) == 36
    assert all((output / row["individual_sheet_path"]).is_file() for row in index_rows)
    assert all((output / "contact_sheets" / row["contact_sheet_id"]).is_file() for row in index_rows)
    assert all(all(row[field] == "" for field in review.MANUAL_FIELDS) for row in manual_rows)
    assert "MANUAL REVIEW: WAITING" in (output / "step4b_review_pack_report.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="덮어쓰지"):
        review.generate_pack(plan, output)


def test_dry_run_does_not_open_video_or_write(tmp_path: Path) -> None:
    plan = [{"candidate": _candidate(review_order=index), "selected_indices": [1, 2, 3]} for index in range(36)]
    output = tmp_path / "review"
    result = review.dry_run(plan, output)
    assert result["raw_video_opened"] is False
    assert result["output_written"] is False
    assert not output.exists()
