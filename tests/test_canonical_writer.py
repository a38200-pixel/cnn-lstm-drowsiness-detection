"""원자적 발행·결측 보존·JPEG 채널 순서를 synthetic 자료로 검사한다."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness_detection.preprocessing_v2.canonical_integrity import validate_bundle
from drowsiness_detection.preprocessing_v2.canonical_integrity import validate_global_rows
from drowsiness_detection.preprocessing_v2.canonical_schema import empty_frame_row
from drowsiness_detection.preprocessing_v2.canonical_writer import publish_bundle, resume_status, write_rgb_jpeg
from drowsiness_detection.preprocessing_v2.canonical_sampling import context_indices


def fixture_rows() -> list[dict]:
    slots = {index: slot for slot, index in enumerate(context_indices())}
    rows = [empty_frame_row("v1", "train", "drowsy", index, index / 10, index * 3, 30.0, 300,
                            slots.get(index), False) for index in range(100)]
    for row in rows:
        row["decode_status"] = "FRAME_DECODE_FAILED"
        row["failure_flags"] = "FRAME_DECODE_FAILED"
    return rows


def test_atomic_and_resume(tmp_path: Path) -> None:
    rows = fixture_rows()
    points = np.full((100, 68, 2), np.nan, dtype=np.float32)
    marker = {"video_id": "v1", "policy_hash": "p", "run_config_hash": "r",
              "source_path": "source", "source_file_size": 1, "source_mtime_ns": 2,
              "canonical_row_count": 100}
    summary = {"video_id": "v1", "split": "train", "label": "drowsy", "bundle_status": "COMPLETE"}
    final = publish_bundle(tmp_path, "train", "v1", rows, points, {}, summary, marker)
    assert final.exists() and not validate_bundle(final)
    assert resume_status(final, "p", {"source_path": "source", "source_file_size": 1, "source_mtime_ns": 2}, True) == "SKIP"
    assert resume_status(final, "other", {"source_path": "source"}, True) == "CONFLICT_PROVENANCE"
    assert resume_status(final, "p", {"source_path": "source", "source_file_size": 2}, True) == "CONFLICT_PROVENANCE"
    with pytest.raises(FileExistsError):
        publish_bundle(tmp_path, "train", "v1", rows, points, {}, summary, marker)


def test_invalid_bundle_never_published(tmp_path: Path) -> None:
    rows = fixture_rows()[:99]
    points = np.full((100, 68, 2), np.nan, dtype=np.float32)
    with pytest.raises((ValueError, IndexError)):
        publish_bundle(tmp_path, "train", "v1", rows, points, {}, {}, {})
    assert not (tmp_path / "per_video" / "train" / "v1").exists()


def test_rgb_jpeg_roundtrip(tmp_path: Path) -> None:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[:, :112] = (255, 0, 0)
    rgb[:, 112:] = (0, 0, 255)
    path = tmp_path / "colors.jpg"
    write_rgb_jpeg(path, rgb)
    decoded = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    assert decoded[100, 20, 0] > 240 and decoded[100, 20, 2] < 15
    assert decoded[100, 200, 2] > 240 and decoded[100, 200, 0] < 15


def test_global_rejects_test_and_duplicate() -> None:
    videos = [{"video_id": "v1", "split": "train"}, {"video_id": "v1", "split": "val"}]
    assert "DUPLICATE_VIDEO_ID" in validate_global_rows([], videos)
    assert "TEST_OR_UNKNOWN_SPLIT" in validate_global_rows([{"video_id": "v2", "split": "test"}], [])
