"""실제 SUST 영상을 쓰지 않고 실패 전파와 분기 독립성을 검증한다."""

from types import SimpleNamespace

import numpy as np

from drowsiness_detection.preprocessing_v2.canonical_preprocessing import _process_frame, completed_resume_result, pilot_rows
from drowsiness_detection.preprocessing_v2.canonical_schema import empty_frame_row
from drowsiness_detection.preprocessing_v2.detectors import BoundingBox, FaceCandidate


CONTEXT = {"margin_ratio": 0.10, "output_size": 224, "rgb_mean": [0.485, 0.456, 0.406]}


def _row(selected: bool = True) -> dict:
    return empty_frame_row("v1", "train", "drowsy", 0, 0.0, 0, 30.0, 300,
                           0 if selected else None, False)


def test_detector_missing_propagates() -> None:
    detector = SimpleNamespace(detect=lambda frame: SimpleNamespace(success=False, selected=None,
                               elapsed_ms=1.0, detection_count=0, failure_reason="YUNET_FACE_NOT_FOUND"))
    row = _row()
    points, crop = _process_frame(np.zeros((300, 300, 3), np.uint8), row, detector, None, CONTEXT)
    assert points is None and crop is None
    assert not row["yunet_success"] and not row["landmark_success"]
    assert np.isnan(row["ear"]) and np.isnan(row["mar"])
    assert row["context_selected"] and not row["context_crop_available"]


def test_dlib_missing_still_allows_context() -> None:
    detector = SimpleNamespace(detect=lambda frame: SimpleNamespace(success=True,
                               selected=FaceCandidate(BoundingBox(75, 75, 225, 225), 0.95),
                               elapsed_ms=1.0, detection_count=1))
    predictor = SimpleNamespace(predict=lambda frame, bbox: SimpleNamespace(success=False, points=None, elapsed_ms=1.0))
    row = _row()
    points, crop = _process_frame(np.zeros((300, 300, 3), np.uint8), row, detector, predictor, CONTEXT)
    assert points is None and crop is not None and crop.shape == (224, 224, 3)
    assert row["yunet_success"] and not row["landmark_success"]
    assert row["context_crop_available"] and np.isnan(row["ear"])


def test_pilot_selection_is_deterministic() -> None:
    rows = [{"video_id": f"{split}_{label}_{index}", "split": split, "label": label}
            for split in ("train", "val") for label in ("drowsy", "not_drowsy") for index in range(10)]
    selected = pilot_rows(rows)
    assert selected == pilot_rows(list(reversed(rows)))
    assert len(selected) == 20 and all(row["split"] != "test" for row in selected)


def test_completed_resume_preserves_initial_report() -> None:
    original = {"video_count": 20, "frame_count": 2000, "policy_hash": "policy"}
    result = completed_resume_result(["SKIP"] * 20, original)
    assert result is not None
    assert result["completed_this_run"] == result["detector_frames"] == 0
    assert result["skipped"] == 20 and result["original_report_preserved"]
    assert completed_resume_result(["SKIP", "PROCESS"], original) is None


def test_pose_failure_keeps_ratios(monkeypatch) -> None:
    detector = SimpleNamespace(detect=lambda frame: SimpleNamespace(success=True,
                               selected=FaceCandidate(BoundingBox(75, 75, 225, 225), 0.95),
                               elapsed_ms=1.0, detection_count=1))
    landmarks = np.zeros((68, 2), np.float64)
    for start in (36, 42):
        landmarks[start:start + 6] = [(0, 0), (1, 1), (2, 1), (4, 0), (2, -1), (1, -1)]
    landmarks[60:68] = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 0), (3, -1), (2, -1), (1, -1)]
    predictor = SimpleNamespace(predict=lambda frame, bbox: SimpleNamespace(success=True, points=landmarks, elapsed_ms=1.0))
    monkeypatch.setattr("drowsiness_detection.preprocessing_v2.canonical_preprocessing.estimate_head_pose",
                        lambda *args: SimpleNamespace(success=False))
    row = _row(False)
    points, crop = _process_frame(np.zeros((300, 300, 3), np.uint8), row, detector, predictor, CONTEXT)
    assert points is not None and crop is None
    assert np.isfinite(row["ear"]) and np.isfinite(row["mar"])
    assert not row["pose_success"] and np.isnan(row["pitch_raw"])


def test_short_source_is_not_clamped(monkeypatch, tmp_path) -> None:
    from drowsiness_detection.preprocessing_v2.canonical_preprocessing import process_video

    class FakeCapture:
        reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            import cv2
            return 10 if prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT) else 0

        def read(self):
            self.reads += 1
            return True, np.zeros((32, 32, 3), np.uint8)

        def release(self):
            pass

    capture = FakeCapture()
    monkeypatch.setattr("drowsiness_detection.preprocessing_v2.canonical_preprocessing.cv2.VideoCapture",
                        lambda path: capture)
    detector = SimpleNamespace(detect=lambda frame: SimpleNamespace(success=False, selected=None,
                               elapsed_ms=1.0, detection_count=0, failure_reason="YUNET_FACE_NOT_FOUND"))
    rows, points, crops, summary, previews = process_video(tmp_path / "synthetic.mp4",
        {"video_id": "v1", "split": "train", "label": "drowsy"}, detector, None, CONTEXT)
    assert len(rows) == 100 and capture.reads == 10
    assert summary["out_of_range_slots"] == 90
    assert rows[99]["decode_status"] == "SOURCE_INDEX_OUT_OF_RANGE"
    assert len(crops) == len(previews) == 0 and np.isnan(points).all()
