"""STEP 2-C ROI 확장, 설정, 출력 utility를 검증한다."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drowsiness_detection.preprocessing_v2.detectors import BoundingBox
from drowsiness_detection.preprocessing_v2.landmark_roi_margin_audit import (
    DECISION,
    CandidateResult,
    ExpandedRoi,
    _comparison_image,
    _render_report,
    _write_csv,
    describe_roi_audit_dry_run,
    expand_landmark_roi,
    load_roi_audit_config,
)
from drowsiness_detection.preprocessing_v2.landmarks import Dlib68Predictor, LandmarkResult
from drowsiness_detection.preprocessing_v2.geometric_features import RatioResult
from drowsiness_detection.preprocessing_v2.head_pose import HeadPoseResult


def test_config_and_dry_run_reuse_step2b_yunet_bbox() -> None:
    """동일 160개 중 YuNet 성공 157개만 사용하고 detector를 재실행하지 않아야 한다."""

    config = load_roi_audit_config(Path("configs/landmark_roi_margin_audit.yaml"), PROJECT_ROOT)
    plan = describe_roi_audit_dry_run(config)

    assert plan["source_frame_count"] == 160
    assert plan["yunet_success_frame_count"] == 157
    assert plan["split_counts"] == {"train": 80, "val": 77}
    assert plan["stored_yunet_bbox_reused"] is True
    assert plan["detector_rerun"] is False
    assert plan["test_split_used"] is False
    assert plan["decision"] == DECISION


def test_roi_expansion_is_symmetric_and_axis_independent() -> None:
    """정사각형과 비정사각형에서 width와 height margin을 독립 적용한다."""

    square = expand_landmark_roi(BoundingBox(100, 100, 300, 300), 0.05, 500, 500)
    rectangle = expand_landmark_roi(BoundingBox(100, 100, 300, 200), 0.10, 500, 500)

    assert square.bbox == BoundingBox(90, 90, 310, 310)
    assert rectangle.bbox == BoundingBox(80, 90, 320, 210)
    assert square.clipped is False


def test_zero_margin_clipping_and_invalid_bbox() -> None:
    """0%는 raw bbox와 같고 경계 초과분은 frame으로 clipping해야 한다."""

    raw = expand_landmark_roi(BoundingBox(10, 20, 110, 120), 0.0, 200, 200)
    clipped = expand_landmark_roi(BoundingBox(0, 0, 100, 100), 0.15, 200, 200)

    assert raw.bbox == BoundingBox(10, 20, 110, 120)
    assert raw.clipped is False
    assert clipped.bbox == BoundingBox(0, 0, 115, 115)
    assert clipped.clipped is True
    assert clipped.clipped_fraction > 0
    with pytest.raises(ValueError):
        expand_landmark_roi(BoundingBox(10, 10, 10, 20), 0.05, 200, 200)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        (BoundingBox(200, 150, 400, 350), BoundingBox(190, 140, 410, 360)),
        (BoundingBox(5, 150, 205, 350), BoundingBox(0, 140, 215, 360)),
        (BoundingBox(200, 5, 400, 205), BoundingBox(190, 0, 410, 215)),
        (BoundingBox(435, 150, 635, 350), BoundingBox(425, 140, 640, 360)),
        (BoundingBox(200, 275, 400, 475), BoundingBox(190, 265, 410, 480)),
    ],
)
def test_frame_boundary_clipping_cases(bbox: BoundingBox, expected: BoundingBox) -> None:
    """내부와 좌·상·우·하 초과를 같은 반개방 정수 좌표계에서 구분한다."""

    result = expand_landmark_roi(bbox, 0.05, 640, 480)
    assert result.bbox == expected
    if bbox.x1 == 200 and bbox.y1 == 150:
        assert result.clipped is False
        assert result.clipped_fraction == 0.0
    else:
        assert result.clipped is True
        assert result.clipped_fraction > 0.0


def test_exact_boundary_is_not_clipping() -> None:
    """반개방 ROI 끝이 frame width와 정확히 같아도 손실은 없다."""

    result = expand_landmark_roi(BoundingBox(0, 100, 640, 300), 0.0, 640, 480)
    assert result.bbox == BoundingBox(0, 100, 640, 300)
    assert result.clipped is False
    assert result.clipped_fraction == 0.0


def test_fractional_rounding_is_not_frame_clipping() -> None:
    """float expansion의 floor/ceil 차이만으로 clipping flag가 켜지지 않는다."""

    result = expand_landmark_roi(BoundingBox(100, 100, 109, 201), 0.05, 640, 480)
    assert result.bbox == BoundingBox(99, 94, 110, 207)
    assert result.clipped is False
    assert result.clipped_fraction == 0.0


def test_clipped_fraction_matches_integer_area_loss() -> None:
    """의도한 반개방 ROI 면적 중 frame 밖으로 제거된 비율을 검증한다."""

    result = expand_landmark_roi(BoundingBox(0, 50, 100, 150), 0.10, 200, 200)
    assert result.bbox == BoundingBox(0, 40, 110, 160)
    assert result.clipped is True
    assert result.clipped_fraction == pytest.approx(1.0 - (110 * 120) / (120 * 120))


def test_dlib_predictor_uses_inclusive_rectangle_endpoints() -> None:
    """반개방 bbox의 x2/y2를 dlib inclusive 끝점으로 변환한다."""

    captured: dict[str, tuple[int, int, int, int]] = {}

    class FakeDlib:
        @staticmethod
        def rectangle(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
            captured["rectangle"] = (x1, y1, x2, y2)
            return captured["rectangle"]

    class Point:
        x = 1
        y = 2

    class Shape:
        num_parts = 68

        @staticmethod
        def part(_index: int) -> Point:
            return Point()

    predictor = Dlib68Predictor.__new__(Dlib68Predictor)
    predictor._dlib = FakeDlib()  # type: ignore[attr-defined]
    predictor._predictor = lambda _frame, _rectangle: Shape()  # type: ignore[attr-defined]
    result = predictor.predict(np.zeros((100, 100, 3), dtype=np.uint8), BoundingBox(10, 20, 50, 70))

    assert result.success is True
    assert captured["rectangle"] == (10, 20, 49, 69)


def test_csv_report_and_visualization_smoke() -> None:
    """CSV/report writer와 5-panel visualization을 합성 입력으로 확인한다."""

    points = np.column_stack((np.linspace(30, 90, 68), np.linspace(20, 80, 68)))
    result = CandidateResult(
        ExpandedRoi(BoundingBox(15, 15, 105, 95), False, 0.0),
        LandmarkResult(True, points, 1.0), RatioResult(0.3, True), RatioResult(0.4, True),
        HeadPoseResult(True, 180.0, 1.0, 2.0, None, 1.0), True, 1.0, 2.0, "",
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    image = _comparison_image(
        frame, "sample", BoundingBox(20, 20, 100, 100),
        {name: result for name in ("RAW", "M05", "M10", "M15")}, (160, 120),
    )
    summary = {
        "scope": {"source_frames": 160, "yunet_success_frames": 157, "unique_videos": 32},
        "candidates": {name: {
            "landmark_success": 157, "count": 157, "geometry_valid": 157,
            "ear_valid": 157, "mar_valid": 157, "pose_success": 157,
            "roi_clipped": 0, "pipeline_ms": {"mean": 2.0, "median": 2.0, "p95": 2.0},
        } for name in ("raw", "m05", "m10", "m15")},
        "candidate_differences": {pair: {
            feature: {"median": 0.0, "p95": 0.0}
            for feature in ("ear", "mar", "pitch_raw", "yaw", "roll")
        } for pair in ("raw_m05", "raw_m10", "raw_m15", "m05_m10", "m10_m15")},
        "visual_review": {"sample_count": 35, "manual_csv": "roi_margin_manual_review.csv"},
    }

    assert image.shape == (120, 800, 3)
    assert DECISION in _render_report(summary)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
        root = Path(temporary_directory)
        assert cv2.imwrite(str(root / "comparison.jpg"), image)
        _write_csv(root / "result.csv", ("sample_id", "value"), [{"sample_id": "s1", "value": 1}])
        with (root / "result.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            assert list(csv.DictReader(handle))[0]["sample_id"] == "s1"
