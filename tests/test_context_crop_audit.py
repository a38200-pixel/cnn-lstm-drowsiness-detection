"""STEP 2-D crop geometry, 색상, 출력과 기존 표본 재사용을 검증한다."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.context_crop_audit import (
    BASE_COLUMNS, DECISION, RESULT_COLUMNS, _record, _report, _select_visual_rows,
    _summary, _visual_image, _write_csv, crop_geometry, crop_preview,
    describe_context_crop_dry_run, imagenet_padding_bgr, load_context_crop_config,
    raw_distortion_ratio, resize_interpolation, square_crop_roi,
)
from drowsiness_detection.preprocessing_v2.detectors import BoundingBox


def test_dry_run_reuses_only_yunet_success_frames() -> None:
    """STEP 2-B의 160개 중 YuNet 성공 157개와 저장 bbox를 재사용한다."""

    config = load_context_crop_config(Path("configs/context_crop_audit.yaml"), PROJECT_ROOT)
    plan = describe_context_crop_dry_run(config)
    assert plan["source_frame_count"] == 160
    assert plan["yunet_success_frame_count"] == 157
    assert plan["split_counts"] == {"train": 80, "val": 77}
    assert plan["test_split_used"] is False
    assert plan["detector_rerun"] is False
    assert plan["landmark_rerun"] is False
    assert plan["candidates"] == ["RAW_RESIZE", "SQUARE_0", "SQUARE_M10", "SQUARE_M20"]


def test_square_tall_wide_and_center_preservation() -> None:
    """긴 축은 줄이지 않고 짧은 축만 중심 기준으로 확장한다."""

    tall = BoundingBox(100, 50, 300, 350)
    wide = BoundingBox(50, 100, 350, 300)
    assert square_crop_roi(tall, 0) == BoundingBox(50, 50, 350, 350)
    assert square_crop_roi(wide, 0) == BoundingBox(50, 50, 350, 350)
    for bbox in (tall, wide):
        roi = square_crop_roi(bbox, 0.20)
        assert roi.width == roi.height
        assert roi.x1 + roi.x2 == bbox.x1 + bbox.x2
        assert roi.y1 + roi.y2 == bbox.y1 + bbox.y2
        assert roi.x1 <= bbox.x1 and roi.y1 <= bbox.y1
        assert roi.x2 >= bbox.x2 and roi.y2 >= bbox.y2


def test_margin_definitions() -> None:
    """M10/M20은 width·height 각각의 양쪽 margin을 적용한 뒤 square로 만든다."""

    bbox = BoundingBox(100, 50, 300, 350)
    assert square_crop_roi(bbox, 0.10) == BoundingBox(20, 20, 380, 380)
    assert square_crop_roi(bbox, 0.20) == BoundingBox(-10, -10, 410, 410)


def test_frame_edge_padding_and_fraction() -> None:
    """frame 밖 정사각형 ROI를 줄이지 않고 누락 면적을 padding으로 계산한다."""

    bbox = BoundingBox(0, 0, 100, 100)
    roi = square_crop_roi(bbox, 0.10)
    geometry = crop_geometry(bbox, roi, 200, 200)
    assert roi == BoundingBox(-10, -10, 110, 110)
    assert (geometry.padding_left, geometry.padding_top, geometry.padding_right, geometry.padding_bottom) == (10, 10, 0, 0)
    assert geometry.padding_required is True
    assert geometry.padding_fraction == pytest.approx(1 - (110 * 110) / (120 * 120))


def test_imagenet_padding_bgr_and_rgb_preview() -> None:
    """BGR canvas 값이 RGB ImageNet mean 순서로 바뀌어 반환된다."""

    bgr = imagenet_padding_bgr()
    assert bgr == (104, 116, 124)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    geometry = crop_geometry(BoundingBox(0, 0, 10, 10), BoundingBox(-10, -10, 10, 10), 20, 20)
    rgb = crop_preview(frame, geometry, 20, bgr)
    assert rgb.shape == (20, 20, 3)
    assert tuple(rgb[0, 0]) == (124, 116, 104)
    assert tuple(rgb[-1, -1]) == (0, 0, 0)


def test_resize_distortion_and_face_area() -> None:
    """축소·확대 보간법, RAW 왜곡 배수와 얼굴 점유율을 확인한다."""

    assert resize_interpolation(300, 300, 224) == cv2.INTER_AREA
    assert resize_interpolation(200, 300, 224) == cv2.INTER_LINEAR
    bbox = BoundingBox(100, 50, 300, 350)
    assert raw_distortion_ratio(bbox) == 1.5
    geometry = crop_geometry(bbox, square_crop_roi(bbox, 0), 640, 480)
    assert geometry.face_area_fraction == pytest.approx(2 / 3)
    assert geometry.padding_fraction == 0


def test_existing_visual_selection_and_record_columns() -> None:
    """기존 검토 팩에서만 균형 잡힌 42개를 선택하고 필수 CSV 열을 만든다."""

    config = load_context_crop_config(Path("configs/context_crop_audit.yaml"), PROJECT_ROOT)
    source = config.path("source_results_dir") / "paired_primary_results.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["yunet_success"] == "True"]
    records = [_record(row, 720, 1280) for row in rows]
    selected = _select_visual_rows(config, records)
    assert len(selected) == 42
    assert len({item["record"]["sample_id"] for item in selected}) == 42
    assert set(BASE_COLUMNS) <= set(records[0])
    assert set(RESULT_COLUMNS) <= set(records[0])
    assert all(item["reason"] for item in selected)


def test_csv_report_and_visualization_smoke() -> None:
    """합성 영상으로 5-panel, CSV와 report writer를 확인한다."""

    bbox = BoundingBox(30, 20, 90, 100)
    source = {
        "sample_id": "sample", "video_id": "video", "split": "train", "label": "drowsy",
        "timestamp_sec": "1.0", "source_frame_index": "30",
        **{f"yunet_bbox_{key}": str(value) for key, value in zip(("x1", "y1", "x2", "y2"), bbox.as_tuple())},
    }
    record = _record(source, 120, 120)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    image = _visual_image(frame, bbox, record, 320, 224, imagenet_padding_bgr())
    summary = _summary([record], 1, 1)
    assert image.shape == (360, 1600, 3)
    assert DECISION in _report(summary)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
        output = Path(temporary_directory) / "sample.csv"
        _write_csv(output, RESULT_COLUMNS, [record])
        with output.open("r", encoding="utf-8-sig", newline="") as handle:
            assert list(csv.DictReader(handle))[0]["sample_id"] == "sample"
