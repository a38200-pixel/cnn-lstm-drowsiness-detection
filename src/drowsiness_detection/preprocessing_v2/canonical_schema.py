"""Canonical CSV의 고정 필드와 결측 row 기본값이다."""

from __future__ import annotations

import math
from typing import Any


FRAME_COLUMNS = (
    "video_id", "split", "label", "canonical_index", "target_timestamp_sec",
    "source_frame_index", "actual_timestamp_sec", "timestamp_error_ms", "source_fps",
    "reported_source_frame_count", "frame_width", "frame_height", "source_index_duplicate",
    "source_index_in_range", "decode_ok", "decode_status", "yunet_success", "yunet_confidence",
    "yunet_detection_count", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width",
    "bbox_height", "detector_ms", "detector_status", "landmark_success", "landmark_ms",
    "landmark_status", "ear", "mar", "pose_success", "pitch_raw",
    "pitch_centered_candidate", "yaw", "roll", "pose_status", "feature_ms",
    "context_selected", "context_slot", "context_crop_available", "context_crop_relpath",
    "context_padding_fraction", "context_face_area_fraction", "context_status", "context_crop_ms",
    "context_write_ms", "frame_total_ms", "failure_flags",
)

VIDEO_COLUMNS = (
    "video_id", "split", "label", "source_path", "source_fps", "reported_source_frame_count",
    "reported_duration_sec", "expected_canonical_slots", "decoded_slots", "out_of_range_slots",
    "decode_failed_slots", "duplicate_source_index_count", "yunet_success_count",
    "yunet_failure_count", "landmark_success_count", "landmark_failure_count", "ear_valid_count",
    "mar_valid_count", "pose_valid_count", "context_selected_count", "context_crop_available_count",
    "context_crop_missing_count", "multiple_face_frame_count", "processing_time_sec",
    "bundle_status", "policy_hash", "run_config_hash",
)


def empty_frame_row(video_id: str, split: str, label: str, canonical_index: int,
                    target_timestamp_sec: float, source_frame_index: int, fps: float,
                    frame_count: int, context_slot: int | None, duplicate: bool) -> dict[str, Any]:
    """실패 slot도 완전한 schema를 갖도록 NaN과 단계별 상태를 초기화한다."""

    row: dict[str, Any] = {column: "" for column in FRAME_COLUMNS}
    row.update(video_id=video_id, split=split, label=label, canonical_index=canonical_index,
               target_timestamp_sec=target_timestamp_sec, source_frame_index=source_frame_index,
               actual_timestamp_sec=source_frame_index / fps,
               timestamp_error_ms=(source_frame_index / fps - target_timestamp_sec) * 1000.0,
               source_fps=fps, reported_source_frame_count=frame_count,
               source_index_duplicate=duplicate, source_index_in_range=source_frame_index < frame_count,
               decode_ok=False, decode_status="NOT_RUN", yunet_success=False,
               yunet_detection_count=0, detector_status="NOT_RUN", landmark_success=False,
               landmark_status="NOT_RUN", pose_success=False, pose_status="NOT_RUN",
               context_selected=context_slot is not None, context_slot=context_slot if context_slot is not None else "",
               context_crop_available=False,
               context_status="NOT_AVAILABLE" if context_slot is not None else "NOT_SELECTED",
               frame_total_ms=0.0, failure_flags="")
    for column in ("yunet_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                   "bbox_width", "bbox_height", "detector_ms", "landmark_ms", "ear", "mar",
                   "pitch_raw", "pitch_centered_candidate", "yaw", "roll", "feature_ms",
                   "context_padding_fraction", "context_face_area_fraction", "context_crop_ms", "context_write_ms"):
        row[column] = math.nan
    return row
