"""영상 bundle과 집계 데이터의 구조적 불변식을 검증한다."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _yes(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_bundle(bundle: Path, strict_crop_decode: bool = False) -> list[str]:
    """완성 marker와 100×68 구조, 결측 전파, crop 파일을 검사한다."""

    errors: list[str] = []
    try:
        marker = json.loads((bundle / "COMPLETE.json").read_text(encoding="utf-8"))
        with (bundle / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with np.load(bundle / "landmarks.npz") as data:
            points = data["landmarks"]
            valid = data["landmark_valid"]
            indices = data["canonical_index"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return [f"BUNDLE_READ_FAILED:{exc}"]
    if marker.get("policy_hash") is None or marker.get("run_config_hash") is None:
        errors.append("MARKER_HASH_MISSING")
    if len(rows) != 100 or marker.get("canonical_row_count") != 100:
        errors.append("CANONICAL_ROW_COUNT")
    if points.shape != (100, 68, 2) or points.dtype != np.float32 or valid.shape != (100,) or indices.shape != (100,):
        errors.append("LANDMARK_NPZ_SHAPE")
        return errors
    if [int(row["canonical_index"]) for row in rows] != list(range(100)) or indices.tolist() != list(range(100)):
        errors.append("CANONICAL_INDEX")
    if any(row["split"] not in {"train", "val"} for row in rows):
        errors.append("TEST_OR_UNKNOWN_SPLIT")
    if len({row["video_id"] for row in rows}) != 1:
        errors.append("VIDEO_ID_INCONSISTENT")
    if sum(_yes(row["context_selected"]) for row in rows) != 32:
        errors.append("CONTEXT_SELECTION_COUNT")
    slots = [int(row["context_slot"]) for row in rows if _yes(row["context_selected"])]
    if sorted(slots) != list(range(32)):
        errors.append("CONTEXT_SLOT_INDEX")
    timestamps = [float(row["target_timestamp_sec"]) for row in rows]
    source_indices = [int(row["source_frame_index"]) for row in rows]
    if timestamps != sorted(timestamps) or source_indices != sorted(source_indices):
        errors.append("SAMPLING_ORDER")
    for index, row in enumerate(rows):
        face, landmark, pose, crop = (_yes(row[key]) for key in ("yunet_success", "landmark_success", "pose_success", "context_crop_available"))
        if not face and landmark or not face and crop or crop and not _yes(row["context_selected"]):
            errors.append(f"DEPENDENCY:{index}")
        if not landmark and (not np.isnan(points[index]).all() or bool(valid[index]) or not _missing(row["ear"]) or not _missing(row["mar"])):
            errors.append(f"LANDMARK_MISSING:{index}")
        if landmark and (not bool(valid[index]) or not np.isfinite(points[index]).all()):
            errors.append(f"LANDMARK_VALID:{index}")
        if not pose and any(not _missing(row[key]) for key in ("pitch_raw", "pitch_centered_candidate", "yaw", "roll")):
            errors.append(f"POSE_MISSING:{index}")
        if crop:
            path = bundle / row["context_crop_relpath"]
            if not path.is_file():
                errors.append(f"CONTEXT_FILE_MISSING:{index}")
            elif strict_crop_decode:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None or image.shape != (224, 224, 3):
                    errors.append(f"CONTEXT_IMAGE_INVALID:{index}")
    return errors


def validate_global_rows(frames: Sequence[Mapping[str, Any]], videos: Sequence[Mapping[str, Any]]) -> list[str]:
    """집계 시 test·중복 및 완료 영상의 100 row를 재확인한다."""

    errors: list[str] = []
    ids = [(row["split"], row["video_id"]) for row in videos]
    if len(ids) != len(set(ids)) or len({video_id for _, video_id in ids}) != len(ids):
        errors.append("DUPLICATE_VIDEO_ID")
    if any(row["split"] not in {"train", "val"} for row in [*frames, *videos]):
        errors.append("TEST_OR_UNKNOWN_SPLIT")
    for split, video_id in ids:
        subset = [row for row in frames if row["split"] == split and row["video_id"] == video_id]
        if len(subset) != 100:
            errors.append(f"FRAME_COUNT:{video_id}")
    return errors
