"""완성된 train/val canonical CSV만으로 결측·품질을 진단한다."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


POLICY_HASH = "29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721"
EXPECTED_SPLITS = {"train": 1452, "val": 311}
LABELS = {"drowsy", "not_drowsy"}
OUTPUT_FILES = (
    "quality_missing_report.txt", "quality_missing_summary.json", "per_video_quality.csv",
    "split_label_quality_summary.csv", "missing_runs.csv", "missing_run_neighbor_pose.csv",
    "missing_run_length_histogram.csv",
    "behavior_missing_count_histogram.csv", "context_missing_count_histogram.csv",
    "behavior_missing_cumulative.csv", "context_missing_cumulative.csv",
    "temporal_missing_by_canonical_slot.csv", "temporal_missing_by_context_slot.csv",
    "missing_concentration_summary.csv", "top_missing_videos.csv",
    "top_context_missing_videos.csv", "top_consecutive_missing_videos.csv",
    "numeric_quality_anomalies.csv", "bbox_quality_summary.json", "step4b_review_candidates.csv",
)
VIDEO_COLUMNS = (
    "video_id", "split", "label", "canonical_frames", "yunet_success_count",
    "yunet_missing_count", "yunet_missing_rate", "behavior_valid_count",
    "behavior_missing_count", "behavior_valid_rate", "pose_valid_count", "pose_missing_count",
    "context_selected_count", "context_available_count", "context_missing_count",
    "context_missing_rate", "padding_count", "longest_yunet_missing_run",
    "number_of_yunet_missing_runs", "longest_context_missing_run",
    "number_of_context_missing_runs", "first_missing_index", "last_missing_index",
    "missing_at_start", "missing_at_end", "policy_hash",
)
RUN_COLUMNS = (
    "video_id", "split", "label", "run_id", "start_index", "end_index", "run_length",
    "start_timestamp_sec", "end_timestamp_sec", "touches_clip_start", "touches_clip_end",
    "context_slots_inside_run", "context_missing_slots_inside_run",
)
CANDIDATE_COLUMNS = (
    "review_order", "video_id", "split", "label", "selection_reason",
    "yunet_missing_count", "yunet_missing_rate", "longest_missing_run",
    "context_missing_count", "behavior_valid_rate", "representative_missing_start",
    "representative_missing_end", "recommended_frame_indices",
)


def is_true(value: str) -> bool:
    """CSV bool 표현을 엄격하게 읽는다."""
    if value not in {"True", "False"}:
        raise ValueError(f"잘못된 boolean 값: {value!r}")
    return value == "True"


def finite(value: str) -> float | None:
    """NaN·빈칸·무한대를 결측으로 다룬다."""
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def is_infinite(value: str) -> bool:
    """결측 NaN과 구별해 무한대 오염을 찾는다."""
    return bool(value) and math.isinf(float(value))


def missing_runs(flags: Iterable[bool]) -> list[tuple[int, int]]:
    """True가 연속되는 포함 구간의 시작·끝을 반환한다."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    length = 0
    for index, missing in enumerate(flags):
        length = index + 1
        if missing and start is None:
            start = index
        elif not missing and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, length - 1))
    return runs


def distribution(values: list[float | int]) -> dict[str, float | int | None]:
    """연속값의 설명 통계를 계산하며 빈 입력도 허용한다."""
    if not values:
        return {key: None for key in ("count", "min", "mean", "std", "p01", "p05", "median", "p75", "p90", "p95", "p99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "mean": float(array.mean()),
            "std": float(array.std()), "p01": float(np.percentile(array, 1)),
            "p05": float(np.percentile(array, 5)), "median": float(np.median(array)),
            "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
            "max": float(array.max())}


def cumulative_counts(values: list[int], maximum: int) -> list[dict[str, int | float]]:
    """허용 개수를 선택하지 않고 모든 N의 보존 영상 수를 제시한다."""
    return [{"max_missing_allowed": limit,
             "videos_satisfying": sum(value <= limit for value in values),
             "fraction": sum(value <= limit for value in values) / len(values) if values else 0.0}
            for limit in range(maximum + 1)]


def concentration(values: list[int]) -> list[dict[str, int | float]]:
    """결측이 상위 영상 몇 개에 모이는지 기술한다."""
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    return [{"top_video_percent": percent, "top_video_count": math.ceil(len(ordered) * percent / 100),
             "missing_in_top_videos": sum(ordered[:math.ceil(len(ordered) * percent / 100)]),
             "fraction_of_all_missing": sum(ordered[:math.ceil(len(ordered) * percent / 100)]) / total if total else 0.0}
            for percent in (1, 5, 10)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _video_key(row: Mapping[str, str]) -> tuple[str, str]:
    return row["split"], row["video_id"]


def validate_metadata(canonical_root: Path, expected_splits: Mapping[str, int] = EXPECTED_SPLITS,
                      expected_policy: str = POLICY_HASH) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
    """test·정책·영상 수를 영상 경로 해석 전에 검사한다."""
    metadata = canonical_root / "metadata"
    videos = _read_csv(metadata / "canonical_videos.csv")
    run = json.loads((metadata / "run_metadata.json").read_text(encoding="utf-8"))
    if run.get("policy_hash") != expected_policy or run.get("test_split_processed") is not False:
        raise ValueError("STEP_4A_BLOCKED_POLICY_HASH_MISMATCH 또는 test 처리 기록")
    counts = Counter(row["split"] for row in videos)
    if set(counts) - {"train", "val"}:
        raise ValueError("STEP_4A_BLOCKED_TEST_LEAKAGE: video metadata")
    if counts != dict(expected_splits):
        raise ValueError(f"완료 video 수 불일치: {counts}")
    keys = [_video_key(row) for row in videos]
    if len(set(keys)) != len(keys) or len({row["video_id"] for row in videos}) != len(videos):
        raise ValueError("video_id 중복")
    if any(row["label"] not in LABELS or row["policy_hash"] != expected_policy or row["bundle_status"] != "COMPLETE" for row in videos):
        raise ValueError("label·policy hash·bundle 상태 불일치")
    return dict(zip(keys, videos)), run


def scan_frames(canonical_root: Path, video_metadata: Mapping[tuple[str, str], dict[str, str]],
                expected_policy: str = POLICY_HASH) -> Iterator[tuple[dict[str, str], list[dict[str, str]]]]:
    """전역 frame CSV만 읽어 영상별 100행과 Context 32행을 확인한다."""
    seen: set[tuple[str, str]] = set()
    current_key: tuple[str, str] | None = None
    current: list[dict[str, str]] = []

    def finish() -> tuple[dict[str, str], list[dict[str, str]]] | None:
        if current_key is None:
            return None
        video = video_metadata.get(current_key)
        if video is None or len(current) != 100:
            raise ValueError(f"영상별 100행 또는 video metadata 불일치: {current_key}")
        indices = [int(row["canonical_index"]) for row in current]
        slots = [int(row["context_slot"]) for row in current if is_true(row["context_selected"])]
        if indices != list(range(100)) or slots != list(range(32)):
            raise ValueError(f"canonical index 또는 Context 32 slot 불일치: {current_key}")
        if any(row["label"] != video["label"] for row in current):
            raise ValueError(f"video label 불일치: {current_key}")
        if any(float(current[index]["target_timestamp_sec"]) > float(current[index + 1]["target_timestamp_sec"]) or
               int(current[index]["source_frame_index"]) > int(current[index + 1]["source_frame_index"])
               for index in range(99)):
            raise ValueError(f"timestamp/source index 순서 불일치: {current_key}")
        return video, current.copy()

    path = canonical_root / "metadata/canonical_frames.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "val"}:
                raise ValueError("STEP_4A_BLOCKED_TEST_LEAKAGE: frame metadata")
            key = _video_key(row)
            if key != current_key:
                finished = finish()
                if finished is not None:
                    yield finished
                if key in seen:
                    raise ValueError(f"전역 frame CSV에서 영상 행이 흩어져 있거나 중복됨: {key}")
                seen.add(key)
                current_key, current = key, []
            if key not in video_metadata:
                raise ValueError(f"전역 frame CSV에 알 수 없는 영상: {key}")
            current.append(row)
    finished = finish()
    if finished is not None:
        yield finished
    if len(seen) != len(video_metadata):
        raise ValueError("전역 영상/frame 수 불일치")
    if any(video["policy_hash"] != expected_policy for video in video_metadata.values()):
        raise ValueError("STEP_4A_BLOCKED_POLICY_HASH_MISMATCH")


def _analyze_video(video: dict[str, str], rows: list[dict[str, str]]) -> dict[str, Any]:
    """한 영상의 프레임·Context 결측과 수치 진단을 계산한다."""
    identity = {key: video[key] for key in ("video_id", "split", "label")}
    face = [is_true(row["yunet_success"]) for row in rows]
    landmark = [is_true(row["landmark_success"]) for row in rows]
    pose = [is_true(row["pose_success"]) for row in rows]
    selected = [is_true(row["context_selected"]) for row in rows]
    available = [is_true(row["context_crop_available"]) for row in rows]
    selected_indices = [index for index, chosen in enumerate(selected) if chosen]
    context_missing = [selected[index] and not available[index] for index in range(100)]
    context_flags = [context_missing[index] for index in selected_indices]
    ear = [finite(row["ear"]) for row in rows]
    mar = [finite(row["mar"]) for row in rows]
    pose_values = {name: [finite(row[name]) for row in rows]
                   for name in ("pitch_raw", "pitch_centered_candidate", "yaw", "roll")}
    behavior = [face[index] and landmark[index] and ear[index] is not None and mar[index] is not None
                for index in range(100)]
    valid_pose = [pose[index] and all(pose_values[name][index] is not None for name in pose_values)
                  for index in range(100)]
    miss_flags = [not success for success in face]
    runs = missing_runs(miss_flags)
    context_runs = missing_runs(context_flags)
    miss_indices = [index for index, missing in enumerate(miss_flags) if missing]
    output = {**identity, "canonical_frames": 100, "yunet_success_count": sum(face),
              "yunet_missing_count": sum(miss_flags), "yunet_missing_rate": sum(miss_flags) / 100,
              "behavior_valid_count": sum(behavior), "behavior_missing_count": 100 - sum(behavior),
              "behavior_valid_rate": sum(behavior) / 100, "pose_valid_count": sum(valid_pose),
              "pose_missing_count": 100 - sum(valid_pose), "context_selected_count": sum(selected),
              "context_available_count": sum(available), "context_missing_count": sum(context_flags),
              "context_missing_rate": sum(context_flags) / 32,
              "padding_count": sum(available[index] and finite(rows[index]["context_padding_fraction"]) is not None and
                                   float(rows[index]["context_padding_fraction"]) > 0 for index in range(100)),
              "longest_yunet_missing_run": max((end - start + 1 for start, end in runs), default=0),
              "number_of_yunet_missing_runs": len(runs),
              "longest_context_missing_run": max((end - start + 1 for start, end in context_runs), default=0),
              "number_of_context_missing_runs": len(context_runs),
              "first_missing_index": miss_indices[0] if miss_indices else "",
              "last_missing_index": miss_indices[-1] if miss_indices else "",
              "missing_at_start": bool(miss_flags[0]), "missing_at_end": bool(miss_flags[-1]),
              "policy_hash": video["policy_hash"]}
    expected = {
        "yunet_success_count": sum(face), "yunet_failure_count": sum(miss_flags),
        "landmark_success_count": sum(landmark), "landmark_failure_count": sum(face) - sum(landmark),
        "ear_valid_count": sum(value is not None for value in ear),
        "mar_valid_count": sum(value is not None for value in mar),
        "pose_valid_count": sum(pose), "context_selected_count": sum(selected),
        "context_crop_available_count": sum(available), "context_crop_missing_count": sum(context_flags),
        "decoded_slots": sum(is_true(row["decode_ok"]) for row in rows),
        "duplicate_source_index_count": sum(is_true(row["source_index_duplicate"]) for row in rows),
    }
    if any(int(video[key]) != value for key, value in expected.items()):
        raise ValueError(f"전역 frame과 video summary count 불일치: {identity}")
    if sum(face) + sum(miss_flags) != 100 or sum(selected) != 32 or sum(available) + sum(context_flags) != 32:
        raise ValueError(f"영상별 분모 불일치: {identity}")

    run_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    for run_id, (start, end) in enumerate(runs, 1):
        inside = [index for index in selected_indices if start <= index <= end]
        run_rows.append({**identity, "run_id": run_id, "start_index": start, "end_index": end,
                         "run_length": end - start + 1,
                         "start_timestamp_sec": rows[start]["target_timestamp_sec"],
                         "end_timestamp_sec": rows[end]["target_timestamp_sec"],
                         "touches_clip_start": start == 0, "touches_clip_end": end == 99,
                         "context_slots_inside_run": len(inside),
                         "context_missing_slots_inside_run": sum(context_missing[index] for index in inside)})
        before = next((index for index in range(start - 1, max(-1, start - 6), -1) if valid_pose[index]), None)
        after = next((index for index in range(end + 1, min(100, end + 6)) if valid_pose[index]), None)
        neighbor_rows.append({**identity, "run_id": run_id, "start_index": start, "end_index": end,
                              "nearest_valid_before_index": before if before is not None else "",
                              "nearest_valid_before_yaw": pose_values["yaw"][before] if before is not None else "",
                              "nearest_valid_after_index": after if after is not None else "",
                              "nearest_valid_after_yaw": pose_values["yaw"][after] if after is not None else "",
                              "hint_scope": "neighboring pose only; not pose of missing frames"})

    anomalies: list[dict[str, Any]] = []
    features: dict[str, list[float]] = defaultdict(list)
    bbox: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        issues = []
        if landmark[index] and (ear[index] is None or mar[index] is None):
            issues.append("LANDMARK_SUCCESS_NONFINITE_EAR_MAR")
        if not landmark[index] and (ear[index] is not None or mar[index] is not None):
            issues.append("LANDMARK_FAILURE_FINITE_EAR_MAR")
        if pose[index] and not valid_pose[index]:
            issues.append("POSE_SUCCESS_NONFINITE_VALUES")
        if not pose[index] and any(pose_values[name][index] is not None for name in pose_values):
            issues.append("POSE_FAILURE_FINITE_VALUES")
        if landmark[index] and not face[index] or pose[index] and not landmark[index]:
            issues.append("FEATURE_DEPENDENCY")
        if available[index] and (not selected[index] or not face[index]):
            issues.append("CROP_DEPENDENCY")
        if not is_true(row["decode_ok"]) or not is_true(row["source_index_in_range"]):
            issues.append("DECODE_OR_SOURCE_INDEX")
        if any(is_infinite(row[name]) for name in
               ("ear", "mar", "pitch_raw", "pitch_centered_candidate", "yaw", "roll")):
            issues.append("NUMERIC_INFINITY")
        if any(value is not None and abs(value) > 1_000_000 for value in
               (ear[index], mar[index], *(pose_values[name][index] for name in pose_values))):
            issues.append("NUMERIC_OVERFLOW_DIAGNOSTIC")
        if face[index]:
            width, height = finite(row["bbox_width"]), finite(row["bbox_height"])
            frame_width, frame_height = finite(row["frame_width"]), finite(row["frame_height"])
            confidence = finite(row["yunet_confidence"])
            if (width is None or height is None or frame_width is None or frame_height is None or
                    confidence is None or min(width, height, frame_width, frame_height) <= 0):
                issues.append("INVALID_BBOX_OR_CONFIDENCE")
            else:
                bbox["bbox_width"].append(width)
                bbox["bbox_height"].append(height)
                bbox["bbox_area_fraction"].append(width * height / (frame_width * frame_height))
                bbox["yunet_confidence"].append(confidence)
        if behavior[index]:
            features["ear"].append(ear[index])
            features["mar"].append(mar[index])
        if valid_pose[index]:
            for name in pose_values:
                features[name].append(pose_values[name][index])
        for issue in issues:
            anomalies.append({**identity, "canonical_index": index, "issue": issue})
    return {"quality": output, "runs": run_rows, "neighbors": neighbor_rows,
            "anomalies": anomalies, "features": features, "bbox": bbox,
            "face": face, "behavior": behavior, "pose": valid_pose,
            "context_selected": selected, "context_missing": context_missing,
            "multiple_face_count": int(video["multiple_face_frame_count"]),
            "missing_indices": miss_indices}


def _group_row(name: str, group_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """split·label·교차층의 동일 분모 지표를 만든다."""
    videos = len(rows)
    frames = videos * 100
    context = videos * 32
    missing = sum(int(row["yunet_missing_count"]) for row in rows)
    behavior = sum(int(row["behavior_valid_count"]) for row in rows)
    context_missing = sum(int(row["context_missing_count"]) for row in rows)
    return {"group_type": group_type, "group_name": name, "videos": videos, "canonical_frames": frames,
            "yunet_success": frames - missing, "yunet_missing": missing,
            "yunet_missing_rate": missing / frames if frames else 0.0,
            "behavior_valid": behavior, "behavior_valid_rate": behavior / frames if frames else 0.0,
            "context_selected": context, "context_available": context - context_missing,
            "context_missing": context_missing,
            "context_missing_rate": context_missing / context if context else 0.0,
            "zero_yunet_missing_videos": sum(row["yunet_missing_count"] == 0 for row in rows),
            "zero_context_missing_videos": sum(row["context_missing_count"] == 0 for row in rows)}


def choose_candidates(videos: list[dict[str, Any]], runs: list[dict[str, Any]],
                      target: int = 36) -> list[dict[str, Any]]:
    """중복 없이 4개 split×label층과 5개 검토 유형을 결정적으로 채운다."""
    if target < 0:
        raise ValueError("후보 수는 음수일 수 없습니다")
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    run_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        run_by_video[run["video_id"]].append(run)
    specifications = (
        ("highest_missing_count", 8,
         lambda row: row["yunet_missing_count"] > 0,
         lambda row: (-row["yunet_missing_count"], -row["longest_yunet_missing_run"], row["video_id"])),
        ("longest_consecutive_run", 8,
         lambda row: row["longest_yunet_missing_run"] > 0,
         lambda row: (-row["longest_yunet_missing_run"], -row["yunet_missing_count"], row["video_id"])),
        ("highest_context_missing", 8,
         lambda row: row["context_missing_count"] > 0,
         lambda row: (-row["context_missing_count"], -row["yunet_missing_count"], row["video_id"])),
        ("isolated_single_miss", 6,
         lambda row: row["yunet_missing_count"] > 0 and row["longest_yunet_missing_run"] == 1,
         lambda row: (-row["yunet_missing_count"], row["video_id"])),
        ("zero_missing_reference", 6,
         lambda row: row["yunet_missing_count"] == 0 and row["context_missing_count"] == 0,
         lambda row: (row["video_id"],)),
    )

    def append(row: dict[str, Any], reason: str) -> None:
        video_id = str(row["video_id"])
        if video_id in used:
            return
        used.add(video_id)
        video_runs = sorted(run_by_video[video_id], key=lambda run: (-int(run["run_length"]), int(run["start_index"])))
        representative = video_runs[0] if video_runs else None
        start, end = ((int(representative["start_index"]), int(representative["end_index"]))
                      if representative else (None, None))
        suggested = (sorted({max(0, start - 1), start, (start + end) // 2, end, min(99, end + 1)})
                     if start is not None else [0, 50, 99])
        chosen.append({"review_order": len(chosen), "video_id": video_id, "split": row["split"],
                       "label": row["label"], "selection_reason": reason,
                       "yunet_missing_count": row["yunet_missing_count"],
                       "yunet_missing_rate": row["yunet_missing_rate"],
                       "longest_missing_run": row["longest_yunet_missing_run"],
                       "context_missing_count": row["context_missing_count"],
                       "behavior_valid_rate": row["behavior_valid_rate"],
                       "representative_missing_start": start if start is not None else "",
                       "representative_missing_end": end if end is not None else "",
                       "recommended_frame_indices": ";".join(map(str, suggested))})

    strata = [(split, label) for split in ("train", "val") for label in ("drowsy", "not_drowsy")]
    for reason, quota, eligible, rank in specifications:
        queues = {stratum: sorted((row for row in videos if (row["split"], row["label"]) == stratum and eligible(row)), key=rank)
                  for stratum in strata}
        added = 0
        while added < quota and any(queues.values()) and len(chosen) < target:
            progress = False
            for stratum in strata:
                queue = queues[stratum]
                while queue and queue[0]["video_id"] in used:
                    queue.pop(0)
                if queue and added < quota and len(chosen) < target:
                    append(queue.pop(0), reason)
                    added += 1
                    progress = True
            if not progress:
                break
    if len(chosen) < min(target, len(videos)):
        counts = Counter((row["split"], row["label"]) for row in chosen)
        remaining = sorted((row for row in videos if row["video_id"] not in used),
                           key=lambda row: (counts[(row["split"], row["label"])], -row["yunet_missing_count"], row["video_id"]))
        for row in remaining[:target - len(chosen)]:
            append(row, "balanced_fill")
    return chosen


def _verify_step3c(summary: Mapping[str, Any], overall: Mapping[str, Any],
                   per_video: list[dict[str, Any]], multiple_faces: int, padding: int) -> None:
    """STEP 3-C의 확정 결과와 새 계산값이 같아야 분석을 발행한다."""
    expected = {
        "video total": (summary["videos"]["total"], len(per_video)),
        "frame total": (summary["canonical_rows"]["total"], overall["canonical_frames"]),
        "YuNet success": (summary["yunet"]["success"], overall["yunet_success"]),
        "YuNet miss": (summary["yunet"]["miss"], overall["yunet_missing"]),
        "Dlib success": (summary["dlib68"]["success"], sum(int(row["_landmark_count"]) for row in per_video)),
        "EAR valid": (summary["behavior_features"]["ear_valid"], sum(int(row["_ear_count"]) for row in per_video)),
        "MAR valid": (summary["behavior_features"]["mar_valid"], sum(int(row["_mar_count"]) for row in per_video)),
        "Pose valid": (summary["behavior_features"]["pose_valid"], sum(int(row["_pose_count"]) for row in per_video)),
        "Context selected": (summary["context"]["selected"], overall["context_selected"]),
        "Context available": (summary["context"]["available"], overall["context_available"]),
        "Context missing": (summary["context"]["missing"], overall["context_missing"]),
        "padding": (summary["padding"]["crops_requiring_padding"], padding),
        "multiple face": (summary["multiple_face_frames"], multiple_faces),
    }
    mismatch = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if mismatch:
        raise ValueError(f"STEP 3-C 요약과 새 분석값 불일치: {mismatch}")


def analyze(canonical_root: Path, full_summary_path: Path | None = None,
            expected_splits: Mapping[str, int] = EXPECTED_SPLITS,
            expected_policy: str = POLICY_HASH) -> dict[str, Any]:
    """읽기 전용으로 전체 분포·run·수치 품질·후보를 계산한다."""
    metadata, run_metadata = validate_metadata(canonical_root, expected_splits, expected_policy)
    per_video: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    neighbors: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    feature_values: dict[str, list[float]] = defaultdict(list)
    bbox_values: dict[str, list[float]] = defaultdict(list)
    temporal = [{"canonical_index": index, "total_frames": 0, "yunet_missing_count": 0} for index in range(100)]
    context_temporal = [{"context_slot": index, "selected_count": 0, "missing_count": 0} for index in range(32)]
    periods = {name: {"total_frames": 0, "yunet_missing_count": 0} for name in ("first_10", "middle_80", "last_10")}
    multiple_faces = 0
    padding = 0
    decoded = 0
    for video, rows in scan_frames(canonical_root, metadata, expected_policy):
        result = _analyze_video(video, rows)
        quality = result["quality"]
        quality["_landmark_count"] = int(video["landmark_success_count"])
        quality["_ear_count"] = int(video["ear_valid_count"])
        quality["_mar_count"] = int(video["mar_valid_count"])
        quality["_pose_count"] = int(video["pose_valid_count"])
        per_video.append(quality)
        runs.extend(result["runs"])
        neighbors.extend(result["neighbors"])
        anomalies.extend(result["anomalies"])
        multiple_faces += result["multiple_face_count"]
        padding += quality["padding_count"]
        decoded += int(video["decoded_slots"])
        for key, values in result["features"].items():
            feature_values[key].extend(values)
        for key, values in result["bbox"].items():
            bbox_values[key].extend(values)
        for index in range(100):
            temporal[index]["total_frames"] += 1
            temporal[index]["yunet_missing_count"] += not result["face"][index]
            period = "first_10" if index < 10 else "last_10" if index >= 90 else "middle_80"
            periods[period]["total_frames"] += 1
            periods[period]["yunet_missing_count"] += not result["face"][index]
            if result["context_selected"][index]:
                slot = int(rows[index]["context_slot"])
                context_temporal[slot]["selected_count"] += 1
                context_temporal[slot]["missing_count"] += result["context_missing"][index]

    groups: list[dict[str, Any]] = [_group_row("all", "overall", per_video)]
    for split in ("train", "val"):
        groups.append(_group_row(split, "split", [row for row in per_video if row["split"] == split]))
    for label in ("drowsy", "not_drowsy"):
        groups.append(_group_row(label, "label", [row for row in per_video if row["label"] == label]))
    for split in ("train", "val"):
        for label in ("drowsy", "not_drowsy"):
            groups.append(_group_row(f"{split}:{label}", "split_label",
                                     [row for row in per_video if row["split"] == split and row["label"] == label]))
    overall = groups[0]
    if overall["canonical_frames"] != decoded or overall["context_selected"] != len(per_video) * 32:
        raise ValueError("decode 또는 Context 주요 집계 불일치")
    if full_summary_path is not None:
        full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
        if full_summary.get("policy_hash") != expected_policy:
            raise ValueError("STEP_4A_BLOCKED_POLICY_HASH_MISMATCH: STEP 3-C summary")
        _verify_step3c(full_summary, overall, per_video, multiple_faces, padding)

    for row in temporal:
        row["yunet_missing_rate"] = row["yunet_missing_count"] / row["total_frames"]
    for row in context_temporal:
        row["missing_rate"] = row["missing_count"] / row["selected_count"]
    for row in periods.values():
        row["yunet_missing_rate"] = row["yunet_missing_count"] / row["total_frames"]
    behavior_values = [int(row["behavior_missing_count"]) for row in per_video]
    context_values = [int(row["context_missing_count"]) for row in per_video]
    run_hist = Counter(int(run["run_length"]) for run in runs)
    longest = max((int(run["run_length"]) for run in runs), default=0)
    run_distribution = [{"run_length": length, "count": run_hist[length],
                         "fraction": run_hist[length] / len(runs) if runs else 0.0}
                        for length in range(1, longest + 1)]
    behavior_hist = Counter(behavior_values)
    context_hist = Counter(context_values)
    histograms = {
        "behavior": [{"missing_count": count, "video_count": behavior_hist[count],
                      "video_fraction": behavior_hist[count] / len(per_video)} for count in range(101)],
        "context": [{"context_missing_count": count, "video_count": context_hist[count],
                     "video_fraction": context_hist[count] / len(per_video)} for count in range(33)],
    }
    quality_stats = {
        "yunet_missing_count_per_video": distribution([row["yunet_missing_count"] for row in per_video]),
        "behavior_valid_rate_per_video": distribution([row["behavior_valid_rate"] for row in per_video]),
        "context_missing_count_per_video": distribution(context_values),
        "longest_yunet_missing_run_per_video": distribution([row["longest_yunet_missing_run"] for row in per_video]),
    }
    extremes = {
        "zero_yunet_missing_videos": sum(row["yunet_missing_count"] == 0 for row in per_video),
        "zero_context_missing_videos": sum(row["context_missing_count"] == 0 for row in per_video),
        "behavior_100_of_100_valid_videos": sum(row["behavior_valid_count"] == 100 for row in per_video),
        "context_32_of_32_available_videos": sum(row["context_available_count"] == 32 for row in per_video),
        "yunet_success_zero_videos": sum(row["yunet_success_count"] == 0 for row in per_video),
        "yunet_success_below_50_videos": sum(row["yunet_success_count"] < 50 for row in per_video),
        "context_available_zero_videos": sum(row["context_available_count"] == 0 for row in per_video),
        "context_available_below_16_videos": sum(row["context_available_count"] < 16 for row in per_video),
    }
    ranked_missing = sorted(per_video, key=lambda row: (-row["yunet_missing_count"], -row["longest_yunet_missing_run"], row["video_id"]))
    ranked_context = sorted(per_video, key=lambda row: (-row["context_missing_count"], -row["yunet_missing_count"], row["video_id"]))
    ranked_runs = sorted(per_video, key=lambda row: (-row["longest_yunet_missing_run"], -row["yunet_missing_count"], row["video_id"]))
    candidates = choose_candidates(per_video, runs)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if not anomalies else "NUMERIC_REVIEW_REQUIRED",
        "decision": "STEP 4-A AUTOMATIC QUALITY & MISSING AUDIT COMPLETE" if not anomalies else "STEP_4A_REVIEW_REQUIRED",
        "source_root": str(canonical_root.resolve()), "policy_hash": expected_policy,
        "run_config_hash": run_metadata["run_config_hash"],
        "test_rows": 0, "raw_video_opened": False, "detector_inference_performed": False,
        "overall": overall, "by_split": {row["group_name"]: row for row in groups if row["group_type"] == "split"},
        "by_label": {row["group_name"]: row for row in groups if row["group_type"] == "label"},
        "by_split_label": {row["group_name"]: row for row in groups if row["group_type"] == "split_label"},
        "feature_availability": {
            "dlib_success": sum(int(row["_landmark_count"]) for row in per_video),
            "dlib_failure_after_yunet": overall["yunet_success"] - sum(int(row["_landmark_count"]) for row in per_video),
            "ear_valid": sum(int(row["_ear_count"]) for row in per_video),
            "mar_valid": sum(int(row["_mar_count"]) for row in per_video),
            "pose_valid": sum(int(row["_pose_count"]) for row in per_video),
            "behavior_valid": overall["behavior_valid"],
            "behavior_valid_rate": overall["behavior_valid_rate"],
        },
        "padding_count": padding, "multiple_face_frame_count": multiple_faces,
        "val_minus_train_yunet_missing_rate": groups[2]["yunet_missing_rate"] - groups[1]["yunet_missing_rate"],
        "val_minus_train_context_missing_rate": groups[2]["context_missing_rate"] - groups[1]["context_missing_rate"],
        "video_distributions": quality_stats, "extreme_video_counts": extremes,
        "missing_runs": {"count": len(runs), "single_frame_count": run_hist[1],
                         "multi_frame_count": len(runs) - run_hist[1], "longest": longest,
                         "length_histogram": run_distribution},
        "temporal_periods": periods, "missing_concentration": concentration([row["yunet_missing_count"] for row in per_video]),
        "feature_numeric_distribution": {key: distribution(feature_values[key]) for key in ("ear", "mar", "pitch_raw", "pitch_centered_candidate", "yaw", "roll")},
        "bbox_quality": {key: distribution(bbox_values[key]) for key in ("bbox_width", "bbox_height", "bbox_area_fraction", "yunet_confidence")},
        "numeric_anomaly_count": len(anomalies), "step4b_candidate_count": len(candidates),
        "step4b_selection_categories": dict(Counter(row["selection_reason"] for row in candidates)),
        "pilot_visual_pose_observation": "27 pilot missing frames: user observed yaw/profile association; not generalized to full data",
        "full_pose_missing_inference": "NOT_AVAILABLE_FOR_UNDETECTED_FRAMES",
        "missing_policy_selected": False, "step4b_visual_review_started": False, "step4c_started": False,
    }
    return {"summary": summary, "per_video": per_video, "groups": groups, "runs": runs,
            "neighbors": neighbors, "anomalies": anomalies, "temporal": temporal,
            "context_temporal": context_temporal, "histograms": histograms,
            "behavior_cumulative": cumulative_counts(behavior_values, 100),
            "context_cumulative": cumulative_counts(context_values, 32),
            "concentration": summary["missing_concentration"], "top_missing": ranked_missing[:50],
            "top_context": ranked_context[:50], "top_runs": ranked_runs[:50],
            "candidates": candidates}


def _percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def _report(result: Mapping[str, Any]) -> str:
    """자동 관찰과 정책 결정을 분리한 한국어 보고서를 작성한다."""
    summary = result["summary"]
    overall = summary["overall"]
    feature = summary["feature_availability"]
    train, val = summary["by_split"]["train"], summary["by_split"]["val"]
    drowsy, not_drowsy = summary["by_label"]["drowsy"], summary["by_label"]["not_drowsy"]
    runs = summary["missing_runs"]
    extremes = summary["extreme_video_counts"]
    top = result["top_missing"][0] if result["top_missing"] else None
    context_cumulative = {row["max_missing_allowed"]: row for row in result["context_cumulative"]}
    sections = [
        "STEP 4-A Full Train/Val Automatic Quality & Missing Audit",
        "",
        "[Scope]",
        f"입력: {summary['source_root']}/metadata/canonical_frames.csv, canonical_videos.csv, run_metadata.json",
        "저장된 train/val metadata만 사용했다. 원본 영상·test split은 읽거나 처리하지 않았다.",
        f"Policy hash: {summary['policy_hash']}",
        "",
        "[Dataset Integrity]",
        f"영상 {overall['videos']}, canonical frame {overall['canonical_frames']}, 영상당 100행·Context 32 slot, test row 0. STEP 3-C 요약과 재계산값 일치.",
        "이 audit는 기존 전체 JPEG strict decode를 반복하지 않는다. STEP 3-C의 무결성 결과를 근거로 metadata만 재검사했다.",
        "",
        "[Overall Missing]",
        f"YuNet 성공 {overall['yunet_success']}, 미검출 {overall['yunet_missing']} / {overall['canonical_frames']} ({_percent(overall['yunet_missing_rate'])}).",
        f"Context 선택 {overall['context_selected']}, 가용 {overall['context_available']}, 결측 {overall['context_missing']} ({_percent(overall['context_missing_rate'])}).",
        f"Dlib 성공 {feature['dlib_success']} / YuNet 성공 {overall['yunet_success']}; 검출 후 Dlib 실패 {feature['dlib_failure_after_yunet']}.",
        f"EAR·MAR·Pose 유효 {feature['ear_valid']}·{feature['mar_valid']}·{feature['pose_valid']}; Behavior 유효 {feature['behavior_valid']} ({_percent(feature['behavior_valid_rate'])}).",
        f"Padding 필요 {summary['padding_count']}, multiple-face frame {summary['multiple_face_frame_count']}.",
        "",
        "[Train vs Val]",
        f"Train {train['videos']}개: YuNet 결측 {train['yunet_missing']}/{train['canonical_frames']} ({_percent(train['yunet_missing_rate'])}), Context 결측 {train['context_missing']}/{train['context_selected']} ({_percent(train['context_missing_rate'])}).",
        f"Val {val['videos']}개: YuNet 결측 {val['yunet_missing']}/{val['canonical_frames']} ({_percent(val['yunet_missing_rate'])}), Context 결측 {val['context_missing']}/{val['context_selected']} ({_percent(val['context_missing_rate'])}).",
        f"Val - train 결측률: YuNet {summary['val_minus_train_yunet_missing_rate']:+.6f}, Context {summary['val_minus_train_context_missing_rate']:+.6f}. 기술 통계이며 난이도 우열 판단이 아니다.",
        "",
        "[Drowsy vs Not-Drowsy]",
        f"Drowsy {drowsy['videos']}개: YuNet 결측 {_percent(drowsy['yunet_missing_rate'])}, Context 결측 {_percent(drowsy['context_missing_rate'])}.",
        f"Not-drowsy {not_drowsy['videos']}개: YuNet 결측 {_percent(not_drowsy['yunet_missing_rate'])}, Context 결측 {_percent(not_drowsy['context_missing_rate'])}.",
        "Label gap은 관찰값일 뿐 pose·조명·운전자 구성의 영향을 분리하지 못하므로 인과 해석하지 않는다.",
        "",
        "[Split × Label]",
    ]
    for name, row in summary["by_split_label"].items():
        sections.append(f"{name}: {row['videos']}개, YuNet 결측 {row['yunet_missing']}, Context 결측 {row['context_missing']}, Behavior 유효률 {_percent(row['behavior_valid_rate'])}.")
    sections += [
        "",
        "[Per-Video Missing Distribution]",
        f"YuNet 결측/video: {summary['video_distributions']['yunet_missing_count_per_video']}",
        f"결측 0 video {extremes['zero_yunet_missing_videos']}; Context 결측 0 video {extremes['zero_context_missing_videos']}.",
        f"극단값 diagnostic: YuNet 성공 0 video {extremes['yunet_success_zero_videos']}, 성공 50 미만 {extremes['yunet_success_below_50_videos']}; Context 가용 0 video {extremes['context_available_zero_videos']}, 가용 16 미만 {extremes['context_available_below_16_videos']}. 자동 제외 기준이 아니다.",
        f"최다 결측 video: {top['video_id']} ({top['split']}/{top['label']}), {top['yunet_missing_count']} frame." if top else "최다 결측 video 없음.",
        "",
        "[Consecutive Missing Runs]",
        f"총 run {runs['count']}, 단일 frame {runs['single_frame_count']}, 다중 frame {runs['multi_frame_count']}, 최장 {runs['longest']} frame. 길이별 count/fraction은 missing_run_length_histogram.csv.",
        "연속 길이는 관찰값이며 위험·허용 임계값으로 사용하지 않는다.",
        "",
        "[Context Missing Distribution]",
        f"Context 결측/video: {summary['video_distributions']['context_missing_count_per_video']}",
        "Context run은 32개 선택 slot 순서에서 연속 결측을 뜻한다. 원래 100개 canonical frame의 연속성과 구별한다.",
        "누적 가용성(결측 허용 N)은 policy 선택이 아닌 영상 손실 진단이다: " +
        ", ".join(f"N={n}: {context_cumulative[n]['videos_satisfying']}개" for n in (0, 1, 2, 4, 8, 16, 32)),
        "",
        "[Temporal Slot Distribution]",
        "구간: first 10=k0..9, middle 80=k10..89, last 10=k90..99. 각 slot의 실제 수치는 temporal_missing_by_*.csv에 기록했다.",
        "; ".join(f"{name}: {value['yunet_missing_count']}/{value['total_frames']} ({_percent(value['yunet_missing_rate'])})" for name, value in summary["temporal_periods"].items()),
        "",
        "[Missing Concentration]",
        "; ".join(f"상위 {row['top_video_percent']}%({row['top_video_count']}개) 영상: 전체 결측의 {_percent(row['fraction_of_all_missing'])}" for row in summary["missing_concentration"]),
        "",
        "[Behavior Feature Availability]",
        f"Behavior 유효률 {_percent(feature['behavior_valid_rate'])}; 영상별 분포 {summary['video_distributions']['behavior_valid_rate_per_video']}",
        "",
        "[Numeric Quality]",
        f"Flag-수치 모순 및 bbox 이상 {summary['numeric_anomaly_count']}건. 유효 EAR/MAR/Pose 및 bbox의 설명 분포는 summary JSON·bbox_quality_summary.json에 기록했다. 행동 threshold는 정하지 않았다.",
        "",
        "[Pilot Observation vs Full Audit]",
        "Pilot: STEP 3-B 사용자 시각 검토에서 27건의 yaw/profile 연관성을 관찰했다.",
        "Full: 본 자동 audit는 결측 빈도·run·집중도만 측정했다. YuNet 미검출 frame의 해당 frame yaw는 없다. 이웃 유효 frame yaw는 별도 보조 힌트일 뿐 결측 frame pose가 아니다.",
        "",
        "[STEP 4-B Candidate Selection]",
        f"중복 제거한 {summary['step4b_candidate_count']}개 영상 후보: {summary['step4b_selection_categories']}.",
        "STEP 4-B용 이미지는 만들지 않았고 수동 시각 검토도 시작하지 않았다.",
        "",
        "[Limitations]",
        "Video-level split은 unseen-driver subject split을 보장하지 않는다. Label gap의 인과 원인, 결측 frame의 pose, 적절한 missing 허용 개수는 이번 단계에서 결정할 수 없다.",
        "",
        "[Decision]",
        summary["decision"],
        "MISSING POLICY: NOT SELECTED",
        "MANUAL VISUAL REVIEW: NOT STARTED",
        "STEP 4-B: WAITING",
        "STEP 4-C: NOT STARTED",
        "TEST SPLIT: SEALED",
        "RAW VIDEO REPROCESSING: NOT PERFORMED",
    ]
    return "\n".join(sections) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    """0행 테이블도 헤더를 유지한다."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dry_run(canonical_root: Path, full_summary_path: Path | None = None,
            expected_splits: Mapping[str, int] = EXPECTED_SPLITS,
            expected_policy: str = POLICY_HASH) -> dict[str, Any]:
    """metadata 구조만 읽고 결과 디렉터리를 만들지 않는다."""
    videos, run = validate_metadata(canonical_root, expected_splits, expected_policy)
    group_count = sum(1 for _ in scan_frames(canonical_root, videos, expected_policy))
    if full_summary_path is not None:
        summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
        if (summary.get("policy_hash") != expected_policy or
                summary["videos"]["total"] != group_count or
                summary["canonical_rows"]["total"] != group_count * 100):
            raise ValueError("STEP 3-C summary와 dry-run 불일치")
    return {"mode": "DRY_RUN", "canonical_root": str(canonical_root.resolve()),
            "video_count": group_count, "frame_count": group_count * 100,
            "split_counts": dict(Counter(split for split, _ in videos)),
            "context_slots_per_video": 32, "policy_hash": run["policy_hash"],
            "test_rows": 0, "raw_video_opened": False, "output_written": False,
            "planned_output_files": list(OUTPUT_FILES)}


def write_audit(result: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """기존 출력은 건드리지 않고 새 형제 임시 경로에서 검증 후 발행한다."""
    if output_dir.exists():
        raise FileExistsError(f"기존 audit 출력 덮어쓰기 금지: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = output_dir.parent / f".{output_dir.name}_{uuid.uuid4().hex}.tmp"
    temp.mkdir()
    try:
        summary = result["summary"]
        (temp / "quality_missing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp / "quality_missing_report.txt").write_text(_report(result), encoding="utf-8")
        (temp / "bbox_quality_summary.json").write_text(json.dumps(summary["bbox_quality"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_csv(temp / "per_video_quality.csv", result["per_video"], VIDEO_COLUMNS)
        _write_csv(temp / "split_label_quality_summary.csv", result["groups"], result["groups"][0].keys())
        _write_csv(temp / "missing_runs.csv", result["runs"], RUN_COLUMNS)
        _write_csv(temp / "missing_run_neighbor_pose.csv", result["neighbors"],
                   ("video_id", "split", "label", "run_id", "start_index", "end_index",
                    "nearest_valid_before_index", "nearest_valid_before_yaw",
                    "nearest_valid_after_index", "nearest_valid_after_yaw", "hint_scope"))
        _write_csv(temp / "missing_run_length_histogram.csv", summary["missing_runs"]["length_histogram"],
                   ("run_length", "count", "fraction"))
        _write_csv(temp / "behavior_missing_count_histogram.csv", result["histograms"]["behavior"],
                   ("missing_count", "video_count", "video_fraction"))
        _write_csv(temp / "context_missing_count_histogram.csv", result["histograms"]["context"],
                   ("context_missing_count", "video_count", "video_fraction"))
        _write_csv(temp / "behavior_missing_cumulative.csv", result["behavior_cumulative"],
                   ("max_missing_allowed", "videos_satisfying", "fraction"))
        _write_csv(temp / "context_missing_cumulative.csv", result["context_cumulative"],
                   ("max_missing_allowed", "videos_satisfying", "fraction"))
        _write_csv(temp / "temporal_missing_by_canonical_slot.csv", result["temporal"],
                   ("canonical_index", "total_frames", "yunet_missing_count", "yunet_missing_rate"))
        _write_csv(temp / "temporal_missing_by_context_slot.csv", result["context_temporal"],
                   ("context_slot", "selected_count", "missing_count", "missing_rate"))
        _write_csv(temp / "missing_concentration_summary.csv", result["concentration"],
                   ("top_video_percent", "top_video_count", "missing_in_top_videos", "fraction_of_all_missing"))
        for name, rows in (("top_missing_videos.csv", result["top_missing"]),
                           ("top_context_missing_videos.csv", result["top_context"]),
                           ("top_consecutive_missing_videos.csv", result["top_runs"])):
            _write_csv(temp / name, rows, VIDEO_COLUMNS)
        _write_csv(temp / "numeric_quality_anomalies.csv", result["anomalies"],
                   ("video_id", "split", "label", "canonical_index", "issue"))
        _write_csv(temp / "step4b_review_candidates.csv", result["candidates"], CANDIDATE_COLUMNS)
        if set(path.name for path in temp.iterdir()) != set(OUTPUT_FILES):
            raise RuntimeError("audit 출력 파일 목록 불일치")
        if output_dir.exists():
            raise FileExistsError(f"발행 중 기존 출력 발견: {output_dir}")
        os.replace(temp, output_dir)
    except Exception:
        if temp.exists():
            if temp.resolve().parent != output_dir.parent.resolve():
                raise RuntimeError("audit 임시 출력 경로가 허용 범위를 벗어났습니다")
            shutil.rmtree(temp)
        raise
    return result["summary"]
