"""STEP 5-A Context 32-slot을 기존 JPEG 참조·mask·출처 metadata로 구성한다."""

from __future__ import annotations

import csv
import json
import math
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from drowsiness_detection.evaluation_v2.missing_policy_analysis import missing_runs, nearest_valid_slot
from drowsiness_detection.evaluation_v2.missing_policy_freeze import (
    BEHAVIOR_POLICY_ID, CONTEXT_POLICY_ID, POLICY_HASH, load_config,
    sequence_missing_policy_hash,
)
from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256, stable_hash


SEQUENCE_POLICY_HASH = "f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d"
FRAME_COLUMNS = (
    "video_id", "split", "label", "target_context_index", "target_canonical_index",
    "target_timestamp_sec", "original_available", "imputed", "source_context_index",
    "source_canonical_index", "source_timestamp_sec", "source_image_relpath",
    "imputation_offset_context_slots", "imputation_distance_context_slots",
    "imputation_offset_canonical_slots", "imputation_distance_canonical_slots",
    "imputation_offset_sec", "imputation_distance_sec", "canonical_policy_hash",
    "sequence_missing_policy_hash",
)
VIDEO_COLUMNS = (
    "video_id", "split", "label", "sequence_length", "original_valid_slots", "imputed_slots",
    "context_eligible", "behavior_eligible", "source_metadata_fingerprint",
    "canonical_policy_hash", "sequence_missing_policy_hash",
)
EXCLUDED_COLUMNS = (
    "video_id", "split", "label", "context_missing_count", "longest_context_missing_run",
    "context_exclusion_reason", "canonical_policy_hash", "sequence_missing_policy_hash",
)
HISTOGRAM_COLUMNS = ("imputed_slots", "videos", "fraction_of_eligible_videos")
GROUP_COLUMNS = ("group_type", "group_name", "videos", "original_valid_slots",
                 "imputed_slots", "videos_requiring_imputation")
EXPECTED = {
    "input_videos": 1763, "eligible_videos": 1677, "excluded_videos": 86,
    "frame_rows": 53664, "original_valid_rows": 52965, "imputed_rows": 699,
    "videos_without_imputation": 1425, "videos_requiring_imputation": 252,
}


@dataclass(frozen=True)
class VideoPlan:
    """한 영상의 원본 Context 상태와 확정된 target→source 참조."""

    video_id: str
    split: str
    label: str
    behavior_eligible: bool
    rows: tuple[dict[str, Any], ...]
    original_mask: tuple[bool, ...]
    imputed_mask: tuple[bool, ...]
    fingerprint: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _yes(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _relative_source_image(project_root: Path, canonical_root: Path, split: str,
                           video_id: str, relpath: str) -> tuple[str, Path]:
    """Canonical bundle 밖 경로·절대 경로를 차단하고 이식 가능한 참조를 만든다."""
    relative = Path(relpath)
    bundle = canonical_root / "per_video" / split / video_id
    source = bundle / relative
    if (relative.is_absolute() or ".." in relative.parts
            or not canonical_root.is_relative_to(project_root)):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: Context JPEG 경로 이탈 {video_id}: {relpath}")
    return source.relative_to(project_root).as_posix(), source


def build_video_plan(video_id: str, split: str, label: str,
                     selected: Sequence[Mapping[str, str]], behavior_eligible: bool,
                     canonical_root: Path, project_root: Path,
                     check_images: bool = True) -> VideoPlan:
    """원본 유효 slot만 source로 사용해 결정적 32행을 구성한다."""
    if split not in {"train", "val"}:
        raise ValueError("STEP_5A_BLOCKED_TEST_LEAKAGE: video split")
    if not video_id or Path(video_id).name != video_id:
        raise ValueError("잘못된 video_id")
    rows = sorted(selected, key=lambda row: int(row["context_slot"]))
    if len(rows) != 32 or [int(row["context_slot"]) for row in rows] != list(range(32)):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: Context slot 0..31/32행 불일치 {video_id}")
    if any(row["video_id"] != video_id or row["split"] != split or row["label"] != label
           or not _yes(row["context_selected"]) for row in rows):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: target metadata 불일치 {video_id}")
    available = tuple(_yes(row["context_crop_available"]) for row in rows)
    if not any(available):
        raise ValueError(f"STEP_5A_BLOCKED_NO_VALID_SOURCE: {video_id}")
    indices = [int(row["canonical_index"]) for row in rows]
    timestamps = [float(row["actual_timestamp_sec"]) for row in rows]
    if len(set(indices)) != 32 or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: target canonical index 순서 오류 {video_id}")
    if any(not math.isfinite(value) for value in timestamps) or any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: target timestamp 순서 오류 {video_id}")
    valid_sources: dict[int, tuple[str, Path]] = {}
    for index, row in enumerate(rows):
        if not available[index]:
            continue
        relpath = row.get("context_crop_relpath", "")
        if not relpath:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 유효 Context JPEG 경로 누락 {video_id}/ctx{index}")
        source_ref = _relative_source_image(project_root, canonical_root, split, video_id, relpath)
        if check_images and not source_ref[1].is_file():
            raise FileNotFoundError(f"STEP_5A_REVIEW_REQUIRED: Context JPEG 파일 없음 {source_ref[1]}")
        valid_sources[index] = source_ref
    output_rows: list[dict[str, Any]] = []
    for target, row in enumerate(rows):
        source = target if available[target] else nearest_valid_slot(available, target)
        if source is None or source not in valid_sources or not available[source]:
            raise ValueError(f"STEP_5A_BLOCKED_NO_VALID_SOURCE: {video_id}/ctx{target}")
        source_index = indices[source]
        source_timestamp = timestamps[source]
        context_offset = source - target
        canonical_offset = source_index - indices[target]
        time_offset = source_timestamp - timestamps[target]
        output_rows.append({
            "video_id": video_id, "split": split, "label": label,
            "target_context_index": target, "target_canonical_index": indices[target],
            "target_timestamp_sec": timestamps[target], "original_available": available[target],
            "imputed": not available[target], "source_context_index": source,
            "source_canonical_index": source_index, "source_timestamp_sec": source_timestamp,
            "source_image_relpath": valid_sources[source][0],
            "imputation_offset_context_slots": context_offset,
            "imputation_distance_context_slots": abs(context_offset),
            "imputation_offset_canonical_slots": canonical_offset,
            "imputation_distance_canonical_slots": abs(canonical_offset),
            "imputation_offset_sec": time_offset, "imputation_distance_sec": abs(time_offset),
            "canonical_policy_hash": POLICY_HASH, "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        })
    fingerprint = stable_hash({"video_id": video_id, "split": split, "label": label,
                               "targets": [{"context_slot": index, "canonical_index": indices[index],
                                            "timestamp_sec": timestamps[index], "original_available": available[index],
                                            "source_image_relpath": valid_sources[index][0] if available[index] else ""}
                                           for index in range(32)]})
    plan = VideoPlan(video_id, split, label, behavior_eligible, tuple(output_rows),
                     available, tuple(not value for value in available), fingerprint)
    validate_video_plan(plan, project_root, check_images=check_images)
    return plan


def validate_video_plan(plan: VideoPlan, project_root: Path, check_images: bool = True) -> None:
    """Mask·최근접·원본 source·타임스탬프·참조 파일을 독립 재계산한다."""
    if len(plan.rows) != 32 or len(plan.original_mask) != 32 or len(plan.imputed_mask) != 32:
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 32-slot shape 오류 {plan.video_id}")
    if [int(row["target_context_index"]) for row in plan.rows] != list(range(32)):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: target 순서 오류 {plan.video_id}")
    if any(valid == imputed for valid, imputed in zip(plan.original_mask, plan.imputed_mask)):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: original/imputed mask 불일치 {plan.video_id}")
    for target, row in enumerate(plan.rows):
        source = int(row["source_context_index"])
        if not 0 <= source < 32 or not plan.original_mask[source]:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: chained/비유효 source {plan.video_id}/ctx{target}")
        expected_source = target if plan.original_mask[target] else nearest_valid_slot(plan.original_mask, target)
        if source != expected_source:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: nearest source 불일치 {plan.video_id}/ctx{target}")
        source_row = plan.rows[source]
        if (int(row["source_canonical_index"]) != int(source_row["target_canonical_index"])
                or float(row["source_timestamp_sec"]) != float(source_row["target_timestamp_sec"])
                or row["source_image_relpath"] != source_row["source_image_relpath"]):
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: source provenance 불일치 {plan.video_id}/ctx{target}")
        if (_yes(row["original_available"]) != plan.original_mask[target]
                or _yes(row["imputed"]) != plan.imputed_mask[target]
                or int(row["imputation_offset_context_slots"]) != source - target
                or int(row["imputation_distance_context_slots"]) != abs(source - target)
                or int(row["imputation_offset_canonical_slots"]) != int(row["source_canonical_index"]) - int(row["target_canonical_index"])):
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: mask/offset 불일치 {plan.video_id}/ctx{target}")
        relative = Path(row["source_image_relpath"])
        source_path = project_root / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 절대/외부 이미지 경로 {plan.video_id}/ctx{target}")
        if check_images and not source_path.is_file():
            raise FileNotFoundError(f"STEP_5A_REVIEW_REQUIRED: source JPEG 없음 {source_path}")
        if row["canonical_policy_hash"] != POLICY_HASH or row["sequence_missing_policy_hash"] != SEQUENCE_POLICY_HASH:
            raise ValueError(f"STEP_5A_BLOCKED_POLICY_MISMATCH: {plan.video_id}")


def load_plan(canonical_root: Path, freeze_root: Path, config_path: Path,
              project_root: Path, check_images: bool = True,
              expected: Mapping[str, int] = EXPECTED) -> tuple[list[VideoPlan], list[dict[str, str]]]:
    """정책·test·적격성·32-slot과 기존 JPEG를 결과 쓰기 전에 전수 검증한다."""
    config = load_config(config_path)
    semantic_hash = sequence_missing_policy_hash(config)
    if semantic_hash != SEQUENCE_POLICY_HASH:
        raise ValueError("STEP_5A_BLOCKED_POLICY_MISMATCH: sequence missing semantic hash")
    frozen = json.loads((freeze_root / "sequence_missing_policy_frozen.json").read_text(encoding="utf-8"))
    canonical_run = json.loads((canonical_root / "metadata/run_metadata.json").read_text(encoding="utf-8"))
    if (frozen.get("sequence_missing_policy_hash") != SEQUENCE_POLICY_HASH
            or frozen.get("canonical_policy_hash") != POLICY_HASH
            or frozen.get("context_policy", {}).get("policy_id") != CONTEXT_POLICY_ID
            or frozen.get("behavior_policy", {}).get("policy_id") != BEHAVIOR_POLICY_ID
            or not frozen.get("test_sealed") or canonical_run.get("policy_hash") != POLICY_HASH):
        raise ValueError("STEP_5A_BLOCKED_POLICY_MISMATCH: frozen/canonical metadata")
    frozen_summary = json.loads((freeze_root / "sequence_missing_policy_summary.json").read_text(encoding="utf-8"))
    if (frozen_summary.get("sequence_missing_policy_hash") != SEQUENCE_POLICY_HASH
            or frozen_summary.get("context_eligible") != expected["eligible_videos"]
            or frozen_summary.get("test_rows") != 0):
        raise ValueError("STEP_5A_BLOCKED_POLICY_MISMATCH: frozen summary")
    eligibility = _read_csv(freeze_root / "sequence_missing_policy_eligibility.csv")
    if any(row["split"] not in {"train", "val"} for row in eligibility):
        raise ValueError("STEP_5A_BLOCKED_TEST_LEAKAGE: frozen eligibility")
    manifest = {row["video_id"]: row for row in eligibility}
    if len(manifest) != len(eligibility) or len(manifest) != expected["input_videos"]:
        raise ValueError("STEP_5A_REVIEW_REQUIRED: frozen eligibility ID/행 수 불일치")
    video_rows = _read_csv(canonical_root / "metadata/canonical_videos.csv")
    if any(row["split"] not in {"train", "val"} for row in video_rows):
        raise ValueError("STEP_5A_BLOCKED_TEST_LEAKAGE: canonical_videos")
    video_map = {row["video_id"]: row for row in video_rows}
    if len(video_map) != len(video_rows) or set(video_map) != set(manifest):
        raise ValueError("STEP_5A_REVIEW_REQUIRED: canonical video/eligibility ID 불일치")
    selected_by_video: dict[str, list[dict[str, str]]] = {video_id: [] for video_id in manifest}
    frame_counts = Counter()
    with (canonical_root / "metadata/canonical_frames.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "val"}:
                raise ValueError("STEP_5A_BLOCKED_TEST_LEAKAGE: canonical_frames")
            video_id = row["video_id"]
            if video_id not in manifest:
                raise ValueError("STEP_5A_REVIEW_REQUIRED: canonical frame video ID 오류")
            frame_counts[video_id] += 1
            if _yes(row["context_selected"]):
                selected_by_video[video_id].append(row)
    if any(frame_counts[video_id] != 100 for video_id in manifest):
        raise ValueError("STEP_5A_REVIEW_REQUIRED: 영상당 canonical 100 frame 불일치")
    plans: list[VideoPlan] = []
    excluded: list[dict[str, str]] = []
    for video_id in sorted(manifest):
        entry = manifest[video_id]
        video = video_map[video_id]
        if (entry["split"] != video["split"] or entry["label"] != video["label"]
                or entry["canonical_policy_hash"] != POLICY_HASH
                or entry["sequence_missing_policy_hash"] != SEQUENCE_POLICY_HASH
                or video["policy_hash"] != POLICY_HASH):
            raise ValueError(f"STEP_5A_BLOCKED_POLICY_MISMATCH: {video_id}")
        selected = selected_by_video[video_id]
        if len(selected) != 32 or len({row["context_slot"] for row in selected}) != 32:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 32 Context target 오류 {video_id}")
        available = tuple(_yes(row["context_crop_available"]) for row in sorted(selected, key=lambda row: int(row["context_slot"])))
        longest = max((end - start + 1 for start, end in missing_runs(available)), default=0)
        missing_count = 32 - sum(available)
        if (missing_count != int(entry["context_missing_count"])
                or longest != int(entry["longest_context_missing_run"])):
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: frozen eligibility와 Context metadata 불일치 {video_id}")
        eligible = _yes(entry["context_eligible"])
        expected_eligible = missing_count <= 8 and longest <= 4 and any(available)
        if eligible != expected_eligible:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: C3 적격성 불일치 {video_id}")
        if eligible:
            plans.append(build_video_plan(video_id, entry["split"], entry["label"], selected,
                                          _yes(entry["behavior_eligible"]), canonical_root,
                                          project_root, check_images=check_images))
        else:
            excluded.append({key: entry[key] for key in EXCLUDED_COLUMNS})
    if len(plans) != expected["eligible_videos"] or len(excluded) != expected["excluded_videos"]:
        raise ValueError("STEP_5A_REVIEW_REQUIRED: Context 적격/제외 수 불일치")
    if expected["input_videos"] == 1763 and any(plan.video_id == "n_246" for plan in plans):
        raise ValueError("STEP_5A_REVIEW_REQUIRED: n_246 Context sequence 생성 금지")
    return plans, excluded


def plan_statistics(plans: Sequence[VideoPlan], excluded: Sequence[Mapping[str, str]],
                    expected: Mapping[str, int] = EXPECTED) -> dict[str, Any]:
    """동결된 STEP 4-C1 수치와 실제 생성 계획을 대조한다."""
    counts = Counter({
        "input_videos": len(plans) + len(excluded),
        "eligible_videos": len(plans), "excluded_videos": len(excluded),
        "frame_rows": sum(len(plan.rows) for plan in plans),
        "original_valid_rows": sum(sum(plan.original_mask) for plan in plans),
        "imputed_rows": sum(sum(plan.imputed_mask) for plan in plans),
        "videos_without_imputation": sum(not any(plan.imputed_mask) for plan in plans),
        "videos_requiring_imputation": sum(any(plan.imputed_mask) for plan in plans),
    })
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: {key}: {counts[key]} != {value}")
    if expected["input_videos"] == 1763 and "n_246" not in {row["video_id"] for row in excluded}:
        raise ValueError("STEP_5A_REVIEW_REQUIRED: n_246 제외 manifest 누락")
    if len({plan.video_id for plan in plans}) != len(plans):
        raise ValueError("STEP_5A_REVIEW_REQUIRED: 중복 video ID")
    return dict(counts)


def sample_decode(plans: Sequence[VideoPlan], project_root: Path, limit: int = 256) -> int:
    """결정적인 고유 JPEG 표본만 읽어 224×224 RGB 입력 크기를 확인한다."""
    paths = sorted({row["source_image_relpath"] for plan in plans for row in plan.rows})
    if not paths:
        return 0
    selected = [paths[index] for index in np.linspace(0, len(paths) - 1, min(limit, len(paths)), dtype=int)]
    for relative in selected:
        image = cv2.imdecode(np.fromfile(project_root / relative, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (224, 224, 3):
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: JPEG 표본 디코드/크기 실패 {relative}")
    return len(selected)


def _bundle_arrays(plan: VideoPlan) -> dict[str, np.ndarray]:
    return {
        "target_context_index": np.arange(32, dtype=np.int16),
        "target_canonical_index": np.asarray([row["target_canonical_index"] for row in plan.rows], dtype=np.int16),
        "source_context_index": np.asarray([row["source_context_index"] for row in plan.rows], dtype=np.int16),
        "source_canonical_index": np.asarray([row["source_canonical_index"] for row in plan.rows], dtype=np.int16),
        "original_context_valid_mask": np.asarray(plan.original_mask, dtype=np.bool_),
        "context_imputed_mask": np.asarray(plan.imputed_mask, dtype=np.bool_),
        "imputation_distance_context_slots": np.asarray(
            [row["imputation_distance_context_slots"] for row in plan.rows], dtype=np.float32),
        "imputation_distance_sec": np.asarray(
            [row["imputation_distance_sec"] for row in plan.rows], dtype=np.float32),
    }


def validate_bundle(bundle: Path, plan: VideoPlan) -> None:
    """COMPLETE 표시와 파일 해시, 32행 CSV, mask를 생성 계획과 대조한다."""
    marker = json.loads((bundle / "COMPLETE.json").read_text(encoding="utf-8"))
    if (marker.get("canonical_policy_hash") != POLICY_HASH
            or marker.get("sequence_missing_policy_hash") != SEQUENCE_POLICY_HASH
            or marker.get("source_metadata_fingerprint") != plan.fingerprint
            or marker.get("video_id") != plan.video_id):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle hash/fingerprint 충돌 {plan.video_id}")
    for name in ("sequence.csv", "masks.npz", "summary.json"):
        if marker.get("file_sha256", {}).get(name) != file_sha256(bundle / name):
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle 파일 해시 충돌 {plan.video_id}/{name}")
    records = _read_csv(bundle / "sequence.csv")
    if len(records) != 32 or any(set(record) != set(FRAME_COLUMNS) for record in records):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle CSV 스키마 {plan.video_id}")
    for actual, planned in zip(records, plan.rows):
        if actual != {key: str(value) for key, value in planned.items()}:
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle CSV 내용 충돌 {plan.video_id}")
    with np.load(bundle / "masks.npz", allow_pickle=False) as loaded:
        expected_arrays = _bundle_arrays(plan)
        if set(loaded.files) != set(expected_arrays):
            raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle mask 키 충돌 {plan.video_id}")
        for key, expected_array in expected_arrays.items():
            if loaded[key].dtype != expected_array.dtype or not np.array_equal(loaded[key], expected_array):
                raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle mask 값 충돌 {plan.video_id}/{key}")
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("video_id") != plan.video_id or summary.get("sequence_length") != 32
            or summary.get("original_valid_slots") != sum(plan.original_mask)
            or summary.get("imputed_slots") != sum(plan.imputed_mask)):
        raise ValueError(f"STEP_5A_REVIEW_REQUIRED: bundle 요약 충돌 {plan.video_id}")


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish_bundle(root: Path, plan: VideoPlan) -> None:
    """임시 bundle을 완전히 검증한 뒤 최종 위치에 원자적으로 게시한다."""
    parent = root / "per_video" / plan.split
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / plan.video_id
    if final.exists():
        raise FileExistsError(f"STEP_5A_REVIEW_REQUIRED: 기존 bundle 덮어쓰기 금지 {final}")
    with tempfile.TemporaryDirectory(prefix=f".step5a_{plan.video_id}_", dir=parent) as name:
        staged = Path(name)
        _write_csv(staged / "sequence.csv", FRAME_COLUMNS, plan.rows)
        np.savez_compressed(staged / "masks.npz", **_bundle_arrays(plan))
        _json_write(staged / "summary.json", {
            "video_id": plan.video_id, "split": plan.split, "label": plan.label,
            "sequence_length": 32, "original_valid_slots": sum(plan.original_mask),
            "imputed_slots": sum(plan.imputed_mask), "context_eligible": True,
            "behavior_eligible": plan.behavior_eligible,
            "source_metadata_fingerprint": plan.fingerprint,
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        })
        _json_write(staged / "COMPLETE.json", {
            "video_id": plan.video_id, "source_metadata_fingerprint": plan.fingerprint,
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
            "file_sha256": {filename: file_sha256(staged / filename)
                            for filename in ("sequence.csv", "masks.npz", "summary.json")},
        })
        validate_bundle(staged, plan)
        staged.rename(final)


def _distance_statistics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = sorted(float(row[key]) for row in rows)
    return {"min": values[0], "mean": statistics.mean(values),
            "median": statistics.median(values), "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)), "max": values[-1]}


def _write_once_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"STEP_5A_REVIEW_REQUIRED: 기존 결과 덮어쓰기 금지 {path}")
    _write_csv(path, columns, rows)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"STEP_5A_REVIEW_REQUIRED: 기존 결과 덮어쓰기 금지 {path}")
    _json_write(path, value)


def materialize(plans: Sequence[VideoPlan], excluded: Sequence[Mapping[str, str]],
                root: Path, report_root: Path, project_root: Path,
                resume: bool = False, expected: Mapping[str, int] = EXPECTED,
                decode_limit: int = 256) -> dict[str, Any]:
    """기존 JPEG를 참조하는 시퀀스, mask, 출처 및 검증 보고서만 생성한다."""
    counts = plan_statistics(plans, excluded, expected)
    if root.exists() and not resume:
        raise FileExistsError(f"STEP_5A_REVIEW_REQUIRED: 출력 경로가 이미 존재함 {root}")
    if report_root.exists() and not resume:
        raise FileExistsError(f"STEP_5A_REVIEW_REQUIRED: 보고서 경로가 이미 존재함 {report_root}")
    for plan in plans:
        validate_video_plan(plan, project_root)
    decoded = sample_decode(plans, project_root, decode_limit)
    created = skipped = 0
    for plan in plans:
        bundle = root / "per_video" / plan.split / plan.video_id
        if bundle.exists():
            if not resume:
                raise FileExistsError(f"STEP_5A_REVIEW_REQUIRED: 기존 bundle {bundle}")
            validate_bundle(bundle, plan)
            skipped += 1
        else:
            publish_bundle(root, plan)
            created += 1
    root.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata"
    frame_rows = [row for plan in plans for row in plan.rows]
    video_rows = [{"video_id": plan.video_id, "split": plan.split, "label": plan.label,
                   "sequence_length": 32, "original_valid_slots": sum(plan.original_mask),
                   "imputed_slots": sum(plan.imputed_mask), "context_eligible": True,
                   "behavior_eligible": plan.behavior_eligible,
                   "source_metadata_fingerprint": plan.fingerprint,
                   "canonical_policy_hash": POLICY_HASH,
                   "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH} for plan in plans]
    histogram = Counter(sum(plan.imputed_mask) for plan in plans)
    histogram_rows = [{"imputed_slots": value, "videos": histogram[value],
                       "fraction_of_eligible_videos": histogram[value] / len(plans)}
                      for value in range(9)]
    groups = []
    for group_type in ("split", "label", "split_x_label"):
        names = sorted({(plan.split if group_type == "split" else plan.label if group_type == "label"
                         else f"{plan.split}/{plan.label}") for plan in plans})
        for name in names:
            members = [plan for plan in plans if (plan.split if group_type == "split" else plan.label
                        if group_type == "label" else f"{plan.split}/{plan.label}") == name]
            groups.append({"group_type": group_type, "group_name": name, "videos": len(members),
                           "original_valid_slots": sum(sum(plan.original_mask) for plan in members),
                           "imputed_slots": sum(sum(plan.imputed_mask) for plan in members),
                           "videos_requiring_imputation": sum(any(plan.imputed_mask) for plan in members)})
    imputed = [row for row in frame_rows if row["imputed"]]
    summary = {**counts, "status": "STEP_5A_CONTEXT_SEQUENCE_CONSTRUCTION_COMPLETE",
               "sequence_length": 32, "canonical_policy_hash": POLICY_HASH,
               "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
               "context_policy_id": CONTEXT_POLICY_ID, "test_rows": 0, "test_access": 0,
               "chained_imputation_count": 0, "nearest_source_mismatch_count": 0,
               "missing_source_jpeg_count": 0, "new_or_copied_jpeg_count": 0,
               "sample_decoded_unique_jpegs": decoded,
               "past_source_count": sum(row["imputation_offset_context_slots"] < 0 for row in imputed),
               "future_source_count": sum(row["imputation_offset_context_slots"] > 0 for row in imputed),
               "distance_statistics": {
                   "context_slots": _distance_statistics(imputed, "imputation_distance_context_slots"),
                   "canonical_slots": _distance_statistics(imputed, "imputation_distance_canonical_slots"),
                   "seconds": _distance_statistics(imputed, "imputation_distance_sec")},
               "imputation_histogram": histogram_rows, "split_label_summary": groups,
               "bundles_created_this_run": created, "bundles_skipped_this_run": skipped,
               "canonical_data_modified": False, "raw_video_opened": False,
               "cnn_feature_extraction_started": False}
    outputs = [
        (metadata / "context_sequence_frames.csv", FRAME_COLUMNS, frame_rows),
        (metadata / "context_sequence_videos.csv", VIDEO_COLUMNS, video_rows),
        (metadata / "context_sequence_excluded_videos.csv", EXCLUDED_COLUMNS, excluded),
        (report_root / "excluded_context_videos.csv", EXCLUDED_COLUMNS, excluded),
        (report_root / "context_sequence_imputation_histogram.csv", HISTOGRAM_COLUMNS, histogram_rows),
        (report_root / "context_sequence_split_label_summary.csv", GROUP_COLUMNS, groups),
    ]
    for path, columns, rows in outputs:
        if path.exists() and resume:
            if _read_csv(path) != [{key: str(value) for key, value in row.items()} for row in rows]:
                raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 기존 manifest 내용 충돌 {path}")
        else:
            _write_once_csv(path, columns, rows)
    metadata_run = {"status": summary["status"], "canonical_policy_hash": POLICY_HASH,
                    "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
                    "input_videos": counts["input_videos"], "eligible_videos": counts["eligible_videos"],
                    "sequence_rows": counts["frame_rows"], "test_sealed": True,
                    "source_metadata_fingerprints": stable_hash({plan.video_id: plan.fingerprint for plan in plans})}
    for path, value in ((metadata / "run_metadata.json", metadata_run),
                        (report_root / "context_sequence_summary.json", {key: value for key, value in summary.items()
                         if key not in {"bundles_created_this_run", "bundles_skipped_this_run"}})):
        if path.exists() and resume:
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 기존 JSON 내용 충돌 {path}")
        else:
            _write_once_json(path, value)
    reports = {
        "context_sequence_integrity_report.txt": "STEP 5-A INTEGRITY: PASS\n32 slots/video; masks, hashes, source paths, nearest-valid, no chained imputation: PASS\nTEST: SEALED\n",
        "context_sequence_generation_report.txt": f"STEP 5-A COMPLETE\nBundles: {len(plans)}\nRows: {len(frame_rows)}\nOriginal: {counts['original_valid_rows']}\nImputed: {counts['imputed_rows']}\nJPEG created/copied: 0\n",
    }
    for name, content in reports.items():
        path = report_root / name
        if path.exists() and resume:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"STEP_5A_REVIEW_REQUIRED: 기존 보고서 내용 충돌 {path}")
        else:
            path.write_text(content, encoding="utf-8")
    (report_root / "resume_report.txt").write_text(
        f"resume={resume}\ncreated={created}\nskipped_verified={skipped}\nconflicts=0\n", encoding="utf-8")
    return summary
