"""동결 B2 정책으로 100-slot Behavior 연속값·NaN·mask를 패키징한다."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drowsiness_detection.evaluation_v2.missing_policy_analysis import missing_runs
from drowsiness_detection.evaluation_v2.missing_policy_freeze import (
    BEHAVIOR_POLICY_ID, POLICY_HASH, load_config, sequence_missing_policy_hash,
)
from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256, stable_hash


SEQUENCE_POLICY_HASH = "f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d"
FEATURE_NAMES = ("ear", "mar", "pitch_raw", "pitch_centered_candidate", "yaw", "roll")
FEATURE_SCHEMA = {
    "feature_names": list(FEATURE_NAMES), "feature_dtype": "float32",
    "feature_valid_mask": "isfinite_per_feature",
    "behavior_valid_mask": "yunet_success AND landmark_success AND finite(ear) AND finite(mar)",
    "detector_valid_mask": "yunet_success", "landmark_valid_mask": "landmark_success",
    "pose_valid_mask": "pose_success AND finite(pitch_raw,pitch_centered_candidate,yaw,roll)",
    "missing_semantics": "NaN", "interpolation": "none",
}
FEATURE_SCHEMA_HASH = stable_hash(FEATURE_SCHEMA)
FRAME_COLUMNS = (
    "video_id", "split", "label", "canonical_index", "target_timestamp_sec",
    "yunet_success", "landmark_success", "pose_success", "behavior_valid",
    *FEATURE_NAMES, "canonical_policy_hash", "sequence_missing_policy_hash",
)
VIDEO_COLUMNS = (
    "video_id", "split", "label", "sequence_length", "behavior_valid_count",
    "behavior_missing_count", "longest_yunet_missing_run", "context_eligible",
    "behavior_eligible", "source_metadata_fingerprint", "behavior_feature_schema_hash",
    "canonical_policy_hash", "sequence_missing_policy_hash",
)
EXCLUDED_COLUMNS = (
    "video_id", "split", "label", "behavior_valid_count", "behavior_missing_count",
    "longest_missing_run", "behavior_exclusion_reason", "context_eligible",
    "canonical_policy_hash", "sequence_missing_policy_hash",
)
HISTOGRAM_COLUMNS = ("metric", "value", "videos", "fraction_of_eligible_videos")
GROUP_COLUMNS = ("group_type", "group_name", "videos", "behavior_valid_slots",
                 "behavior_missing_slots")
FEATURE_SUMMARY_COLUMNS = ("feature", "count", "mean", "std", "min", "p01", "p05",
                           "median", "p95", "p99", "max")
EXPECTED = {"input_videos": 1763, "eligible_videos": 1520, "excluded_videos": 243,
            "frame_rows": 152000}


@dataclass(frozen=True)
class BehaviorPlan:
    """한 영상의 canonical 연속값과 서로 독립적인 유효 mask를 보존한다."""

    video_id: str
    split: str
    label: str
    context_eligible: bool
    rows: tuple[dict[str, Any], ...]
    features: np.ndarray
    feature_valid_mask: np.ndarray
    behavior_valid_mask: np.ndarray
    detector_valid_mask: np.ndarray
    landmark_valid_mask: np.ndarray
    pose_valid_mask: np.ndarray
    longest_yunet_missing_run: int
    fingerprint: str


def _yes(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _number(value: Any) -> float:
    if value is None or str(value).strip().casefold() in {"", "nan", "none"}:
        return math.nan
    number = float(value)
    if math.isinf(number):
        raise ValueError("STEP_5B_REVIEW_REQUIRED: Behavior feature infinity")
    return number


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_video_plan(video_id: str, split: str, label: str,
                     frames: Sequence[Mapping[str, str]], context_eligible: bool) -> BehaviorPlan:
    """새 계산이나 보간 없이 canonical 100행을 같은 순서로 float32에 담는다."""
    if split not in {"train", "val"}:
        raise ValueError("STEP_5B_BLOCKED_TEST_LEAKAGE: video split")
    ordered = sorted(frames, key=lambda row: int(row["canonical_index"]))
    if len(ordered) != 100 or [int(row["canonical_index"]) for row in ordered] != list(range(100)):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: canonical index 0..99/100 불일치 {video_id}")
    if any(row["video_id"] != video_id or row["split"] != split or row["label"] != label
           for row in ordered):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: frame identity 불일치 {video_id}")
    timestamps = np.asarray([float(row["actual_timestamp_sec"]) for row in ordered], dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: timestamp 불일치 {video_id}")
    features64 = np.asarray([[_number(row[name]) for name in FEATURE_NAMES] for row in ordered],
                            dtype=np.float64)
    features = features64.astype(np.float32)
    feature_valid = np.isfinite(features64)
    detector = np.asarray([_yes(row["yunet_success"]) for row in ordered], dtype=np.bool_)
    landmark = np.asarray([_yes(row["landmark_success"]) for row in ordered], dtype=np.bool_)
    pose_success = np.asarray([_yes(row["pose_success"]) for row in ordered], dtype=np.bool_)
    pose = pose_success & feature_valid[:, 2:].all(axis=1)
    behavior = detector & landmark & feature_valid[:, :2].all(axis=1)
    longest = max((end - start + 1 for start, end in missing_runs(detector)), default=0)
    output_rows = []
    for index, source in enumerate(ordered):
        output_rows.append({
            "video_id": video_id, "split": split, "label": label,
            "canonical_index": index, "target_timestamp_sec": timestamps[index],
            "yunet_success": bool(detector[index]), "landmark_success": bool(landmark[index]),
            "pose_success": bool(pose_success[index]), "behavior_valid": bool(behavior[index]),
            **{name: float(features[index, column]) for column, name in enumerate(FEATURE_NAMES)},
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        })
    semantic_features = [[None if not math.isfinite(value) else value for value in row]
                         for row in features64.tolist()]
    fingerprint = stable_hash({
        "video_id": video_id, "split": split, "label": label,
        "timestamps": timestamps.tolist(), "features": semantic_features,
        "detector": detector.tolist(), "landmark": landmark.tolist(),
        "pose_success": pose_success.tolist(), "feature_schema_hash": FEATURE_SCHEMA_HASH,
    })
    plan = BehaviorPlan(video_id, split, label, context_eligible, tuple(output_rows), features,
                        feature_valid, behavior, detector, landmark, pose, longest, fingerprint)
    validate_video_plan(plan, ordered)
    return plan


def validate_video_plan(plan: BehaviorPlan,
                        canonical_rows: Sequence[Mapping[str, str]] | None = None) -> None:
    """100-slot shape, NaN/finite, B2 mask 의미와 canonical 수치 보존을 검사한다."""
    arrays = (plan.behavior_valid_mask, plan.detector_valid_mask,
              plan.landmark_valid_mask, plan.pose_valid_mask)
    if len(plan.rows) != 100 or plan.features.shape != (100, len(FEATURE_NAMES)):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: 100-slot feature shape {plan.video_id}")
    if plan.feature_valid_mask.shape != plan.features.shape or any(
            array.shape != (100,) or array.dtype != np.bool_ for array in arrays):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: mask shape/dtype {plan.video_id}")
    if plan.features.dtype != np.float32 or plan.feature_valid_mask.dtype != np.bool_:
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: feature/mask dtype {plan.video_id}")
    if np.isinf(plan.features).any():
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: infinity {plan.video_id}")
    if not np.array_equal(np.isfinite(plan.features), plan.feature_valid_mask):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: feature-valid mask {plan.video_id}")
    expected_behavior = (plan.detector_valid_mask & plan.landmark_valid_mask
                         & plan.feature_valid_mask[:, :2].all(axis=1))
    if not np.array_equal(expected_behavior, plan.behavior_valid_mask):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: behavior-valid mask {plan.video_id}")
    if np.isnan(plan.features[plan.behavior_valid_mask, :2]).any():
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: required-valid NaN contradiction {plan.video_id}")
    if plan.longest_yunet_missing_run != max(
            (end - start + 1 for start, end in missing_runs(plan.detector_valid_mask)), default=0):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: missing run 불일치 {plan.video_id}")
    if canonical_rows is not None:
        ordered = sorted(canonical_rows, key=lambda row: int(row["canonical_index"]))
        for index, source in enumerate(ordered):
            for column, name in enumerate(FEATURE_NAMES):
                original = _number(source[name])
                stored = plan.features[index, column]
                if math.isnan(original):
                    if not math.isnan(float(stored)):
                        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: canonical NaN mismatch {plan.video_id}")
                elif stored.tobytes() != np.float32(original).tobytes():
                    raise ValueError(f"STEP_5B_REVIEW_REQUIRED: canonical numeric mismatch {plan.video_id}")


def load_plan(canonical_root: Path, freeze_root: Path, config_path: Path,
              expected: Mapping[str, int] = EXPECTED) -> tuple[list[BehaviorPlan], list[dict[str, Any]]]:
    """Canonical과 frozen B2를 읽어 test·hash·적격성·count/run을 전부 대조한다."""
    config = load_config(config_path)
    if (sequence_missing_policy_hash(config) != SEQUENCE_POLICY_HASH
            or config["behavior"]["policy_id"] != BEHAVIOR_POLICY_ID):
        raise ValueError("STEP_5B_BLOCKED_POLICY_MISMATCH: sequence config")
    frozen = json.loads((freeze_root / "sequence_missing_policy_frozen.json").read_text(encoding="utf-8"))
    summary = json.loads((freeze_root / "sequence_missing_policy_summary.json").read_text(encoding="utf-8"))
    canonical_run = json.loads((canonical_root / "metadata/run_metadata.json").read_text(encoding="utf-8"))
    if (frozen.get("canonical_policy_hash") != POLICY_HASH
            or frozen.get("sequence_missing_policy_hash") != SEQUENCE_POLICY_HASH
            or frozen.get("behavior_policy", {}).get("policy_id") != BEHAVIOR_POLICY_ID
            or not frozen.get("test_sealed") or canonical_run.get("policy_hash") != POLICY_HASH
            or summary.get("behavior_eligible") != expected["eligible_videos"]
            or summary.get("test_rows") != 0):
        raise ValueError("STEP_5B_BLOCKED_POLICY_MISMATCH: frozen/canonical metadata")
    eligibility = _read_csv(freeze_root / "sequence_missing_policy_eligibility.csv")
    videos = _read_csv(canonical_root / "metadata/canonical_videos.csv")
    if any(row["split"] not in {"train", "val"} for row in eligibility + videos):
        raise ValueError("STEP_5B_BLOCKED_TEST_LEAKAGE: manifest/video")
    manifest = {row["video_id"]: row for row in eligibility}
    video_map = {row["video_id"]: row for row in videos}
    if (len(manifest) != expected["input_videos"] or len(video_map) != expected["input_videos"]
            or set(manifest) != set(video_map)):
        raise ValueError("STEP_5B_REVIEW_REQUIRED: video manifest ID/count 불일치")
    rows_by_video: dict[str, list[dict[str, str]]] = {video_id: [] for video_id in manifest}
    with (canonical_root / "metadata/canonical_frames.csv").open(
            "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "val"}:
                raise ValueError("STEP_5B_BLOCKED_TEST_LEAKAGE: canonical frames")
            if row["video_id"] not in rows_by_video:
                raise ValueError("STEP_5B_REVIEW_REQUIRED: unknown canonical video")
            rows_by_video[row["video_id"]].append(row)
    plans: list[BehaviorPlan] = []
    excluded: list[dict[str, Any]] = []
    for video_id in sorted(manifest):
        entry, video = manifest[video_id], video_map[video_id]
        if (entry["split"] != video["split"] or entry["label"] != video["label"]
                or entry["canonical_policy_hash"] != POLICY_HASH
                or entry["sequence_missing_policy_hash"] != SEQUENCE_POLICY_HASH
                or video["policy_hash"] != POLICY_HASH):
            raise ValueError(f"STEP_5B_BLOCKED_POLICY_MISMATCH: {video_id}")
        plan = build_video_plan(video_id, entry["split"], entry["label"],
                                rows_by_video[video_id], _yes(entry["context_eligible"]))
        valid_count = int(plan.behavior_valid_mask.sum())
        if (valid_count != int(entry["behavior_valid_count"])
                or 100 - valid_count != int(entry["behavior_missing_count"])
                or plan.longest_yunet_missing_run != int(entry["longest_yunet_missing_run"])):
            raise ValueError(f"STEP_5B_REVIEW_REQUIRED: frozen count/run mismatch {video_id}")
        expected_eligible = is_b2_eligible(valid_count, plan.longest_yunet_missing_run)
        if _yes(entry["behavior_eligible"]) != expected_eligible:
            raise ValueError(f"STEP_5B_REVIEW_REQUIRED: B2 eligibility mismatch {video_id}")
        if expected_eligible:
            plans.append(plan)
        else:
            excluded.append({
                "video_id": video_id, "split": entry["split"], "label": entry["label"],
                "behavior_valid_count": valid_count, "behavior_missing_count": 100 - valid_count,
                "longest_missing_run": plan.longest_yunet_missing_run,
                "behavior_exclusion_reason": entry["behavior_exclusion_reason"],
                "context_eligible": _yes(entry["context_eligible"]),
                "canonical_policy_hash": POLICY_HASH,
                "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
            })
    if len(plans) != expected["eligible_videos"] or len(excluded) != expected["excluded_videos"]:
        raise ValueError("STEP_5B_REVIEW_REQUIRED: eligible/excluded count")
    n246 = next((row for row in excluded if row["video_id"] == "n_246"), None)
    if n246 is None or n246["behavior_exclusion_reason"] != "NO_VALID_BEHAVIOR":
        raise ValueError("STEP_5B_REVIEW_REQUIRED: n_246 exclusion")
    return plans, excluded


def plan_statistics(plans: Sequence[BehaviorPlan], excluded: Sequence[Mapping[str, Any]],
                    expected: Mapping[str, int] = EXPECTED) -> dict[str, int]:
    counts = {"input_videos": len(plans) + len(excluded), "eligible_videos": len(plans),
              "excluded_videos": len(excluded), "frame_rows": sum(len(plan.rows) for plan in plans)}
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f"STEP_5B_REVIEW_REQUIRED: {key}: {counts[key]} != {value}")
    return counts


def is_b2_eligible(valid_count: int, longest_yunet_missing_run: int) -> bool:
    """동결 B2의 coverage와 연속 YuNet 결측 guard만 적용한다."""
    return valid_count >= 95 and longest_yunet_missing_run <= 5


def _bundle_arrays(plan: BehaviorPlan) -> dict[str, np.ndarray]:
    return {
        "canonical_index": np.arange(100, dtype=np.int16),
        "target_timestamp_sec": np.asarray([row["target_timestamp_sec"] for row in plan.rows], dtype=np.float64),
        "behavior_features": plan.features,
        "behavior_valid_mask": plan.behavior_valid_mask,
        "detector_valid_mask": plan.detector_valid_mask,
        "landmark_valid_mask": plan.landmark_valid_mask,
        "pose_valid_mask": plan.pose_valid_mask,
        "feature_valid_mask": plan.feature_valid_mask,
    }


def validate_bundle(bundle: Path, plan: BehaviorPlan) -> None:
    marker = json.loads((bundle / "COMPLETE.json").read_text(encoding="utf-8"))
    if (marker.get("canonical_policy_hash") != POLICY_HASH
            or marker.get("sequence_missing_policy_hash") != SEQUENCE_POLICY_HASH
            or marker.get("behavior_feature_schema_hash") != FEATURE_SCHEMA_HASH
            or marker.get("source_metadata_fingerprint") != plan.fingerprint):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: resume hash/schema conflict {plan.video_id}")
    for filename in ("sequence.csv", "behavior.npz", "summary.json"):
        if marker["file_sha256"].get(filename) != file_sha256(bundle / filename):
            raise ValueError(f"STEP_5B_REVIEW_REQUIRED: bundle file hash {plan.video_id}/{filename}")
    records = _read_csv(bundle / "sequence.csv")
    if len(records) != 100 or [int(row["canonical_index"]) for row in records] != list(range(100)):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: bundle sequence rows {plan.video_id}")
    for actual, planned in zip(records, plan.rows):
        if actual != {key: str(value) for key, value in planned.items()}:
            raise ValueError(f"STEP_5B_REVIEW_REQUIRED: bundle sequence content {plan.video_id}")
    with np.load(bundle / "behavior.npz", allow_pickle=False) as loaded:
        expected = _bundle_arrays(plan)
        if set(loaded.files) != set(expected):
            raise ValueError(f"STEP_5B_REVIEW_REQUIRED: NPZ schema {plan.video_id}")
        for key, array in expected.items():
            if loaded[key].dtype != array.dtype or not np.array_equal(loaded[key], array, equal_nan=True):
                raise ValueError(f"STEP_5B_REVIEW_REQUIRED: NPZ mismatch {plan.video_id}/{key}")
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("video_id") != plan.video_id or summary.get("sequence_length") != 100
            or summary.get("feature_names") != list(FEATURE_NAMES)
            or summary.get("feature_shape") != [100, len(FEATURE_NAMES)]
            or summary.get("behavior_feature_schema_hash") != FEATURE_SCHEMA_HASH
            or summary.get("behavior_valid_count") != int(plan.behavior_valid_mask.sum())):
        raise ValueError(f"STEP_5B_REVIEW_REQUIRED: bundle summary content {plan.video_id}")


def publish_bundle(root: Path, plan: BehaviorPlan) -> None:
    parent = root / "per_video" / plan.split
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / plan.video_id
    if final.exists():
        raise FileExistsError(f"STEP_5B_REVIEW_REQUIRED: bundle overwrite forbidden {final}")
    with tempfile.TemporaryDirectory(prefix=f".step5b_{plan.video_id}_", dir=parent) as temporary:
        stage = Path(temporary)
        _write_csv(stage / "sequence.csv", FRAME_COLUMNS, plan.rows)
        np.savez_compressed(stage / "behavior.npz", **_bundle_arrays(plan))
        _write_json(stage / "summary.json", {
            "video_id": plan.video_id, "split": plan.split, "label": plan.label,
            "sequence_length": 100, "behavior_valid_count": int(plan.behavior_valid_mask.sum()),
            "behavior_missing_count": int((~plan.behavior_valid_mask).sum()),
            "longest_yunet_missing_run": plan.longest_yunet_missing_run,
            "feature_names": list(FEATURE_NAMES), "feature_shape": [100, len(FEATURE_NAMES)],
            "feature_schema": FEATURE_SCHEMA, "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
            "source_metadata_fingerprint": plan.fingerprint,
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        })
        _write_json(stage / "COMPLETE.json", {
            "video_id": plan.video_id, "source_metadata_fingerprint": plan.fingerprint,
            "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
            "canonical_policy_hash": POLICY_HASH,
            "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
            "file_sha256": {name: file_sha256(stage / name)
                            for name in ("sequence.csv", "behavior.npz", "summary.json")},
        })
        validate_bundle(stage, plan)
        stage.rename(final)


def _feature_summary(plans: Sequence[BehaviorPlan]) -> list[dict[str, Any]]:
    matrix = np.concatenate([plan.features for plan in plans])
    rows = []
    for column, name in enumerate(FEATURE_NAMES):
        values = matrix[np.isfinite(matrix[:, column]), column].astype(np.float64)
        rows.append({"feature": name, "count": len(values), "mean": float(values.mean()),
                     "std": float(values.std()), "min": float(values.min()),
                     "p01": float(np.percentile(values, 1)), "p05": float(np.percentile(values, 5)),
                     "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                     "p99": float(np.percentile(values, 99)), "max": float(values.max())})
    return rows


def materialize(plans: Sequence[BehaviorPlan], excluded: Sequence[Mapping[str, Any]],
                root: Path, report_root: Path, resume: bool = False,
                expected: Mapping[str, int] = EXPECTED) -> dict[str, Any]:
    """Behavior CSV/NPZ/출처만 원자적으로 쓰며 보간·이미지·CNN은 사용하지 않는다."""
    counts = plan_statistics(plans, excluded, expected)
    if root.exists() and not resume:
        raise FileExistsError(f"STEP_5B_REVIEW_REQUIRED: output exists {root}")
    if report_root.exists() and not resume:
        raise FileExistsError(f"STEP_5B_REVIEW_REQUIRED: report exists {report_root}")
    created = skipped = 0
    for plan in plans:
        validate_video_plan(plan)
        bundle = root / "per_video" / plan.split / plan.video_id
        if bundle.exists():
            if not resume:
                raise FileExistsError(f"STEP_5B_REVIEW_REQUIRED: bundle exists {bundle}")
            validate_bundle(bundle, plan)
            skipped += 1
        else:
            publish_bundle(root, plan)
            created += 1
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    frame_rows = [row for plan in plans for row in plan.rows]
    video_rows = [{
        "video_id": plan.video_id, "split": plan.split, "label": plan.label,
        "sequence_length": 100, "behavior_valid_count": int(plan.behavior_valid_mask.sum()),
        "behavior_missing_count": int((~plan.behavior_valid_mask).sum()),
        "longest_yunet_missing_run": plan.longest_yunet_missing_run,
        "context_eligible": plan.context_eligible, "behavior_eligible": True,
        "source_metadata_fingerprint": plan.fingerprint,
        "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
        "canonical_policy_hash": POLICY_HASH, "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
    } for plan in plans]
    valid_hist = Counter(int(plan.behavior_valid_mask.sum()) for plan in plans)
    run_hist = Counter(plan.longest_yunet_missing_run for plan in plans)
    histogram = ([{"metric": "valid_count", "value": value, "videos": valid_hist[value],
                   "fraction_of_eligible_videos": valid_hist[value] / len(plans)} for value in range(95, 101)]
                 + [{"metric": "longest_missing_run", "value": value, "videos": run_hist[value],
                     "fraction_of_eligible_videos": run_hist[value] / len(plans)} for value in range(6)])
    groups = []
    for group_type in ("split", "label", "split_x_label"):
        key = lambda plan: (plan.split if group_type == "split" else plan.label if group_type == "label"
                            else f"{plan.split}/{plan.label}")
        for name in sorted({key(plan) for plan in plans}):
            members = [plan for plan in plans if key(plan) == name]
            valid = sum(int(plan.behavior_valid_mask.sum()) for plan in members)
            groups.append({"group_type": group_type, "group_name": name, "videos": len(members),
                           "behavior_valid_slots": valid,
                           "behavior_missing_slots": 100 * len(members) - valid})
    summary = {**counts, "status": "STEP_5B_BEHAVIOR_SEQUENCE_CONSTRUCTION_COMPLETE",
               "sequence_length": 100, "feature_names": list(FEATURE_NAMES),
               "feature_shape_per_video": [100, len(FEATURE_NAMES)],
               "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
               "canonical_policy_hash": POLICY_HASH,
               "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
               "behavior_policy_id": BEHAVIOR_POLICY_ID,
               "behavior_handling": "NaN + valid masks", "interpolation_count": 0,
               "valid_count_mismatch": 0, "missing_run_mismatch": 0,
               "required_valid_nan_contradiction": 0, "infinite_value_count": 0,
               "canonical_numeric_mismatch": 0, "test_rows": 0, "test_access": 0,
               "valid_count_histogram": histogram[:6], "longest_run_histogram": histogram[6:],
               "split_label_summary": groups, "bundles_created_this_run": created,
               "bundles_skipped_this_run": skipped, "raw_video_opened": False,
               "context_sequence_accessed": False, "canonical_data_modified": False,
               "context_sequence_modified": False, "cnn_executed": False}
    csv_outputs = (
        (metadata / "behavior_sequence_frames.csv", FRAME_COLUMNS, frame_rows),
        (metadata / "behavior_sequence_videos.csv", VIDEO_COLUMNS, video_rows),
        (metadata / "behavior_sequence_excluded_videos.csv", EXCLUDED_COLUMNS, excluded),
        (report_root / "excluded_behavior_videos.csv", EXCLUDED_COLUMNS, excluded),
        (report_root / "behavior_sequence_validity_histogram.csv", HISTOGRAM_COLUMNS, histogram),
        (report_root / "behavior_sequence_split_label_summary.csv", GROUP_COLUMNS, groups),
        (report_root / "behavior_sequence_feature_summary.csv", FEATURE_SUMMARY_COLUMNS,
         _feature_summary(plans)),
    )
    for path, columns, rows in csv_outputs:
        if path.exists() and resume:
            if _read_csv(path) != [{key: str(value) for key, value in row.items()} for row in rows]:
                raise ValueError(f"STEP_5B_REVIEW_REQUIRED: existing CSV conflict {path}")
        else:
            if path.exists():
                raise FileExistsError(f"STEP_5B_REVIEW_REQUIRED: overwrite forbidden {path}")
            _write_csv(path, columns, rows)
    run_metadata = {"status": summary["status"], "feature_schema": FEATURE_SCHEMA,
                    "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
                    "canonical_policy_hash": POLICY_HASH,
                    "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
                    "eligible_videos": len(plans), "sequence_rows": len(frame_rows),
                    "test_sealed": True,
                    "source_metadata_fingerprints": stable_hash(
                        {plan.video_id: plan.fingerprint for plan in plans})}
    stable_summary = {key: value for key, value in summary.items()
                      if key not in {"bundles_created_this_run", "bundles_skipped_this_run"}}
    for path, value in ((metadata / "run_metadata.json", run_metadata),
                        (report_root / "behavior_sequence_summary.json", stable_summary)):
        if path.exists() and resume:
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError(f"STEP_5B_REVIEW_REQUIRED: existing JSON conflict {path}")
        else:
            if path.exists():
                raise FileExistsError(f"STEP_5B_REVIEW_REQUIRED: overwrite forbidden {path}")
            _write_json(path, value)
    reports = {
        "behavior_sequence_generation_report.txt":
            f"STEP 5-B COMPLETE\nBundles: {len(plans)}\nRows: {len(frame_rows)}\nInterpolation: 0\nCNN: NOT RUN\n",
        "behavior_sequence_integrity_report.txt":
            "STEP 5-B INTEGRITY: PASS\n100 slots/video; NaN, masks, numeric provenance, hashes: PASS\nTEST: SEALED\n",
    }
    for name, content in reports.items():
        path = report_root / name
        if path.exists() and resume:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"STEP_5B_REVIEW_REQUIRED: existing report conflict {path}")
        else:
            path.write_text(content, encoding="utf-8")
    (report_root / "resume_report.txt").write_text(
        f"resume={resume}\ncreated={created}\nskipped_verified={skipped}\nconflicts=0\n", encoding="utf-8")
    return summary
