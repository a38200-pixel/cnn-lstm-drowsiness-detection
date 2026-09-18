"""STEP 2-A의 동일 frame에서 HOG와 YuNet primary 후보를 직접 비교한다."""

from __future__ import annotations

import csv
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .angle_utils import circular_angle_difference, pitch_centered_candidate
from .audit_visualization import annotate_frame, create_contact_sheets
from .crop_preview import CropPreview, create_face_crop_preview, crop_mean_absolute_difference
from .detectors import BoundingBox, DetectorResult, HogFaceDetector, YuNetFaceDetector, bbox_iou
from .geometric_features import RatioResult, calculate_ear, calculate_mar, validate_geometry
from .head_pose import HeadPoseResult, estimate_head_pose
from .landmarks import Dlib68Predictor, LandmarkResult


DECISION = "WAITING_FOR_MANUAL_PRIMARY_DETECTOR_REVIEW"
RESULT_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec", "source_frame_index", "source_fps",
    "hog_success", "yunet_success", "hog_detection_count", "yunet_detection_count", "yunet_confidence",
    "hog_bbox_x1", "hog_bbox_y1", "hog_bbox_x2", "hog_bbox_y2",
    "yunet_bbox_x1", "yunet_bbox_y1", "yunet_bbox_x2", "yunet_bbox_y2", "bbox_iou",
    "hog_landmark_success", "yunet_landmark_success", "landmark_mean_diff", "normalized_landmark_diff",
    "ear_hog", "ear_yunet", "ear_abs_diff", "mar_hog", "mar_yunet", "mar_abs_diff",
    "pitch_raw_hog", "pitch_raw_yunet", "pitch_centered_hog", "pitch_centered_yunet",
    "pitch_raw_abs_diff", "pitch_circular_diff", "yaw_hog", "yaw_yunet",
    "yaw_raw_abs_diff", "yaw_circular_diff", "roll_hog", "roll_yunet",
    "roll_raw_abs_diff", "roll_circular_diff", "hog_ms", "yunet_ms",
    "hog_landmark_ms", "yunet_landmark_ms", "hog_pipeline_ms", "yunet_pipeline_ms",
    "hog_crop_padding_fraction", "yunet_crop_padding_fraction", "crop_mean_abs_diff",
    "hog_geometry_valid", "yunet_geometry_valid", "quality_flags", "failure_reason",
)
MANUAL_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec", "category", "comparison_image_path",
    "hog_bbox_ok", "yunet_bbox_ok", "hog_full_face_crop_ok", "yunet_full_face_crop_ok",
    "hog_landmarks_ok", "yunet_landmarks_ok", "hog_eyes_ok", "yunet_eyes_ok",
    "hog_mouth_ok", "yunet_mouth_ok", "hog_pose_axis_ok", "yunet_pose_axis_ok",
    "preferred_detector", "manual_decision", "review_note",
)


@dataclass(frozen=True)
class ComparisonConfig:
    """프로젝트 기준으로 해석된 detector primary 비교 설정."""

    project_root: Path
    config_path: Path
    values: dict[str, Any]
    step2a_values: dict[str, Any]

    def path(self, key: str) -> Path:
        """최상위 경로 설정을 절대 경로로 변환한다."""

        value = Path(self.values[key])
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()


@dataclass(frozen=True)
class SourceSample:
    """STEP 2-A에서 실제 처리된 동일 frame을 가리키는 정보."""

    sample_id: str
    video_id: str
    split: str
    label: str
    timestamp_sec: float
    source_frame_index: int
    source_fps: float
    video_path: Path


@dataclass(frozen=True)
class PathwayResult:
    """하나의 detector bbox에 동일한 Dlib68/feature 처리를 적용한 결과."""

    detector: DetectorResult
    landmark: LandmarkResult
    ear: RatioResult
    mar: RatioResult
    pose: HeadPoseResult
    geometry_valid: bool
    feature_ms: float
    pipeline_ms: float
    crop: CropPreview | None
    failure_reason: str


def load_comparison_config(config_path: Path, project_root: Path) -> ComparisonConfig:
    """STEP 2-B와 STEP 2-A config를 읽고 동일 정책 사용 여부를 확인한다."""

    resolved = config_path.resolve() if config_path.is_absolute() else (project_root / config_path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("comparison config는 mapping이어야 합니다")
    required = {
        "step2a_results_dir", "step2a_config", "train_metadata", "val_metadata",
        "dlib_predictor", "yunet_model", "hog_upsample", "face_crop_preview",
        "visual_audit", "comparison_priority", "audit_large_pose_deg", "output_dir",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"comparison config 필수 항목 누락: {missing}")
    if "test.csv" in f"{values['train_metadata']} {values['val_metadata']}".casefold():
        raise ValueError("STEP 2-B에서는 test.csv를 사용할 수 없습니다")
    step2a_path = Path(values["step2a_config"])
    if not step2a_path.is_absolute():
        step2a_path = project_root / step2a_path
    with step2a_path.open("r", encoding="utf-8") as handle:
        step2a_values = yaml.safe_load(handle)
    if values["hog_upsample"] != step2a_values["hog_upsample"]:
        raise ValueError("HOG upsample은 STEP 2-A와 같아야 합니다")
    return ComparisonConfig(project_root.resolve(), resolved, values, step2a_values)


def _load_video_paths(config: ComparisonConfig) -> dict[tuple[str, str], Path]:
    """train/val metadata에서 video ID별 원본 경로를 읽고 test 유입을 차단한다."""

    result: dict[tuple[str, str], Path] = {}
    for split, key in (("train", "train_metadata"), ("val", "val_metadata")):
        with config.path(key).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                declared = row.get("split", split)
                if declared and declared != split:
                    raise ValueError(f"{key}에 예상하지 않은 split이 있습니다: {declared}")
                path = Path(row["video_path"])
                result[(split, row["video_id"])] = path.resolve() if path.is_absolute() else (config.project_root / path).resolve()
    return result


def load_step2a_samples(config: ComparisonConfig) -> list[SourceSample]:
    """STEP 2-A sample_results.csv의 frame index를 그대로 재사용한다."""

    source_path = config.path("step2a_results_dir") / "sample_results.csv"
    video_paths = _load_video_paths(config)
    samples: list[SourceSample] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in {"train", "val"}:
                raise ValueError(f"STEP 2-A artifact에 금지된 split이 있습니다: {split}")
            key = (split, row["video_id"])
            if key not in video_paths:
                raise ValueError(f"metadata에서 영상을 찾을 수 없습니다: {key}")
            samples.append(SourceSample(
                row["sample_id"], row["video_id"], split, row["label"],
                float(row["timestamp_sec"]), int(row["source_frame_index"]),
                float(row["source_fps"]), video_paths[key],
            ))
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("STEP 2-A sample ID가 중복됩니다")
    return samples


def describe_comparison_dry_run(config: ComparisonConfig) -> dict[str, Any]:
    """영상 없이 STEP 2-A 표본 재사용과 출력 격리를 검증한다."""

    samples = load_step2a_samples(config)
    return {
        "mode": "DRY_RUN", "source_sample_count": len(samples),
        "unique_video_count": len({sample.video_id for sample in samples}),
        "split_counts": dict(Counter(sample.split for sample in samples)),
        "label_counts": dict(Counter(sample.label for sample in samples)),
        "test_split_used": False, "new_random_sampling": False,
        "source_results_dir": str(config.path("step2a_results_dir")),
        "output_dir": str(config.path("output_dir")), "decision": DECISION,
    }


def _read_exact_frame(sample: SourceSample) -> tuple[np.ndarray | None, str]:
    """timestamp를 다시 계산하지 않고 STEP 2-A의 source frame index를 직접 읽는다."""

    capture = cv2.VideoCapture(str(sample.video_path))
    try:
        if not capture.isOpened():
            return None, "VIDEO_OPEN_FAILED"
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample.source_frame_index)
        success, frame = capture.read()
        return (frame, "") if success and frame is not None else (None, "FRAME_DECODE_FAILED")
    finally:
        capture.release()


def _failed_pathway(detector: DetectorResult) -> PathwayResult:
    """detector 실패를 이후 공통 구조로 변환한다."""

    landmark = LandmarkResult(False, None, 0.0, "DETECTOR_REQUIRED")
    ratio = RatioResult(None, False, "DETECTOR_REQUIRED")
    pose = HeadPoseResult(False, None, None, None, None, 0.0, "DETECTOR_REQUIRED")
    return PathwayResult(detector, landmark, ratio, ratio, pose, False, 0.0, detector.elapsed_ms, None, detector.failure_reason)


def _run_pathway(
    frame: np.ndarray,
    detector_result: DetectorResult,
    predictor: Dlib68Predictor,
    config: ComparisonConfig,
) -> PathwayResult:
    """detector 결과에 STEP 2-A와 동일한 Dlib68, EAR/MAR, pose 정책을 적용한다."""

    if not detector_result.success or detector_result.selected is None:
        return _failed_pathway(detector_result)
    bbox = detector_result.selected.bbox
    landmark = predictor.predict(frame, bbox)
    if not landmark.success or landmark.points is None:
        failed = _failed_pathway(detector_result)
        return PathwayResult(detector_result, landmark, failed.ear, failed.mar, failed.pose, False, 0.0, detector_result.elapsed_ms + landmark.elapsed_ms, None, landmark.failure_reason)
    feature_started = perf_counter()
    geometry_config = config.step2a_values["geometry_validation"]
    _left, _right, ear = calculate_ear(landmark.points, float(geometry_config["denominator_epsilon"]))
    mar = calculate_mar(landmark.points, float(geometry_config["denominator_epsilon"]))
    feature_ms = (perf_counter() - feature_started) * 1000.0
    pose = estimate_head_pose(landmark.points, frame.shape[1], frame.shape[0])
    geometry = validate_geometry(bbox, landmark.points, frame.shape, ear, mar, pose.success, geometry_config)
    crop = None
    crop_config = config.values["face_crop_preview"]
    if bool(crop_config["enabled"]):
        crop = create_face_crop_preview(
            frame, bbox, float(crop_config["margin_ratio"]), bool(crop_config["square_crop"]),
            int(crop_config["output_size"]), int(crop_config["padding_value"]),
        )
    pipeline_ms = detector_result.elapsed_ms + landmark.elapsed_ms + feature_ms + pose.elapsed_ms
    failures = [value for value in (landmark.failure_reason, ear.failure_reason, mar.failure_reason, pose.failure_reason) if value]
    return PathwayResult(detector_result, landmark, ear, mar, pose, geometry.geometry_valid, feature_ms, pipeline_ms, crop, ";".join(failures))


def _finite_difference(left: float | None, right: float | None) -> float | None:
    """두 유효 숫자가 모두 있을 때만 절댓값 차이를 반환한다."""

    return abs(float(left) - float(right)) if left is not None and right is not None else None


def _pose_values(pathway: PathwayResult) -> tuple[float | None, float | None, float | None, float | None]:
    """raw pitch/yaw/roll과 front-centered pitch 후보를 반환한다."""

    pose = pathway.pose
    centered = pitch_centered_candidate(pose.pitch) if pose.success and pose.pitch is not None else None
    return pose.pitch, centered, pose.yaw, pose.roll


def _bbox_fields(prefix: str, bbox: BoundingBox | None) -> dict[str, int | None]:
    """bbox를 CSV의 네 좌표 컬럼으로 펼친다."""

    values = bbox.as_tuple() if bbox else (None, None, None, None)
    return dict(zip((f"{prefix}_bbox_x1", f"{prefix}_bbox_y1", f"{prefix}_bbox_x2", f"{prefix}_bbox_y2"), values))


def _compare_sample(
    sample: SourceSample, frame: np.ndarray, hog: HogFaceDetector,
    yunet: YuNetFaceDetector, predictor: Dlib68Predictor, config: ComparisonConfig,
) -> tuple[dict[str, Any], PathwayResult, PathwayResult]:
    """동일 frame에 두 detector를 모두 실행하고 공통 feature 차이를 계산한다."""

    hog_path = _run_pathway(frame, hog.detect(frame), predictor, config)
    yunet_path = _run_pathway(frame, yunet.detect(frame), predictor, config)
    hog_bbox = hog_path.detector.selected.bbox if hog_path.detector.selected else None
    yunet_bbox = yunet_path.detector.selected.bbox if yunet_path.detector.selected else None
    both_landmarks = hog_path.landmark.points is not None and yunet_path.landmark.points is not None
    landmark_mean = normalized_landmark = None
    if both_landmarks:
        landmark_mean = float(np.mean(np.linalg.norm(hog_path.landmark.points - yunet_path.landmark.points, axis=1)))
        normalized_landmark = landmark_mean / max(float(np.sqrt(hog_bbox.area)), 1.0)  # type: ignore[union-attr]
    pitch_hog, centered_hog, yaw_hog, roll_hog = _pose_values(hog_path)
    pitch_yunet, centered_yunet, yaw_yunet, roll_yunet = _pose_values(yunet_path)
    crop_diff = None
    if hog_path.crop is not None and yunet_path.crop is not None:
        crop_diff = crop_mean_absolute_difference(hog_path.crop.rgb, yunet_path.crop.rgb)
    flags: list[str] = []
    large_limit = float(config.values["audit_large_pose_deg"])
    if centered_hog is not None and abs(centered_hog) > large_limit:
        flags.append("HOG_AUDIT_LARGE_POSE")
    if centered_yunet is not None and abs(centered_yunet) > large_limit:
        flags.append("YUNET_AUDIT_LARGE_POSE")
    record: dict[str, Any] = {
        "sample_id": sample.sample_id, "video_id": sample.video_id, "split": sample.split,
        "label": sample.label, "timestamp_sec": sample.timestamp_sec,
        "source_frame_index": sample.source_frame_index, "source_fps": sample.source_fps,
        "hog_success": hog_path.detector.success, "yunet_success": yunet_path.detector.success,
        "hog_detection_count": hog_path.detector.detection_count,
        "yunet_detection_count": yunet_path.detector.detection_count,
        "yunet_confidence": yunet_path.detector.selected.confidence if yunet_path.detector.selected else None,
        **_bbox_fields("hog", hog_bbox), **_bbox_fields("yunet", yunet_bbox),
        "bbox_iou": bbox_iou(hog_bbox, yunet_bbox) if hog_bbox and yunet_bbox else None,
        "hog_landmark_success": hog_path.landmark.success,
        "yunet_landmark_success": yunet_path.landmark.success,
        "landmark_mean_diff": landmark_mean, "normalized_landmark_diff": normalized_landmark,
        "ear_hog": hog_path.ear.value, "ear_yunet": yunet_path.ear.value,
        "ear_abs_diff": _finite_difference(hog_path.ear.value, yunet_path.ear.value),
        "mar_hog": hog_path.mar.value, "mar_yunet": yunet_path.mar.value,
        "mar_abs_diff": _finite_difference(hog_path.mar.value, yunet_path.mar.value),
        "pitch_raw_hog": pitch_hog, "pitch_raw_yunet": pitch_yunet,
        "pitch_centered_hog": centered_hog, "pitch_centered_yunet": centered_yunet,
        "pitch_raw_abs_diff": _finite_difference(pitch_hog, pitch_yunet),
        "pitch_circular_diff": circular_angle_difference(pitch_hog, pitch_yunet) if pitch_hog is not None and pitch_yunet is not None else None,
        "yaw_hog": yaw_hog, "yaw_yunet": yaw_yunet,
        "yaw_raw_abs_diff": _finite_difference(yaw_hog, yaw_yunet),
        "yaw_circular_diff": circular_angle_difference(yaw_hog, yaw_yunet) if yaw_hog is not None and yaw_yunet is not None else None,
        "roll_hog": roll_hog, "roll_yunet": roll_yunet,
        "roll_raw_abs_diff": _finite_difference(roll_hog, roll_yunet),
        "roll_circular_diff": circular_angle_difference(roll_hog, roll_yunet) if roll_hog is not None and roll_yunet is not None else None,
        "hog_ms": hog_path.detector.elapsed_ms, "yunet_ms": yunet_path.detector.elapsed_ms,
        "hog_landmark_ms": hog_path.landmark.elapsed_ms, "yunet_landmark_ms": yunet_path.landmark.elapsed_ms,
        "hog_pipeline_ms": hog_path.pipeline_ms, "yunet_pipeline_ms": yunet_path.pipeline_ms,
        "hog_crop_padding_fraction": hog_path.crop.padding_fraction if hog_path.crop else None,
        "yunet_crop_padding_fraction": yunet_path.crop.padding_fraction if yunet_path.crop else None,
        "crop_mean_abs_diff": crop_diff, "hog_geometry_valid": hog_path.geometry_valid,
        "yunet_geometry_valid": yunet_path.geometry_valid, "quality_flags": ";".join(flags),
        "failure_reason": ";".join(value for value in (hog_path.failure_reason, yunet_path.failure_reason) if value),
    }
    return record, hog_path, yunet_path


def _distribution(values: Sequence[Any]) -> dict[str, Any]:
    """유효 숫자의 평균, 중앙값, 범위, P90/P95를 계산한다."""

    array = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))])
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None, "p95": None}
    return {
        "count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "min": float(np.min(array)), "max": float(np.max(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """동점에 평균 순위를 부여해 Spearman 계산에 사용할 rank를 만든다."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def _correlation(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    """외부 통계 패키지 없이 Pearson 및 Spearman correlation을 계산한다."""

    pairs = [(float(a), float(b)) for a, b in zip(left, right) if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b))]
    if len(pairs) < 2:
        return {"count": len(pairs), "pearson": None, "spearman": None}
    first, second = np.asarray(pairs, dtype=np.float64).T
    pearson = float(np.corrcoef(first, second)[0, 1]) if np.std(first) > 0 and np.std(second) > 0 else None
    ranked_first, ranked_second = _rank(first), _rank(second)
    spearman = float(np.corrcoef(ranked_first, ranked_second)[0, 1]) if np.std(ranked_first) > 0 and np.std(ranked_second) > 0 else None
    return {"count": len(pairs), "pearson": pearson, "spearman": spearman}


def _outcome(record: Mapping[str, Any]) -> str:
    """두 detector 성공 조합을 네 가지 outcome으로 분류한다."""

    if record["hog_success"] and record["yunet_success"]:
        return "both_success"
    if record["hog_success"]:
        return "hog_only"
    if record["yunet_success"]:
        return "yunet_only"
    return "both_failed"


def _summary(records: Sequence[Mapping[str, Any]], config: ComparisonConfig, visual_counts: Mapping[str, int]) -> dict[str, Any]:
    """검출, landmark, feature, pose, crop, latency 결과를 집계한다."""

    total = len(records)
    outcomes = Counter(_outcome(record) for record in records)

    def detector_summary(prefix: str) -> dict[str, Any]:
        success = sum(bool(row[f"{prefix}_success"]) for row in records)
        return {
            "attempted": total, "success": success, "failure": total - success,
            "success_rate": success / total if total else 0.0,
            "multiple_face_count": sum(int(row[f"{prefix}_detection_count"] or 0) > 1 for row in records),
            "latency_ms": _distribution([row[f"{prefix}_ms"] for row in records]),
        }

    both = [row for row in records if row["hog_landmark_success"] and row["yunet_landmark_success"]]
    feature_sections: dict[str, Any] = {}
    for name in ("ear", "mar"):
        feature_sections[name] = {
            "pair_count": len(both),
            "hog_distribution": _distribution([row[f"{name}_hog"] for row in both]),
            "yunet_distribution": _distribution([row[f"{name}_yunet"] for row in both]),
            "absolute_difference": _distribution([row[f"{name}_abs_diff"] for row in both]),
            "correlation": _correlation([row[f"{name}_hog"] for row in both], [row[f"{name}_yunet"] for row in both]),
        }
    head_pose = {
        "pitch_raw_hog": _distribution([row["pitch_raw_hog"] for row in records]),
        "pitch_raw_yunet": _distribution([row["pitch_raw_yunet"] for row in records]),
        "pitch_centered_hog": _distribution([row["pitch_centered_hog"] for row in records]),
        "pitch_centered_yunet": _distribution([row["pitch_centered_yunet"] for row in records]),
        "pitch_raw_abs_diff": _distribution([row["pitch_raw_abs_diff"] for row in both]),
        "pitch_circular_diff": _distribution([row["pitch_circular_diff"] for row in both]),
        "yaw_raw_abs_diff": _distribution([row["yaw_raw_abs_diff"] for row in both]),
        "yaw_circular_diff": _distribution([row["yaw_circular_diff"] for row in both]),
        "roll_raw_abs_diff": _distribution([row["roll_raw_abs_diff"] for row in both]),
        "roll_circular_diff": _distribution([row["roll_circular_diff"] for row in both]),
        "pitch_centered_candidate_note": "정면≈0 분석 후보이며 부호와 head up/down 관계는 미확정",
    }
    return {
        "decision": DECISION,
        "scope": {
            "source": "STEP 2-A sample_results.csv", "frame_count": total,
            "unique_video_count": len({row["video_id"] for row in records}),
            "split_counts": dict(Counter(str(row["split"]) for row in records)),
            "label_counts": dict(Counter(str(row["label"]) for row in records)),
            "test_split_used": False, "new_random_sampling": False,
            "label_usage": "reporting_only",
        },
        "detector_detection": {
            "hog": detector_summary("hog"),
            "yunet": {**detector_summary("yunet"), "confidence": _distribution([row["yunet_confidence"] for row in records])},
        },
        "outcome_matrix": {key: {"count": outcomes[key], "ratio": outcomes[key] / total if total else 0.0} for key in ("both_success", "hog_only", "yunet_only", "both_failed")},
        "landmark": {
            "hog": {
                "attempted": sum(bool(row["hog_success"]) for row in records),
                "success": sum(bool(row["hog_landmark_success"]) for row in records),
                "failure": sum(bool(row["hog_success"]) for row in records) - sum(bool(row["hog_landmark_success"]) for row in records),
                "success_rate": (
                    sum(bool(row["hog_landmark_success"]) for row in records)
                    / sum(bool(row["hog_success"]) for row in records)
                    if any(bool(row["hog_success"]) for row in records) else 0.0
                ),
                "latency_ms": _distribution([row["hog_landmark_ms"] for row in records if row["hog_success"]]),
            },
            "yunet": {
                "attempted": sum(bool(row["yunet_success"]) for row in records),
                "success": sum(bool(row["yunet_landmark_success"]) for row in records),
                "failure": sum(bool(row["yunet_success"]) for row in records) - sum(bool(row["yunet_landmark_success"]) for row in records),
                "success_rate": (
                    sum(bool(row["yunet_landmark_success"]) for row in records)
                    / sum(bool(row["yunet_success"]) for row in records)
                    if any(bool(row["yunet_success"]) for row in records) else 0.0
                ),
                "latency_ms": _distribution([row["yunet_landmark_ms"] for row in records if row["yunet_success"]]),
            },
            "mean_difference": _distribution([row["landmark_mean_diff"] for row in both]),
            "normalized_difference": _distribution([row["normalized_landmark_diff"] for row in both]),
        },
        "ear": feature_sections["ear"], "mar": feature_sections["mar"], "head_pose": head_pose,
        "full_face_crop": {
            "preview_pair_count": sum(row["crop_mean_abs_diff"] is not None for row in records),
            "suspicious_padding_count": sum(
                float(row["hog_crop_padding_fraction"] or 0.0) > 0.0
                or float(row["yunet_crop_padding_fraction"] or 0.0) > 0.0
                for row in records
            ),
            "mean_absolute_difference": _distribution([row["crop_mean_abs_diff"] for row in records]),
            "policy": config.values["face_crop_preview"], "training_data_created": False,
        },
        "latency": {
            "hog_detector_ms": _distribution([row["hog_ms"] for row in records]),
            "yunet_detector_ms": _distribution([row["yunet_ms"] for row in records]),
            "hog_behavior_pipeline_ms": _distribution([row["hog_pipeline_ms"] for row in records]),
            "yunet_behavior_pipeline_ms": _distribution([row["yunet_pipeline_ms"] for row in records]),
            "reference_10fps_budget_ms": 100.0,
            "budget_note": "참고값이며 detector 자동 승인 hard rule이 아님",
        },
        "visual_review": {
            "category_counts": dict(visual_counts), "manual_review_csv": "manual_review.csv",
            "contact_sheet_dir": "contact_sheets",
        },
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "os": platform.platform()},
        "limitations": [
            "video-level split이며 subject-wise independence를 보장하지 않음",
            "pitch centered 표현과 부호 convention은 visual review 전 후보 상태",
            "generic camera model의 pose는 ground-truth angle이 아님",
            "crop margin과 크기는 audit preview 후보이며 학습 정책이 아님",
            "수동 paired review 전에는 primary detector를 승인하지 않음",
        ],
    }


def _annotated_panel(frame: np.ndarray, sample: SourceSample, pathway: PathwayResult, detector_name: str) -> np.ndarray:
    """detector별 bbox, landmark, axis, raw/centered pose와 crop을 한 panel에 표시한다."""

    bbox = pathway.detector.selected.bbox if pathway.detector.selected else None
    pitch = pathway.pose.pitch
    values = {
        "sample_id": detector_name, "split": sample.split, "label": sample.label,
        "timestamp_sec": sample.timestamp_sec, "detector_source": detector_name,
        "ear_mean": pathway.ear.value, "mar": pathway.mar.value,
        "pitch": pitch, "yaw": pathway.pose.yaw, "roll": pathway.pose.roll,
    }
    panel = annotate_frame(frame, bbox, pathway.landmark.points, pathway.pose.axis_points, values)
    if pitch is not None:
        text = f"pitch centered candidate={pitch_centered_candidate(pitch):.2f}"
        cv2.putText(panel, text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if pathway.crop is not None:
        crop_bgr = cv2.cvtColor(pathway.crop.rgb, cv2.COLOR_RGB2BGR)
        size = min(180, panel.shape[0] // 3)
        crop_bgr = cv2.resize(crop_bgr, (size, size))
        panel[-size:, -size:] = crop_bgr
        cv2.rectangle(panel, (panel.shape[1] - size, panel.shape[0] - size), (panel.shape[1] - 1, panel.shape[0] - 1), (255, 255, 255), 2)
    return cv2.resize(panel, (480, 360), interpolation=cv2.INTER_AREA)


def _comparison_image(frame: np.ndarray, sample: SourceSample, hog: PathwayResult, yunet: PathwayResult) -> np.ndarray:
    """Original, HOG+Dlib68, YuNet+Dlib68을 가로로 배치한다."""

    original = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(original, f"ORIGINAL {sample.video_id} t={sample.timestamp_sec}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack((original, _annotated_panel(frame, sample, hog, "HOG"), _annotated_panel(frame, sample, yunet, "YUNET")))


def _visual_categories(record: Mapping[str, Any], config: ComparisonConfig) -> list[tuple[str, str]]:
    """outcome과 audit 차이 기준에 따라 comparison image category를 선택한다."""

    outcome = _outcome(record)
    if outcome == "hog_only" or outcome == "yunet_only":
        categories = [(outcome, "max_detector_only")]
    elif outcome == "both_failed":
        categories = [(outcome, "max_both_failed")]
    else:
        categories = [("both_success", "max_normal_pairs")]
    limits = config.values["comparison_priority"]
    checks = (
        ("large_bbox_difference", "max_bbox_difference", record["bbox_iou"], lambda value: value < float(limits["bbox_iou_below"])),
        ("large_landmark_difference", "max_landmark_difference", record["normalized_landmark_diff"], lambda value: value > float(limits["normalized_landmark_diff_above"])),
        ("large_ear_difference", "max_ear_difference", record["ear_abs_diff"], lambda value: value > float(limits["ear_abs_diff_above"])),
        ("large_mar_difference", "max_mar_difference", record["mar_abs_diff"], lambda value: value > float(limits["mar_abs_diff_above"])),
        ("large_pose_difference", "max_pose_difference", record["pitch_circular_diff"], lambda value: value > float(limits["pose_circular_diff_above"])),
        ("full_face_crop_difference", "max_crop_difference", record["crop_mean_abs_diff"], lambda value: value > float(limits["crop_mean_abs_diff_above"])),
    )
    categories.extend((category, cap) for category, cap, value, predicate in checks if value is not None and predicate(float(value)))
    return categories


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """고정 컬럼 순서의 UTF-8 CSV를 기록한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary: Mapping[str, Any]) -> str:
    """STEP 2-B 필수 section과 결정 보류 상태를 report로 변환한다."""

    sections = (
        ("Scope", summary["scope"]), ("Detector Detection", summary["detector_detection"]),
        ("Outcome Matrix", summary["outcome_matrix"]), ("Landmark", summary["landmark"]),
        ("EAR", summary["ear"]), ("MAR", summary["mar"]), ("Head Pose", summary["head_pose"]),
        ("Full Face Crop", summary["full_face_crop"]), ("Latency", summary["latency"]),
        ("Real-Time Context", {"reference": "10 FPS ≈ 100 ms/frame; 자동 채택 기준이 아님"}),
        ("Visual Review", summary["visual_review"]), ("Decision", {"decision": summary["decision"]}),
        ("Limitations", summary["limitations"]),
    )
    lines = [
        "Experiment 2 STEP 2-B — HOG vs YuNet Primary Detector Comparison",
        "동일한 STEP 2-A frame과 동일한 Dlib68/feature/pose 정책을 사용함.",
        "pitch_centered_candidate는 분석 후보이며 sign convention은 미확정.", "",
    ]
    for title, value in sections:
        lines.extend((f"[{title}]", json.dumps(value, ensure_ascii=False, indent=2), ""))
    return "\n".join(lines)


def _write_manual_guide(path: Path) -> None:
    """HOG와 YuNet을 사람이 직접 비교하는 한국어 검토 기준을 기록한다."""

    path.write_text(
        """# STEP 2-B Primary Detector 수동 비교 가이드

자동 비교의 Decision은 `WAITING_FOR_MANUAL_PRIMARY_DETECTOR_REVIEW`입니다. 더 빠르거나 검출 성공률이 높다는 이유만으로 detector를 승인하지 않습니다.

## 1. 얼굴 검출

- 각 detector가 실제 운전자 얼굴을 놓치지 않는지 확인합니다.
- 여러 얼굴 중 배경 인물이나 잘못된 영역을 선택하지 않았는지 확인합니다.

## 2. Full Face crop

- 이마, 턱, 양쪽 눈, 입이 유지되는지 확인합니다.
- 얼굴 한쪽이 과도하게 잘리거나 배경이 지나치게 포함되지 않는지 확인합니다.
- 25% margin, square, 224 크기는 audit preview 후보이며 최종 학습 정책이 아닙니다.

## 3. Dlib68 landmark

- 눈, 입, 코, 턱 좌표가 어느 detector bbox에서 더 자연스러운지 비교합니다.
- 안경테, 눈썹, 배경에 landmark가 붙지 않았는지 확인합니다.

## 4. EAR/MAR

- 실제 눈과 입 상태가 같은데 detector 경로에 따라 값이 과도하게 달라지는지 확인합니다.
- 이 값으로 졸음이나 하품 threshold를 결정하지 않습니다.

## 5. Head Pose

- 각 pose axis가 실제 얼굴 방향과 일치하는지 확인합니다.
- detector에 따라 axis가 크게 달라지는지 확인합니다.
- `pitch_centered_candidate`의 양수/음수가 실제 head up/down과 어떻게 대응하는지 관찰합니다. 이번 단계에서는 sign을 확정하지 않습니다.

## 6. 최종 입력

- `preferred_detector`: `HOG`, `YUNET`, `TIE`, `UNCERTAIN`
- `manual_decision`: `PASS`, `FAIL`, `UNCERTAIN`
- 판단 근거는 `review_note`에 기록합니다.
""",
        encoding="utf-8",
    )


def run_primary_comparison(config: ComparisonConfig) -> dict[str, Any]:
    """STEP 2-A의 160개 frame에서 HOG와 YuNet을 모두 실행한다."""

    samples = load_step2a_samples(config)
    output_dir = config.path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    hog = HogFaceDetector(int(config.values["hog_upsample"]))
    yunet_config = config.step2a_values["yunet"]
    yunet = YuNetFaceDetector(
        str(config.path("yunet_model")), float(yunet_config["score_threshold"]),
        float(yunet_config["nms_threshold"]), int(yunet_config["top_k"]),
    )
    predictor = Dlib68Predictor(str(config.path("dlib_predictor")))
    records: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    visual_paths: dict[str, list[Path]] = defaultdict(list)
    visual_counts: Counter[str] = Counter()
    visual_config = config.values["visual_audit"]

    for sample in samples:
        frame, decode_failure = _read_exact_frame(sample)
        if frame is None:
            record = {column: None for column in RESULT_COLUMNS}
            record.update(
                sample_id=sample.sample_id, video_id=sample.video_id, split=sample.split,
                label=sample.label, timestamp_sec=sample.timestamp_sec,
                source_frame_index=sample.source_frame_index, source_fps=sample.source_fps,
                hog_success=False, yunet_success=False, hog_detection_count=0,
                yunet_detection_count=0, failure_reason=decode_failure,
            )
            records.append(record)
            continue
        record, hog_path, yunet_path = _compare_sample(sample, frame, hog, yunet, predictor, config)
        records.append(record)
        categories = _visual_categories(record, config)
        comparison = None
        for category, cap_key in categories:
            if visual_counts[category] >= int(visual_config[cap_key]):
                continue
            if comparison is None:
                comparison = _comparison_image(frame, sample, hog_path, yunet_path)
            category_dir = output_dir / "visual_samples" / category
            category_dir.mkdir(parents=True, exist_ok=True)
            image_path = category_dir / f"{sample.sample_id}.jpg"
            if not cv2.imwrite(str(image_path), comparison):
                raise OSError(f"comparison image 저장 실패: {image_path}")
            visual_counts[category] += 1
            visual_paths[category].append(image_path)
            manual_rows.append({
                "sample_id": sample.sample_id, "video_id": sample.video_id,
                "split": sample.split, "label": sample.label,
                "timestamp_sec": sample.timestamp_sec, "category": category,
                "comparison_image_path": image_path.relative_to(output_dir).as_posix(),
                **{column: "" for column in MANUAL_COLUMNS[7:]},
            })

    _write_csv(output_dir / "paired_primary_results.csv", RESULT_COLUMNS, records)
    _write_csv(output_dir / "manual_review.csv", MANUAL_COLUMNS, manual_rows)
    _write_manual_guide(output_dir / "MANUAL_REVIEW_GUIDE.md")
    contact_dir = output_dir / "contact_sheets"
    for category, paths in visual_paths.items():
        create_contact_sheets(
            paths, contact_dir, category, int(visual_config["contact_sheet_columns"]),
            tuple(int(value) for value in visual_config["thumbnail_size"]),
        )
    summary = _summary(records, config, visual_counts)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "comparison_report.txt").write_text(_render_report(summary), encoding="utf-8")
    return summary
