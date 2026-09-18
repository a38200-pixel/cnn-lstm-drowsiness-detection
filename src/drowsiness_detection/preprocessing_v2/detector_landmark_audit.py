"""소규모 detector/landmark 호환성 audit을 실행하고 결과를 기록한다."""

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

from .audit_sampling import AuditSample, build_audit_candidates, load_split_rows, take_unique_samples
from .audit_visualization import annotate_frame, create_contact_sheets
from .detectors import BoundingBox, DetectorResult, HogFaceDetector, YuNetFaceDetector, bbox_iou
from .geometric_features import RatioResult, calculate_ear, calculate_mar, validate_geometry
from .head_pose import HeadPoseResult, estimate_head_pose
from .landmarks import Dlib68Predictor, LandmarkResult


DECISION = "WAITING_FOR_MANUAL_VISUAL_REVIEW"
SAMPLE_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec",
    "source_frame_index", "source_fps", "hog_success", "hog_detection_count",
    "hog_ms", "yunet_attempted", "yunet_success", "yunet_detection_count",
    "yunet_confidence", "yunet_ms", "detector_source", "bbox_x1", "bbox_y1",
    "bbox_x2", "bbox_y2", "face_valid", "landmark_success", "landmark_valid", "landmark_ms",
    "ear_left", "ear_right", "ear_mean", "ear_valid", "mar", "mar_valid",
    "pitch", "yaw", "roll", "head_pose_valid", "head_pose_ms", "ear_mar_ms",
    "geometry_valid", "suspicious_flag", "suspicious_reason", "failure_reason",
    "total_ms",
)
PAIRED_COLUMNS = (
    "sample_id", "video_id", "timestamp_sec", "hog_bbox", "yunet_bbox", "bbox_iou",
    "landmark_mean_difference", "normalized_landmark_difference", "ear_hog",
    "ear_yunet", "ear_abs_diff", "mar_hog", "mar_yunet", "mar_abs_diff",
    "pitch_hog", "pitch_yunet", "pitch_abs_diff", "yaw_hog", "yaw_yunet",
    "yaw_abs_diff", "roll_hog", "roll_yunet", "roll_abs_diff", "yunet_ms",
)
MANUAL_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec", "detector_source",
    "category", "image_path", "bbox_ok", "landmarks_overall_ok", "eyes_ok",
    "mouth_ok", "nose_chin_ok", "pose_axis_ok", "manual_decision", "review_note",
)


@dataclass(frozen=True)
class AuditConfig:
    """경로가 해석된 STEP 2 audit 설정."""

    project_root: Path
    config_path: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        """최상위 config 경로 값을 프로젝트 루트 기준으로 해석한다."""

        value = Path(self.values[key])
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()


def load_audit_config(config_path: Path, project_root: Path) -> AuditConfig:
    """YAML 설정을 읽고 필수 값 및 test split 미사용을 검증한다."""

    resolved = config_path.resolve() if config_path.is_absolute() else (project_root / config_path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("audit config 최상위 값은 mapping이어야 합니다")
    required = {
        "random_seed", "raw_dir", "train_metadata", "val_metadata", "dlib_predictor",
        "yunet_model", "hog_upsample", "sampling", "paired_detector_audit",
        "geometry_validation", "visual_audit", "face_selection", "output_dir",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"audit config 필수 항목 누락: {missing}")
    configured_paths = f"{values['train_metadata']} {values['val_metadata']}".casefold()
    if "test.csv" in configured_paths:
        raise ValueError("STEP 2 config에는 test.csv를 사용할 수 없습니다")
    sampling = values["sampling"]
    if int(sampling["initial_candidate_frames"]) > int(sampling["max_candidate_frames"]):
        raise ValueError("initial_candidate_frames는 max_candidate_frames 이하여야 합니다")
    return AuditConfig(project_root.resolve(), resolved, values)


def describe_dry_run(config: AuditConfig) -> dict[str, Any]:
    """영상이나 detector를 열지 않고 설정과 sampling 후보만 검증한다."""

    train_rows = load_split_rows(config.path("train_metadata"), "train")
    val_rows = load_split_rows(config.path("val_metadata"), "val")
    sampling = config.values["sampling"]
    candidates = build_audit_candidates(
        train_rows,
        val_rows,
        config.project_root,
        sampling["timestamps_sec"],
        int(config.values["random_seed"]),
    )
    return {
        "mode": "DRY_RUN",
        "used_splits": ["train", "val"],
        "test_split_used": False,
        "train_metadata_rows": len(train_rows),
        "val_metadata_rows": len(val_rows),
        "available_candidates": len(candidates),
        "initial_candidate_frames": int(sampling["initial_candidate_frames"]),
        "max_candidate_frames": int(sampling["max_candidate_frames"]),
        "output_dir": str(config.path("output_dir")),
    }


def _read_frame(sample: AuditSample) -> tuple[np.ndarray | None, int, float, str]:
    """실제 FPS로 timestamp에 가장 가까운 source frame을 읽는다."""

    capture = cv2.VideoCapture(str(sample.video_path))
    try:
        if not capture.isOpened():
            return None, -1, 0.0, "VIDEO_OPEN_FAILED"
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not np.isfinite(fps) or fps <= 0:
            return None, -1, fps, "INVALID_SOURCE_FPS"
        frame_index = max(0, int(round(sample.timestamp_sec * fps)))
        if frame_count > 0:
            frame_index = min(frame_index, frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            return None, frame_index, fps, "FRAME_DECODE_FAILED"
        return frame, frame_index, fps, ""
    finally:
        capture.release()


def _empty_detector_result(attempted: bool = False) -> DetectorResult:
    """실행하지 않은 detector를 위한 명시적인 빈 결과를 만든다."""

    return DetectorResult(attempted, False, 0, None, 0.0, "")


def _extract_features(
    frame: np.ndarray,
    bbox: BoundingBox,
    predictor: Dlib68Predictor,
    geometry_config: Mapping[str, Any],
) -> tuple[LandmarkResult, RatioResult, RatioResult, RatioResult, RatioResult, HeadPoseResult, Any, float]:
    """Dlib68, EAR/MAR, head pose와 geometry validity를 한 bbox에서 계산한다."""

    landmark = predictor.predict(frame, bbox)
    invalid_ear = RatioResult(None, False, "EAR_INVALID_GEOMETRY")
    invalid_mar = RatioResult(None, False, "MAR_INVALID_GEOMETRY")
    if not landmark.success or landmark.points is None:
        pose = HeadPoseResult(False, None, None, None, None, 0.0, "LANDMARK_REQUIRED")
        geometry = validate_geometry(bbox, None, frame.shape, invalid_ear, invalid_mar, False, dict(geometry_config))
        return landmark, invalid_ear, invalid_ear, invalid_ear, invalid_mar, pose, geometry, 0.0

    feature_started = perf_counter()
    ear_left, ear_right, ear_mean = calculate_ear(
        landmark.points, float(geometry_config["denominator_epsilon"])
    )
    mar = calculate_mar(landmark.points, float(geometry_config["denominator_epsilon"]))
    ear_mar_ms = (perf_counter() - feature_started) * 1000.0
    pose = estimate_head_pose(landmark.points, frame.shape[1], frame.shape[0])
    geometry = validate_geometry(
        bbox, landmark.points, frame.shape, ear_mean, mar, pose.success, dict(geometry_config)
    )
    return landmark, ear_left, ear_right, ear_mean, mar, pose, geometry, ear_mar_ms


def _value(result: RatioResult) -> float | None:
    """유효한 ratio 결과만 CSV 값으로 반환한다."""

    return result.value if result.valid else None


def _process_sample(
    sample: AuditSample,
    hog: HogFaceDetector,
    yunet: YuNetFaceDetector,
    predictor: Dlib68Predictor,
    geometry_config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None, HeadPoseResult | None, DetectorResult]:
    """라벨을 detector에 전달하지 않고 production candidate pipeline을 실행한다."""

    started = perf_counter()
    frame, frame_index, fps, decode_failure = _read_frame(sample)
    record: dict[str, Any] = {column: None for column in SAMPLE_COLUMNS}
    record.update(
        sample_id=sample.sample_id, video_id=sample.video_id, split=sample.split,
        label=sample.label, timestamp_sec=sample.timestamp_sec,
        source_frame_index=frame_index, source_fps=fps,
    )
    if frame is None:
        record.update(
            hog_success=False, hog_detection_count=0, hog_ms=0.0,
            yunet_attempted=False, yunet_success=False, yunet_detection_count=0,
            yunet_ms=0.0, landmark_success=False, landmark_valid=False,
            ear_valid=False, mar_valid=False,
            head_pose_valid=False, geometry_valid=False, suspicious_flag=False,
            failure_reason=decode_failure, total_ms=(perf_counter() - started) * 1000.0,
        )
        return record, None, None, None, _empty_detector_result()

    hog_result = hog.detect(frame)
    yunet_result = _empty_detector_result()
    selected = hog_result.selected
    detector_source = "HOG" if hog_result.success else ""
    if not hog_result.success:
        yunet_result = yunet.detect(frame)
        selected = yunet_result.selected
        detector_source = "YUNET_FALLBACK" if yunet_result.success else "FACE_NOT_FOUND"

    record.update(
        hog_success=hog_result.success, hog_detection_count=hog_result.detection_count,
        hog_ms=hog_result.elapsed_ms, yunet_attempted=yunet_result.attempted,
        yunet_success=yunet_result.success, yunet_detection_count=yunet_result.detection_count,
        yunet_confidence=yunet_result.selected.confidence if yunet_result.selected else None,
        yunet_ms=yunet_result.elapsed_ms, detector_source=detector_source,
    )
    if selected is None:
        record.update(
            face_valid=False, landmark_success=False, landmark_valid=False,
            ear_valid=False, mar_valid=False,
            head_pose_valid=False, geometry_valid=False, suspicious_flag=False,
            failure_reason=yunet_result.failure_reason or hog_result.failure_reason,
            total_ms=(perf_counter() - started) * 1000.0,
        )
        return record, frame, None, None, hog_result

    bbox = selected.bbox
    landmark, ear_left, ear_right, ear_mean, mar, pose, geometry, ear_mar_ms = _extract_features(
        frame, bbox, predictor, geometry_config
    )
    suspicious_reasons = list(geometry.suspicious_reasons)
    pose_limit = float(geometry_config["suspicious_abs_pose_degrees"])
    if pose.success and max(abs(float(pose.pitch)), abs(float(pose.yaw)), abs(float(pose.roll))) > pose_limit:
        suspicious_reasons.append("SUSPICIOUS_LARGE_POSE")
    failures = list(geometry.failure_reasons)
    if landmark.failure_reason:
        failures.append(landmark.failure_reason)
    if pose.failure_reason and pose.failure_reason != "LANDMARK_REQUIRED":
        failures.append(pose.failure_reason)
    record.update(
        bbox_x1=bbox.x1, bbox_y1=bbox.y1, bbox_x2=bbox.x2, bbox_y2=bbox.y2,
        face_valid=geometry.face_valid, landmark_success=landmark.success,
        landmark_valid=geometry.landmark_valid,
        landmark_ms=landmark.elapsed_ms, ear_left=_value(ear_left), ear_right=_value(ear_right),
        ear_mean=_value(ear_mean), ear_valid=ear_mean.valid, mar=_value(mar), mar_valid=mar.valid,
        pitch=pose.pitch, yaw=pose.yaw, roll=pose.roll, head_pose_valid=pose.success,
        head_pose_ms=pose.elapsed_ms, ear_mar_ms=ear_mar_ms,
        geometry_valid=geometry.geometry_valid, suspicious_flag=bool(suspicious_reasons),
        suspicious_reason=";".join(dict.fromkeys(suspicious_reasons)),
        failure_reason=";".join(dict.fromkeys(failures)),
        total_ms=(perf_counter() - started) * 1000.0,
    )
    return record, frame, landmark.points, pose, hog_result


def _paired_result(
    sample: AuditSample,
    frame: np.ndarray,
    hog_bbox: BoundingBox,
    hog_landmarks: np.ndarray,
    hog_record: Mapping[str, Any],
    yunet: YuNetFaceDetector,
    predictor: Dlib68Predictor,
    geometry_config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, BoundingBox | None, np.ndarray | None, HeadPoseResult | None]:
    """소수 HOG 성공 frame에서 YuNet+Dlib68을 추가 실행해 detector 차이를 비교한다."""

    yunet_result = yunet.detect(frame)
    if not yunet_result.success or yunet_result.selected is None:
        return None, None, None, None
    yunet_bbox = yunet_result.selected.bbox
    landmark, _left, _right, ear, mar, pose, _geometry, _feature_ms = _extract_features(
        frame, yunet_bbox, predictor, geometry_config
    )
    if not landmark.success or landmark.points is None:
        return None, yunet_bbox, None, pose
    distances = np.linalg.norm(hog_landmarks - landmark.points, axis=1)
    mean_difference = float(np.mean(distances))
    normalization = max(float(np.sqrt(hog_bbox.area)), 1.0)

    def absolute_difference(left: Any, right: Any) -> float | None:
        return abs(float(left) - float(right)) if left is not None and right is not None else None

    result = {
        "sample_id": sample.sample_id, "video_id": sample.video_id,
        "timestamp_sec": sample.timestamp_sec, "hog_bbox": str(hog_bbox.as_tuple()),
        "yunet_bbox": str(yunet_bbox.as_tuple()), "bbox_iou": bbox_iou(hog_bbox, yunet_bbox),
        "landmark_mean_difference": mean_difference,
        "normalized_landmark_difference": mean_difference / normalization,
        "ear_hog": hog_record.get("ear_mean"), "ear_yunet": _value(ear),
        "ear_abs_diff": absolute_difference(hog_record.get("ear_mean"), _value(ear)),
        "mar_hog": hog_record.get("mar"), "mar_yunet": _value(mar),
        "mar_abs_diff": absolute_difference(hog_record.get("mar"), _value(mar)),
        "pitch_hog": hog_record.get("pitch"), "pitch_yunet": pose.pitch,
        "pitch_abs_diff": absolute_difference(hog_record.get("pitch"), pose.pitch),
        "yaw_hog": hog_record.get("yaw"), "yaw_yunet": pose.yaw,
        "yaw_abs_diff": absolute_difference(hog_record.get("yaw"), pose.yaw),
        "roll_hog": hog_record.get("roll"), "roll_yunet": pose.roll,
        "roll_abs_diff": absolute_difference(hog_record.get("roll"), pose.roll),
        "yunet_ms": yunet_result.elapsed_ms,
    }
    return result, yunet_bbox, landmark.points, pose


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """고정된 컬럼 순서로 UTF-8 CSV를 기록한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(values: Sequence[Any]) -> dict[str, float | int | None]:
    """None과 비유효한 값을 제외한 기본 분포 통계를 계산한다."""

    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(finite), "mean": mean(finite), "median": median(finite),
        "min": min(finite), "max": max(finite),
    }


def _build_summary(
    config: AuditConfig,
    records: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    visual_counts: Mapping[str, int],
) -> dict[str, Any]:
    """자동 audit 수치를 집계하되 detector 정책을 최종 승인하지 않는다."""

    count = len(records)
    hog_attempted = sum(1 for row in records if row["source_frame_index"] != -1)
    hog_success = sum(bool(row["hog_success"]) for row in records)
    yunet_attempted = sum(bool(row["yunet_attempted"]) for row in records)
    yunet_success = sum(bool(row["yunet_success"]) for row in records)
    final_success = sum(row["detector_source"] in {"HOG", "YUNET_FALLBACK"} for row in records)
    by_source: dict[str, dict[str, int | float]] = {}
    for source in ("HOG", "YUNET_FALLBACK"):
        source_rows = [row for row in records if row["detector_source"] == source]
        successes = sum(bool(row["landmark_success"]) for row in source_rows)
        by_source[source] = {
            "attempted": len(source_rows), "success": successes,
            "failure": len(source_rows) - successes,
            "success_rate": successes / len(source_rows) if source_rows else 0.0,
            "latency_ms": _metric_summary([row["landmark_ms"] for row in source_rows]),
        }
    split_counts = Counter(str(row["split"]) for row in records)
    label_counts = Counter(str(row["label"]) for row in records)
    summary = {
        "decision": DECISION,
        "dataset_scope": {
            "used_splits": ["train", "val"], "test_split_used": False,
            "candidate_frames_processed": count,
            "initial_candidate_frames": int(config.values["sampling"]["initial_candidate_frames"]),
            "max_candidate_frames": int(config.values["sampling"]["max_candidate_frames"]),
            "unique_videos": len({row["video_id"] for row in records}),
            "split_counts": dict(split_counts), "label_counts": dict(label_counts),
            "label_usage": "sampling_and_reporting_only",
            "face_selection_policy": config.values["face_selection"]["policy"],
        },
        "hog": {
            "attempted": hog_attempted, "success": hog_success,
            "failure": hog_attempted - hog_success,
            "success_rate": hog_success / hog_attempted if hog_attempted else 0.0,
            "multiple_detection_count": sum(int(row["hog_detection_count"] or 0) > 1 for row in records),
            "latency_ms": _metric_summary([row["hog_ms"] for row in records]),
        },
        "yunet_fallback": {
            "hog_failures": hog_attempted - hog_success, "attempted": yunet_attempted,
            "success": yunet_success, "recovered": yunet_success,
            "recovery_rate": yunet_success / yunet_attempted if yunet_attempted else 0.0,
            "failure": yunet_attempted - yunet_success,
            "confidence": _metric_summary([row["yunet_confidence"] for row in records]),
            "latency_ms": _metric_summary([row["yunet_ms"] for row in records if row["yunet_attempted"]]),
        },
        "combined_face_detection": {
            "success": final_success, "failure": count - final_success,
            "success_rate": final_success / count if count else 0.0,
        },
        "dlib68": by_source,
        "ear": {
            "valid_rate": sum(bool(row["ear_valid"]) for row in records) / count if count else 0.0,
            "hog_distribution": _metric_summary([row["ear_mean"] for row in records if row["detector_source"] == "HOG"]),
            "yunet_distribution": _metric_summary([row["ear_mean"] for row in records if row["detector_source"] == "YUNET_FALLBACK"]),
            "suspicious_extreme_count": sum("SUSPICIOUS_EAR_EXTREME" in str(row["suspicious_reason"]) for row in records),
        },
        "mar": {
            "valid_rate": sum(bool(row["mar_valid"]) for row in records) / count if count else 0.0,
            "hog_distribution": _metric_summary([row["mar"] for row in records if row["detector_source"] == "HOG"]),
            "yunet_distribution": _metric_summary([row["mar"] for row in records if row["detector_source"] == "YUNET_FALLBACK"]),
            "suspicious_extreme_count": sum("SUSPICIOUS_MAR_EXTREME" in str(row["suspicious_reason"]) for row in records),
        },
        "head_pose": {
            "attempted": sum(bool(row["landmark_success"]) for row in records),
            "success": sum(bool(row["head_pose_valid"]) for row in records),
            "success_rate": (
                sum(bool(row["head_pose_valid"]) for row in records)
                / sum(bool(row["landmark_success"]) for row in records)
                if any(bool(row["landmark_success"]) for row in records) else 0.0
            ),
            "pitch": _metric_summary([row["pitch"] for row in records]),
            "yaw": _metric_summary([row["yaw"] for row in records]),
            "roll": _metric_summary([row["roll"] for row in records]),
            "suspicious_large_pose_count": sum("SUSPICIOUS_LARGE_POSE" in str(row["suspicious_reason"]) for row in records),
        },
        "paired_detector_check": {
            "pair_count": len(paired),
            **{column: _metric_summary([row[column] for row in paired]) for column in (
                "bbox_iou", "landmark_mean_difference", "normalized_landmark_difference",
                "ear_abs_diff", "mar_abs_diff", "pitch_abs_diff", "yaw_abs_diff", "roll_abs_diff",
            )},
        },
        "timing_ms": {column: _metric_summary([row[column] for row in records]) for column in (
            "hog_ms", "yunet_ms", "landmark_ms", "ear_mar_ms", "head_pose_ms", "total_ms",
        )},
        "visual_audit": {
            "category_counts": dict(visual_counts),
            "manual_review_csv": "manual_review.csv", "contact_sheet_dir": "contact_sheets",
        },
        "environment": {
            "python": platform.python_version(), "opencv": cv2.__version__, "os": platform.platform(),
        },
        "limitations": [
            "video-level split이며 subject-wise independence를 보장하지 않음",
            "자동 geometry validation은 landmark의 의미적 정확성을 보장하지 않음",
            "solvePnP angle은 근사 camera model에 기반하며 ground-truth가 아님",
            "수동 visual review 전에는 detector 정책을 승인하지 않음",
        ],
        "config_path": str(config.config_path),
    }
    return summary


def _render_report(summary: Mapping[str, Any]) -> str:
    """요구된 section을 포함하는 사람이 읽기 쉬운 audit report를 만든다."""

    sections = (
        ("Dataset Scope", summary["dataset_scope"]), ("HOG", summary["hog"]),
        ("YuNet Fallback", summary["yunet_fallback"]),
        ("Combined Face Detection", summary["combined_face_detection"]),
        ("Dlib68", summary["dlib68"]), ("EAR", summary["ear"]),
        ("MAR", summary["mar"]), ("Head Pose", summary["head_pose"]),
        ("Paired Detector Check", summary["paired_detector_check"]),
        ("Timing", summary["timing_ms"]), ("Visual Audit", summary["visual_audit"]),
        ("Decision", {"decision": summary["decision"]}),
        ("Limitations", summary["limitations"]),
    )
    lines = [
        "Experiment 2 — Detector / Landmark Compatibility Audit",
        "라벨은 sampling/reporting에만 사용하며 preprocessing decision에는 사용하지 않음.",
        "Production candidate: HOG 성공 시 종료, 실패 시에만 YuNet fallback.",
        "Multiple-face policy: HOG/YuNet 모두 유효 bbox 중 가장 큰 얼굴을 선택함.",
        "Paired YuNet 실행은 소수 비교 audit 전용이며 production latency와 분리함.",
        "EAR: Dlib 36–41/42–47, MAR: Dlib 60–67 표준 aspect ratio.",
        "Head pose: Dlib 30/8/36/45/48/54, solvePnP 근사 camera model.",
        "",
    ]
    for title, value in sections:
        lines.extend((f"[{title}]", json.dumps(value, ensure_ascii=False, indent=2), ""))
    return "\n".join(lines)


def _write_manual_guide(path: Path) -> None:
    """자동 수치만으로 발견하기 어려운 landmark 오류의 수동 검토 기준을 기록한다."""

    path.write_text(
        """# STEP 2 수동 Visual Review 가이드

자동 audit의 최종 상태는 `WAITING_FOR_MANUAL_VISUAL_REVIEW`입니다. CSV의 수동 입력 컬럼은 비워 두었으며 `PASS`, `FAIL`, `UNCERTAIN` 중 하나를 직접 기록합니다.

## bbox_ok

- bbox가 얼굴 전체를 적절히 포함하는가?
- 이마, 턱, 눈, 입이 과도하게 잘리지 않았는가?
- 배경을 지나치게 많이 포함하지 않는가?

## landmarks_overall_ok

- 68개 landmark가 얼굴 구조를 전반적으로 따르는가?

## eyes_ok

- 눈 landmark가 실제 눈꺼풀과 눈꼬리에 위치하는가?
- 안경테나 눈썹에 잘못 붙지 않았는가?
- 좌우 눈 geometry가 붕괴하지 않았는가?

## mouth_ok

- 입 landmark가 실제 입술 경계를 따르는가?
- landmark가 입 밖으로 이동하지 않았는가?

## nose_chin_ok

- Head Pose에 사용하는 코끝과 턱 landmark가 정상인가?
- 큰 yaw에서 턱 landmark가 얼굴 밖으로 이탈하지 않는가?

## pose_axis_ok

- 빨강/초록/파랑 pose axis가 실제 얼굴 방향과 대체로 일치하는가?
- axis 방향이 명백히 반대이거나 불안정하지 않은가?

## manual_decision

- `PASS`: 이후 preprocessing에 사용할 수 있음
- `FAIL`: bbox, landmark 또는 pose가 신뢰하기 어려움
- `UNCERTAIN`: 단일 이미지로 판단하기 어렵거나 추가 확인이 필요함

특히 안경테·눈썹에 붙은 eye landmark, 큰 yaw의 반대쪽 눈 붕괴, 입 밖의 mouth landmark, 얼굴 일부를 자른 bbox, 과도한 배경, 실제 방향과 다른 pose axis를 주의합니다. 라벨은 참고 표시일 뿐 판단 기준이 아닙니다.
""",
        encoding="utf-8",
    )


def run_audit(config: AuditConfig) -> dict[str, Any]:
    """제한된 train/validation frame에서 STEP 2 자동 audit을 실행한다."""

    output_dir = config.path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_split_rows(config.path("train_metadata"), "train")
    val_rows = load_split_rows(config.path("val_metadata"), "val")
    sampling = config.values["sampling"]
    candidates = build_audit_candidates(
        train_rows, val_rows, config.project_root, sampling["timestamps_sec"], int(config.values["random_seed"])
    )
    candidates = take_unique_samples(candidates, int(sampling["max_candidate_frames"]))

    hog = HogFaceDetector(int(config.values["hog_upsample"]))
    yunet_config = config.values["yunet"]
    yunet = YuNetFaceDetector(
        str(config.path("yunet_model")), float(yunet_config["score_threshold"]),
        float(yunet_config["nms_threshold"]), int(yunet_config["top_k"]),
    )
    predictor = Dlib68Predictor(str(config.path("dlib_predictor")))
    geometry_config = config.values["geometry_validation"]
    paired_config = config.values["paired_detector_audit"]
    visual_config = config.values["visual_audit"]

    records: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    visual_counts: Counter[str] = Counter()
    visual_paths: dict[str, list[Path]] = defaultdict(list)
    manual_rows: list[dict[str, Any]] = []
    fallback_successes = 0
    initial_limit = int(sampling["initial_candidate_frames"])
    fallback_target = int(sampling["target_yunet_fallback_samples"])

    def save_visual(
        category: str, cap_key: str, sample: AuditSample, frame: np.ndarray,
        record: Mapping[str, Any], bbox: BoundingBox | None,
        landmarks: np.ndarray | None, pose: HeadPoseResult | None,
    ) -> None:
        maximum = int(visual_config[cap_key])
        if visual_counts[category] >= maximum:
            return
        category_dir = output_dir / "visual_samples" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        image = annotate_frame(frame, bbox, landmarks, pose.axis_points if pose else None, record)
        image_path = category_dir / f"{sample.sample_id}.jpg"
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"visual sample 저장 실패: {image_path}")
        visual_counts[category] += 1
        visual_paths[category].append(image_path)
        manual_rows.append({
            "sample_id": sample.sample_id, "video_id": sample.video_id,
            "split": sample.split, "label": sample.label,
            "timestamp_sec": sample.timestamp_sec,
            "detector_source": record.get("detector_source", ""), "category": category,
            "image_path": image_path.relative_to(output_dir).as_posix(),
            "bbox_ok": "", "landmarks_overall_ok": "", "eyes_ok": "",
            "mouth_ok": "", "nose_chin_ok": "", "pose_axis_ok": "",
            "manual_decision": "", "review_note": "",
        })

    for index, sample in enumerate(candidates):
        if index >= initial_limit and fallback_successes >= fallback_target:
            break
        record, frame, landmarks, pose, hog_result = _process_sample(
            sample, hog, yunet, predictor, geometry_config
        )
        records.append(record)
        if record["detector_source"] == "YUNET_FALLBACK":
            fallback_successes += 1
        bbox = None
        if record.get("bbox_x1") is not None:
            bbox = BoundingBox(int(record["bbox_x1"]), int(record["bbox_y1"]), int(record["bbox_x2"]), int(record["bbox_y2"]))
        if frame is not None:
            if record["detector_source"] == "HOG":
                save_visual("hog_success", "max_hog_samples", sample, frame, record, bbox, landmarks, pose)
            elif record["detector_source"] == "YUNET_FALLBACK":
                save_visual("yunet_fallback", "max_yunet_samples", sample, frame, record, bbox, landmarks, pose)
            elif record["detector_source"] == "FACE_NOT_FOUND":
                save_visual("detector_failed", "max_failed_samples", sample, frame, record, None, None, None)
            if bbox is not None and not record["landmark_success"]:
                save_visual("landmark_failed", "max_failed_samples", sample, frame, record, bbox, None, None)
            if record["suspicious_flag"]:
                save_visual("suspicious_geometry", "max_suspicious_samples", sample, frame, record, bbox, landmarks, pose)
            if "SUSPICIOUS_LARGE_POSE" in str(record["suspicious_reason"]):
                save_visual("large_pose", "max_large_pose_samples", sample, frame, record, bbox, landmarks, pose)

        paired_limit = int(paired_config["max_samples"])
        if (
            bool(paired_config["enabled"]) and len(paired_rows) < paired_limit and frame is not None
            and hog_result.success and hog_result.selected is not None and landmarks is not None
        ):
            pair, pair_bbox, pair_landmarks, pair_pose = _paired_result(
                sample, frame, hog_result.selected.bbox, landmarks, record,
                yunet, predictor, geometry_config,
            )
            if pair is not None:
                paired_rows.append(pair)
                pair_record = dict(record)
                pair_record["detector_source"] = "YUNET_PAIRED_AUDIT"
                save_visual(
                    "paired_detector_difference", "max_paired_samples", sample, frame,
                    pair_record, pair_bbox, pair_landmarks, pair_pose,
                )

    _write_csv(output_dir / "sample_results.csv", SAMPLE_COLUMNS, records)
    _write_csv(output_dir / "paired_detector_results.csv", PAIRED_COLUMNS, paired_rows)
    _write_csv(output_dir / "manual_review.csv", MANUAL_COLUMNS, manual_rows)
    _write_manual_guide(output_dir / "MANUAL_REVIEW_GUIDE.md")
    contact_dir = output_dir / "contact_sheets"
    for category, paths in visual_paths.items():
        create_contact_sheets(
            paths, contact_dir, category, int(visual_config["contact_sheet_columns"]),
            tuple(int(value) for value in visual_config["thumbnail_size"]),
        )

    summary = _build_summary(config, records, paired_rows, visual_counts)
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "audit_report.txt").write_text(_render_report(summary), encoding="utf-8")
    return summary
