"""저장된 YuNet bbox에서 Dlib68 fitting ROI margin 후보를 비교한다."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .angle_utils import circular_angle_difference, pitch_centered_candidate
from .detector_primary_comparison import (
    SourceSample,
    _distribution,
    _read_exact_frame,
    load_comparison_config,
    load_step2a_samples,
)
from .detectors import BoundingBox
from .geometric_features import (
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_EYE_INDICES,
    RatioResult,
    calculate_ear,
    calculate_mar,
    validate_geometry,
)
from .head_pose import HeadPoseResult, estimate_head_pose
from .landmarks import Dlib68Predictor, LandmarkResult


DECISION = "WAITING_FOR_MANUAL_LANDMARK_ROI_REVIEW"
MANUAL_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec",
    "raw_eye_ok", "m05_eye_ok", "m10_eye_ok", "m15_eye_ok",
    "raw_mouth_ok", "m05_mouth_ok", "m10_mouth_ok", "m15_mouth_ok",
    "raw_overall_landmark_ok", "m05_overall_landmark_ok",
    "m10_overall_landmark_ok", "m15_overall_landmark_ok",
    "raw_pose_ok", "m05_pose_ok", "m10_pose_ok", "m15_pose_ok",
    "preferred_landmark_roi", "review_decision", "review_note",
)
BASE_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec", "source_frame_index",
    "yunet_success", "yunet_confidence", "det_bbox_x1", "det_bbox_y1",
    "det_bbox_x2", "det_bbox_y2",
)
CANDIDATE_FIELDS = (
    "roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi_clipped", "clipped_fraction",
    "landmark_success", "landmarks_json", "ear", "ear_valid", "mar", "mar_valid",
    "pitch_raw", "pitch_centered", "yaw", "roll", "pose_success",
    "geometry_valid", "landmark_ms", "feature_ms", "pipeline_ms", "failure_reason",
)
DIFFERENCE_COLUMNS = (
    "raw_m05_normalized_landmark_diff", "raw_m10_normalized_landmark_diff",
    "raw_m15_normalized_landmark_diff", "m05_m10_normalized_landmark_diff",
    "m10_m15_normalized_landmark_diff",
)
RESULT_COLUMNS = BASE_COLUMNS + tuple(
    f"{name.lower()}_{field}" for name in ("RAW", "M05", "M10", "M15") for field in CANDIDATE_FIELDS
) + DIFFERENCE_COLUMNS


@dataclass(frozen=True)
class RoiAuditConfig:
    """프로젝트 기준 절대 경로와 STEP 2-A geometry 설정을 보관한다."""

    project_root: Path
    config_path: Path
    values: dict[str, Any]
    geometry: dict[str, Any]

    def path(self, key: str) -> Path:
        """상대 설정 경로를 프로젝트 기준 절대 경로로 바꾼다."""

        path = Path(self.values[key])
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True)
class ExpandedRoi:
    """확장 전후 면적과 frame clipping 정보를 포함한 landmark ROI다."""

    bbox: BoundingBox
    clipped: bool
    clipped_fraction: float


@dataclass(frozen=True)
class CandidateResult:
    """하나의 fitting ROI에서 계산한 landmark와 파생 feature 결과다."""

    roi: ExpandedRoi
    landmark: LandmarkResult
    ear: RatioResult
    mar: RatioResult
    pose: HeadPoseResult
    geometry_valid: bool
    feature_ms: float
    pipeline_ms: float
    failure_reason: str


def load_roi_audit_config(config_path: Path, project_root: Path) -> RoiAuditConfig:
    """STEP 2-C 설정과 기존 geometry validation 설정을 검증해 읽는다."""

    resolved = config_path.resolve() if config_path.is_absolute() else (project_root / config_path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    required = {
        "source_results_dir", "step2b_config", "train_metadata", "val_metadata",
        "dlib_predictor", "margin_candidates", "visual_audit", "output_dir",
    }
    if not isinstance(values, dict) or required - values.keys():
        raise ValueError(f"STEP 2-C config 필수 항목 누락: {sorted(required - set(values or {}))}")
    if "test.csv" in f"{values['train_metadata']} {values['val_metadata']}".casefold():
        raise ValueError("STEP 2-C에서는 test.csv를 사용할 수 없습니다")
    expected = {"RAW": 0.0, "M05": 0.05, "M10": 0.10, "M15": 0.15}
    actual = {str(key).upper(): float(value) for key, value in values["margin_candidates"].items()}
    if actual != expected:
        raise ValueError(f"margin 후보는 {expected}와 같아야 합니다: {actual}")
    comparison = load_comparison_config(Path(values["step2b_config"]), project_root)
    return RoiAuditConfig(project_root.resolve(), resolved, values, comparison.step2a_values["geometry_validation"])


def expand_landmark_roi(
    bbox: BoundingBox, margin_ratio: float, frame_width: int, frame_height: int,
) -> ExpandedRoi:
    """bbox width와 height에 margin을 각각 적용하고 frame 경계로 clipping한다."""

    if not bbox.valid or frame_width <= 0 or frame_height <= 0 or margin_ratio < 0:
        raise ValueError("유효한 bbox, frame 크기, 음수가 아닌 margin이 필요합니다")
    margin_x = bbox.width * margin_ratio
    margin_y = bbox.height * margin_ratio
    raw = (bbox.x1 - margin_x, bbox.y1 - margin_y, bbox.x2 + margin_x, bbox.y2 + margin_y)
    clipped = BoundingBox(
        max(0, int(np.floor(raw[0]))), max(0, int(np.floor(raw[1]))),
        min(frame_width, int(np.ceil(raw[2]))), min(frame_height, int(np.ceil(raw[3]))),
    )
    raw_area = max(0.0, raw[2] - raw[0]) * max(0.0, raw[3] - raw[1])
    fraction = 1.0 - clipped.area / raw_area if raw_area > 0 else 1.0
    was_clipped = any(abs(value - other) > 1.0e-9 for value, other in zip(raw, clipped.as_tuple()))
    return ExpandedRoi(clipped, was_clipped, float(max(0.0, min(1.0, fraction))))


def _load_source_rows(config: RoiAuditConfig) -> list[dict[str, str]]:
    """STEP 2-B 결과에서 새 sampling 없이 저장된 YuNet bbox 행을 읽는다."""

    path = config.path("source_results_dir") / "paired_primary_results.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 160 or any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("STEP 2-B의 train/val 160개 frame 결과가 필요합니다")
    return rows


def _source_samples(config: RoiAuditConfig) -> dict[str, SourceSample]:
    """STEP 2-A와 STEP 2-B가 공유하는 원본 frame 위치를 복구한다."""

    comparison = load_comparison_config(Path(config.values["step2b_config"]), config.project_root)
    return {sample.sample_id: sample for sample in load_step2a_samples(comparison)}


def describe_roi_audit_dry_run(config: RoiAuditConfig) -> dict[str, Any]:
    """영상과 model을 실행하지 않고 STEP 2-C 입력 범위와 출력 계획을 검증한다."""

    rows = _load_source_rows(config)
    usable = [row for row in rows if row["yunet_success"].casefold() == "true"]
    return {
        "mode": "DRY_RUN", "source_frame_count": len(rows), "yunet_success_frame_count": len(usable),
        "unique_video_count": len({row["video_id"] for row in usable}),
        "split_counts": dict(Counter(row["split"] for row in usable)),
        "test_split_used": False, "new_random_sampling": False,
        "detector_rerun": False, "stored_yunet_bbox_reused": True,
        "margin_candidates": config.values["margin_candidates"],
        "forced_square": False, "decision": DECISION,
        "output_dir": str(config.path("output_dir")),
    }


def _failed_candidate(roi: ExpandedRoi, landmark: LandmarkResult) -> CandidateResult:
    """landmark 실패를 공통 결과 구조로 표현한다."""

    ratio = RatioResult(None, False, "LANDMARK_REQUIRED")
    pose = HeadPoseResult(False, None, None, None, None, 0.0, "LANDMARK_REQUIRED")
    return CandidateResult(roi, landmark, ratio, ratio, pose, False, 0.0, landmark.elapsed_ms, landmark.failure_reason)


def _run_candidate(
    frame: np.ndarray, roi: ExpandedRoi, predictor: Dlib68Predictor, geometry: dict[str, Any],
) -> CandidateResult:
    """동일 frame의 하나의 ROI에서 Dlib68과 모든 파생 feature를 계산한다."""

    landmark = predictor.predict(frame, roi.bbox)
    if not landmark.success or landmark.points is None:
        return _failed_candidate(roi, landmark)
    started = perf_counter()
    _left, _right, ear = calculate_ear(landmark.points, float(geometry["denominator_epsilon"]))
    mar = calculate_mar(landmark.points, float(geometry["denominator_epsilon"]))
    pose = estimate_head_pose(landmark.points, frame.shape[1], frame.shape[0])
    validation = validate_geometry(roi.bbox, landmark.points, frame.shape, ear, mar, pose.success, geometry)
    feature_ms = (perf_counter() - started) * 1000.0
    failures = [landmark.failure_reason, ear.failure_reason, mar.failure_reason, pose.failure_reason]
    return CandidateResult(
        roi, landmark, ear, mar, pose, validation.geometry_valid, feature_ms,
        landmark.elapsed_ms + feature_ms, ";".join(value for value in failures if value),
    )


def _candidate_fields(prefix: str, result: CandidateResult) -> dict[str, Any]:
    """후보 결과를 고정된 CSV 열로 평탄화한다."""

    pose = result.pose
    points = result.landmark.points
    return {
        f"{prefix}_roi_x1": result.roi.bbox.x1, f"{prefix}_roi_y1": result.roi.bbox.y1,
        f"{prefix}_roi_x2": result.roi.bbox.x2, f"{prefix}_roi_y2": result.roi.bbox.y2,
        f"{prefix}_roi_clipped": result.roi.clipped,
        f"{prefix}_clipped_fraction": result.roi.clipped_fraction,
        f"{prefix}_landmark_success": result.landmark.success,
        f"{prefix}_landmarks_json": json.dumps(points.tolist(), separators=(",", ":")) if points is not None else "",
        f"{prefix}_ear": result.ear.value, f"{prefix}_ear_valid": result.ear.valid,
        f"{prefix}_mar": result.mar.value, f"{prefix}_mar_valid": result.mar.valid,
        f"{prefix}_pitch_raw": pose.pitch,
        f"{prefix}_pitch_centered": pitch_centered_candidate(pose.pitch) if pose.success and pose.pitch is not None else None,
        f"{prefix}_yaw": pose.yaw, f"{prefix}_roll": pose.roll,
        f"{prefix}_pose_success": pose.success, f"{prefix}_geometry_valid": result.geometry_valid,
        f"{prefix}_landmark_ms": result.landmark.elapsed_ms, f"{prefix}_feature_ms": result.feature_ms,
        f"{prefix}_pipeline_ms": result.pipeline_ms, f"{prefix}_failure_reason": result.failure_reason,
    }


def normalized_landmark_difference(
    left: np.ndarray | None, right: np.ndarray | None, reference_bbox: BoundingBox,
) -> float | None:
    """두 68점의 평균 이동량을 raw detection bbox 면적의 제곱근으로 정규화한다."""

    if left is None or right is None:
        return None
    return float(np.mean(np.linalg.norm(left - right, axis=1)) / max(np.sqrt(reference_bbox.area), 1.0))


def _process_row(
    row: dict[str, str], sample: SourceSample, frame: np.ndarray, predictor: Dlib68Predictor,
    config: RoiAuditConfig,
) -> tuple[dict[str, Any], dict[str, CandidateResult]]:
    """저장된 하나의 YuNet bbox로 네 ROI 후보를 생성하고 detector는 실행하지 않는다."""

    bbox = BoundingBox(*(int(float(row[f"yunet_bbox_{axis}"])) for axis in ("x1", "y1", "x2", "y2")))
    height, width = frame.shape[:2]
    results = {
        name: _run_candidate(frame, expand_landmark_roi(bbox, float(ratio), width, height), predictor, config.geometry)
        for name, ratio in config.values["margin_candidates"].items()
    }
    record: dict[str, Any] = {
        "sample_id": row["sample_id"], "video_id": row["video_id"], "split": row["split"],
        "label": row["label"], "timestamp_sec": row["timestamp_sec"],
        "source_frame_index": row["source_frame_index"], "yunet_success": True,
        "yunet_confidence": row["yunet_confidence"], "det_bbox_x1": bbox.x1,
        "det_bbox_y1": bbox.y1, "det_bbox_x2": bbox.x2, "det_bbox_y2": bbox.y2,
    }
    for name, result in results.items():
        record.update(_candidate_fields(name.lower(), result))
    pairs = (("raw", "m05"), ("raw", "m10"), ("raw", "m15"), ("m05", "m10"), ("m10", "m15"))
    for left, right in pairs:
        record[f"{left}_{right}_normalized_landmark_diff"] = normalized_landmark_difference(
            results[left.upper()].landmark.points, results[right.upper()].landmark.points, bbox,
        )
    return record, results


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """고정 열 순서의 UTF-8 CSV를 기록한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_summary(records: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    """후보별 성공률, 분포, clipping과 latency를 집계한다."""

    count = len(records)
    truth = lambda field: sum(str(row[f"{prefix}_{field}"]).casefold() == "true" for row in records)
    return {
        "count": count,
        "landmark_success": truth("landmark_success"), "geometry_valid": truth("geometry_valid"),
        "ear_valid": truth("ear_valid"), "mar_valid": truth("mar_valid"),
        "pose_success": truth("pose_success"), "roi_clipped": truth("roi_clipped"),
        "clipped_fraction": _distribution([row[f"{prefix}_clipped_fraction"] for row in records]),
        "ear": _distribution([row[f"{prefix}_ear"] for row in records]),
        "mar": _distribution([row[f"{prefix}_mar"] for row in records]),
        "pitch_centered": _distribution([row[f"{prefix}_pitch_centered"] for row in records]),
        "yaw": _distribution([row[f"{prefix}_yaw"] for row in records]),
        "roll": _distribution([row[f"{prefix}_roll"] for row in records]),
        "pipeline_ms": _distribution([row[f"{prefix}_pipeline_ms"] for row in records]),
    }


def _temporal_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """같은 video의 인접 audit timestamp 사이 절대 변화량을 진단용으로 요약한다."""

    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_video[str(row["video_id"])].append(row)
    output: dict[str, Any] = {}
    for candidate in ("raw", "m05", "m10", "m15"):
        output[candidate] = {}
        for feature in ("ear", "mar", "pitch_centered", "yaw", "roll"):
            changes: list[float] = []
            for rows in by_video.values():
                ordered = sorted(rows, key=lambda row: float(row["timestamp_sec"]))
                values = [row[f"{candidate}_{feature}"] for row in ordered]
                for left, right in zip(values, values[1:]):
                    if left is None or right is None:
                        continue
                    if feature in {"pitch_centered", "yaw", "roll"}:
                        changes.append(circular_angle_difference(float(left), float(right)))
                    else:
                        changes.append(abs(float(left) - float(right)))
            output[candidate][feature] = _distribution(changes)
    return output


def _candidate_difference_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """RAW 기준과 인접 margin 후보 사이 feature 차이 분포를 계산한다."""

    pairs = (("raw", "m05"), ("raw", "m10"), ("raw", "m15"), ("m05", "m10"), ("m10", "m15"))
    output: dict[str, Any] = {}
    for left, right in pairs:
        pair = f"{left}_{right}"
        output[pair] = {}
        for feature in ("ear", "mar", "pitch_raw", "yaw", "roll"):
            differences: list[float] = []
            for row in records:
                first, second = row[f"{left}_{feature}"], row[f"{right}_{feature}"]
                if first is None or second is None:
                    continue
                if feature in {"pitch_raw", "yaw", "roll"}:
                    differences.append(circular_angle_difference(float(first), float(second)))
                else:
                    differences.append(abs(float(first) - float(second)))
            output[pair][feature] = _distribution(differences)
    return output


def _select_visual_ids(config: RoiAuditConfig, records: Sequence[Mapping[str, Any]]) -> list[str]:
    """STEP 2-B 문제 표본을 우선 재사용하고 부족분만 정상 참조로 채운다."""

    target = int(config.values["visual_audit"]["target_samples"])
    usable = {str(row["sample_id"]): row for row in records}
    selected: list[str] = []
    manifest = config.path("source_results_dir") / "review_pack" / "review_manifest.csv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["sample_id"]
                if sample_id in usable and sample_id not in selected:
                    selected.append(sample_id)
    for sample_id in sorted(usable):
        if len(selected) >= target:
            break
        if sample_id not in selected:
            selected.append(sample_id)
    return selected[:target]


def _panel(
    frame: np.ndarray, title: str, detection: BoundingBox, result: CandidateResult,
    size: tuple[int, int],
) -> np.ndarray:
    """detection bbox와 fitting ROI를 다른 색으로 표시한 후보 panel을 만든다."""

    canvas = frame.copy()
    cv2.rectangle(canvas, (detection.x1, detection.y1), (detection.x2, detection.y2), (255, 255, 0), 2)
    cv2.rectangle(canvas, (result.roi.bbox.x1, result.roi.bbox.y1), (result.roi.bbox.x2, result.roi.bbox.y2), (0, 255, 0), 2)
    points = result.landmark.points
    if points is not None:
        eyes = set(LEFT_EYE_INDICES + RIGHT_EYE_INDICES)
        mouth = set(MOUTH_INDICES)
        for index, point in enumerate(points.astype(np.int32)):
            color = (0, 255, 255) if index in eyes else (255, 0, 255) if index in mouth else (255, 255, 255)
            cv2.circle(canvas, tuple(point), 2, color, -1)
        if result.pose.axis_points is not None:
            origin = tuple(points[30].astype(np.int32))
            for endpoint, color in zip(result.pose.axis_points.astype(np.int32), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                cv2.line(canvas, origin, tuple(endpoint), color, 2)
    centered = pitch_centered_candidate(result.pose.pitch) if result.pose.pitch is not None else None
    lines = [title, "cyan=det bbox green=landmark ROI", f"EAR={result.ear.value or 0:.3f} MAR={result.mar.value or 0:.3f} P={centered if centered is not None else float('nan'):.2f}"]
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)


def _comparison_image(
    frame: np.ndarray, sample_id: str, detection: BoundingBox,
    results: Mapping[str, CandidateResult], panel_size: tuple[int, int],
) -> np.ndarray:
    """Original, RAW, +5%, +10%, +15%를 한 줄에 배치한다."""

    original = cv2.resize(frame, panel_size, interpolation=cv2.INTER_AREA)
    cv2.putText(original, f"{sample_id} | ORIGINAL", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    panels = [original] + [_panel(frame, name, detection, results[name], panel_size) for name in ("RAW", "M05", "M10", "M15")]
    return np.hstack(panels)


def _write_contact_sheets(images: Sequence[Path], output_dir: Path, rows_per_page: int = 2) -> list[Path]:
    """가로 5-panel 비교 이미지를 원본 해상도로 묶어 contact sheet를 만든다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for start in range(0, len(images), rows_per_page):
        page = [cv2.imread(str(path)) for path in images[start:start + rows_per_page]]
        valid = [image for image in page if image is not None]
        if not valid:
            continue
        canvas = np.vstack(valid)
        path = output_dir / f"roi_margin_{start // rows_per_page + 1:02d}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise OSError(f"contact sheet 저장 실패: {path}")
        saved.append(path)
    return saved


def _render_report(summary: Mapping[str, Any]) -> str:
    """자동 best margin을 선언하지 않는 사람이 읽는 보고서를 만든다."""

    lines = [
        "STEP 2-C YuNet Landmark ROI Margin Audit", "", "[Scope]",
        f"Source frames: {summary['scope']['source_frames']}",
        f"YuNet success frames: {summary['scope']['yunet_success_frames']}",
        f"Unique videos: {summary['scope']['unique_videos']}",
        "Test split used: False", "", "[Margin Candidates]",
        "RAW=0%, M05=5%, M10=10%, M15=15% (width/height independently, no forced square)",
    ]
    for name in ("raw", "m05", "m10", "m15"):
        item = summary["candidates"][name]
        lines.extend([
            "", f"[{name.upper()}]", f"Dlib68 success: {item['landmark_success']}/{item['count']}",
            f"Geometry valid: {item['geometry_valid']}/{item['count']}",
            f"EAR valid: {item['ear_valid']}/{item['count']}",
            f"MAR valid: {item['mar_valid']}/{item['count']}",
            f"Head Pose success: {item['pose_success']}/{item['count']}",
            f"ROI clipped: {item['roi_clipped']}/{item['count']}",
            f"Dlib+feature mean latency: {item['pipeline_ms']['mean']} ms",
        ])
    lines.extend(["", "[Geometry]", "후보별 geometry valid rate는 위 후보 표를 참조하십시오."])
    for feature, title in (("ear", "[EAR]"), ("mar", "[MAR]")):
        lines.append("")
        lines.append(title)
        for pair, values in summary["candidate_differences"].items():
            item = values[feature]
            lines.append(f"{pair} absolute difference: median={item['median']} p95={item['p95']}")
    lines.extend(["", "[Head Pose]"])
    for pair, values in summary["candidate_differences"].items():
        lines.append(
            f"{pair} circular difference: pitch={values['pitch_raw']['median']} "
            f"yaw={values['yaw']['median']} roll={values['roll']['median']} (median degrees)"
        )
    lines.extend(["", "[ROI Boundary]"])
    for name in ("raw", "m05", "m10", "m15"):
        item = summary["candidates"][name]
        lines.append(f"{name.upper()} clipped: {item['roi_clipped']}/{item['count']}")
    lines.extend(["", "[Latency]"])
    for name in ("raw", "m05", "m10", "m15"):
        item = summary["candidates"][name]["pipeline_ms"]
        lines.append(f"{name.upper()} Dlib+feature: mean={item['mean']} median={item['median']} p95={item['p95']} ms")
    lines.extend([
        "", "[Visual Review]", f"Review samples: {summary['visual_review']['sample_count']}",
        f"Manual CSV: {summary['visual_review']['manual_csv']}", "", "[Decision]", DECISION,
        "", "[Limitations]", "No ground-truth landmarks; HOG values are reference only.",
        "Temporal changes are diagnostics, not drowsiness labels or an automatic margin score.",
        "Context CNN crop margin and Head Pose sign convention remain separate unresolved policies.",
    ])
    return "\n".join(lines) + "\n"


def run_roi_margin_audit(config: RoiAuditConfig) -> dict[str, Any]:
    """YuNet 성공 frame에서 저장된 bbox를 재사용해 STEP 2-C artifact를 생성한다."""

    output = config.path("output_dir")
    if output.exists():
        raise FileExistsError(f"기존 STEP 2-C 결과를 덮어쓰지 않습니다: {output}")
    output.mkdir(parents=True)
    rows = [row for row in _load_source_rows(config) if row["yunet_success"].casefold() == "true"]
    samples = _source_samples(config)
    predictor = Dlib68Predictor(str(config.path("dlib_predictor")))
    records: list[dict[str, Any]] = []
    visual_cache: dict[str, tuple[np.ndarray, BoundingBox, dict[str, CandidateResult]]] = {}
    selected_ids = set(_select_visual_ids(config, rows))
    for row in rows:
        sample = samples[row["sample_id"]]
        frame, failure = _read_exact_frame(sample)
        if frame is None:
            raise RuntimeError(f"{sample.sample_id} frame decode 실패: {failure}")
        record, results = _process_row(row, sample, frame, predictor, config)
        records.append(record)
        if sample.sample_id in selected_ids:
            detection = BoundingBox(*(int(record[f"det_bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2")))
            visual_cache[sample.sample_id] = (frame, detection, results)
    _write_csv(output / "roi_margin_results.csv", RESULT_COLUMNS, records)
    images_dir = output / "visual_samples"
    images_dir.mkdir()
    panel_size = tuple(int(value) for value in config.values["visual_audit"]["panel_size"])
    image_paths: list[Path] = []
    for sample_id in _select_visual_ids(config, records):
        frame, detection, results = visual_cache[sample_id]
        path = images_dir / f"{sample_id}.jpg"
        if not cv2.imwrite(str(path), _comparison_image(frame, sample_id, detection, results, panel_size)):
            raise OSError(f"visual sample 저장 실패: {path}")
        image_paths.append(path)
    sheets = _write_contact_sheets(
        image_paths, output / "contact_sheets", int(config.values["visual_audit"]["contact_sheet_rows"]),
    )
    manual_rows = [
        {column: row.get(column, "") if column in {"sample_id", "video_id", "split", "label", "timestamp_sec"} else "" for column in MANUAL_COLUMNS}
        for row in records if row["sample_id"] in selected_ids
    ]
    _write_csv(output / "roi_margin_manual_review.csv", MANUAL_COLUMNS, manual_rows)
    summary = {
        "scope": {
            "source_frames": 160, "yunet_success_frames": len(records),
            "unique_videos": len({row["video_id"] for row in records}),
            "split_counts": dict(Counter(str(row["split"]) for row in records)),
            "test_split_used": False, "new_random_sampling": False,
            "stored_yunet_bbox_reused": True, "detector_rerun": False,
        },
        "margin_candidates": config.values["margin_candidates"],
        "candidates": {name: _candidate_summary(records, name) for name in ("raw", "m05", "m10", "m15")},
        "landmark_differences": {column: _distribution([row[column] for row in records]) for column in DIFFERENCE_COLUMNS},
        "candidate_differences": _candidate_difference_summary(records),
        "temporal_consistency": _temporal_summary(records),
        "visual_review": {
            "sample_count": len(image_paths), "contact_sheet_count": len(sheets),
            "manual_csv": "roi_margin_manual_review.csv",
        },
        "decision": DECISION,
        "limitations": [
            "ground-truth landmark가 없어 자동 best margin을 선택하지 않음",
            "HOG feature는 정답이 아니라 참고값임",
            "Context CNN crop margin은 별도 정책임",
            "Head Pose sign convention과 졸음 threshold는 미확정임",
        ],
    }
    (output / "roi_margin_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "roi_margin_report.txt").write_text(_render_report(summary), encoding="utf-8")
    guide = """# STEP 2-C Manual Review Guide

청록색은 변경하지 않은 YuNet detection bbox이고 녹색은 Dlib68 fitting ROI입니다.
RAW, M05, M10, M15의 눈·입·전체 landmark와 pose axis를 직접 비교하십시오.
`preferred_landmark_roi`에는 RAW, M05, M10, M15, TIE, UNCERTAIN 중 하나를 기록하십시오.
HOG 값이나 자동 수치를 ground truth로 취급하지 말고 최종 margin은 수동 검토 후 결정하십시오.
"""
    (output / "MANUAL_REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    return summary
