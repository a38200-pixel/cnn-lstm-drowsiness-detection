"""STEP 2-B 수동 검토를 위한 중복 제거 visual review pack을 생성한다."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .detector_primary_comparison import (
    ComparisonConfig,
    SourceSample,
    _compare_sample,
    _comparison_image,
    _read_exact_frame,
    load_comparison_config,
    load_step2a_samples,
)
from .detectors import HogFaceDetector, YuNetFaceDetector
from .landmarks import Dlib68Predictor


METRICS = (
    ("normalized_landmark_diff", "LARGE_LANDMARK_DIFF"),
    ("ear_abs_diff", "LARGE_EAR_DIFF"),
    ("mar_abs_diff", "LARGE_MAR_DIFF"),
    ("pitch_circular_diff", "LARGE_PITCH_DIFF"),
    ("yaw_circular_diff", "LARGE_YAW_DIFF"),
    ("roll_circular_diff", "LARGE_ROLL_DIFF"),
    ("crop_mean_abs_diff", "LARGE_CROP_DIFF"),
    ("hog_crop_padding_fraction", "HOG_CROP_PADDING"),
    ("yunet_crop_padding_fraction", "YUNET_CROP_PADDING"),
)
MANUAL_COLUMNS = (
    "review_order", "sample_id", "video_id", "timestamp_sec", "comparison_image_path",
    "selection_reasons", "hog_bbox_ok", "yunet_bbox_ok", "hog_crop_ok", "yunet_crop_ok",
    "hog_landmarks_ok", "yunet_landmarks_ok", "hog_eyes_ok", "yunet_eyes_ok",
    "hog_mouth_ok", "yunet_mouth_ok", "hog_pose_ok", "yunet_pose_ok",
    "preferred_detector", "review_decision", "review_note",
)
MANIFEST_COLUMNS = (
    "review_order", "sample_id", "video_id", "split", "label", "timestamp_sec",
    "hog_success", "yunet_success", "bbox_iou", "normalized_landmark_diff",
    "ear_hog", "ear_yunet", "ear_abs_diff", "mar_hog", "mar_yunet", "mar_abs_diff",
    "pitch_circular_diff", "yaw_circular_diff", "roll_circular_diff",
    "hog_crop_padding_fraction", "yunet_crop_padding_fraction", "crop_mean_abs_diff",
    "selection_reasons", "comparison_image_path", "review_sheet",
)


@dataclass
class SelectedSample:
    """선정 이유를 모두 보존하는 review sample."""

    row: dict[str, str]
    reasons: set[str] = field(default_factory=set)
    group: str = ""
    image_path: Path | None = None
    review_sheet: str = ""


def _float(row: Mapping[str, str], key: str) -> float | None:
    """빈 CSV 숫자를 None으로 안전하게 변환한다."""

    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except ValueError:
        return None


def _boolean(row: Mapping[str, str], key: str) -> bool:
    """CSV boolean 문자열을 해석한다."""

    return row.get(key, "").strip().casefold() == "true"


def _outcome(row: Mapping[str, str]) -> str:
    """HOG/YuNet 성공 조합을 반환한다."""

    hog, yunet = _boolean(row, "hog_success"), _boolean(row, "yunet_success")
    if hog and yunet:
        return "BOTH_SUCCESS"
    if hog:
        return "HOG_ONLY"
    if yunet:
        return "YUNET_ONLY"
    return "BOTH_FAILED"


def _top_rows(rows: Sequence[dict[str, str]], metric: str, count: int) -> list[dict[str, str]]:
    """값이 존재하는 sample을 metric 내림차순으로 결정론적으로 정렬한다."""

    valid = [row for row in rows if _float(row, metric) is not None]
    return sorted(valid, key=lambda row: (-float(_float(row, metric)), row["sample_id"]))[:count]  # type: ignore[arg-type]


def _normal_candidates(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """차이가 중앙값 이하이고 padding이 없는 normal reference 후보를 만든다."""

    keys = ("normalized_landmark_diff", "ear_abs_diff", "mar_abs_diff", "pitch_circular_diff", "yaw_circular_diff", "roll_circular_diff")
    medians = {
        key: float(np.median([value for row in rows if (value := _float(row, key)) is not None]))
        for key in keys
    }
    candidates = []
    for row in rows:
        if _outcome(row) != "BOTH_SUCCESS":
            continue
        if any((_float(row, key) is None or float(_float(row, key)) > medians[key]) for key in keys):
            continue
        if float(_float(row, "hog_crop_padding_fraction") or 0.0) > 0.0:
            continue
        if float(_float(row, "yunet_crop_padding_fraction") or 0.0) > 0.0:
            continue
        candidates.append(row)
    return sorted(candidates, key=lambda row: (row["split"], row["label"], row["sample_id"]))


def select_review_samples(
    rows: Sequence[dict[str, str]], top_count: int = 8,
    maximum_total: int = 45, normal_target: int = 10,
) -> tuple[list[SelectedSample], dict[str, set[str]]]:
    """필수 outcome, metric extreme, 균형 normal을 우선순위에 따라 중복 제거한다."""

    selected: dict[str, SelectedSample] = {}
    outlier_reasons: dict[str, set[str]] = defaultdict(set)

    def add(row: dict[str, str], reason: str, group: str) -> None:
        item = selected.setdefault(row["sample_id"], SelectedSample(row=row, group=group))
        item.reasons.add(reason)
        if not item.group:
            item.group = group

    for row in rows:
        outcome = _outcome(row)
        if outcome != "BOTH_SUCCESS":
            add(row, outcome, "detector_failures")

    ranked: dict[str, list[dict[str, str]]] = {}
    for metric, reason in METRICS:
        ranked[reason] = _top_rows(rows, metric, top_count)
        for row in ranked[reason]:
            outlier_reasons[row["sample_id"]].add(reason)

    # Normal reference 10개를 위한 자리를 남기고 metric을 round-robin으로 추가한다.
    extreme_limit = maximum_total - normal_target
    for rank in range(top_count):
        for _metric, reason in METRICS:
            row = ranked[reason][rank] if rank < len(ranked[reason]) else None
            if row is None:
                continue
            if row["sample_id"] not in selected and len(selected) >= extreme_limit:
                continue
            group = "landmark_ear_mar" if reason in {"LARGE_LANDMARK_DIFF", "LARGE_EAR_DIFF", "LARGE_MAR_DIFF"} else "pose_crop"
            add(row, reason, group)

    # 여러 metric의 Top 8에 속한 이유는 최종 manifest에 모두 병합한다.
    for sample_id, reasons in outlier_reasons.items():
        if sample_id in selected:
            selected[sample_id].reasons.update(reasons)

    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _normal_candidates(rows):
        strata[(row["split"], row["label"])].append(row)
    keys = sorted(strata)
    index = 0
    normal_added = 0
    while normal_added < normal_target and len(selected) < maximum_total:
        progressed = False
        for key in keys:
            if index < len(strata[key]) and normal_added < normal_target:
                row = strata[key][index]
                if row["sample_id"] not in selected:
                    add(row, "NORMAL_REFERENCE", "normal_reference")
                    normal_added += 1
                progressed = True
        if not progressed:
            break
        index += 1

    order = {"detector_failures": 0, "landmark_ear_mar": 1, "pose_crop": 2, "normal_reference": 3}
    result = sorted(selected.values(), key=lambda item: (order[item.group], item.row["sample_id"]))
    return result, outlier_reasons


def _index_existing_images(root: Path) -> dict[str, Path]:
    """기존 STEP 2-B comparison image를 sample ID별로 한 번만 인덱싱한다."""

    priority = {"yunet_only": 0, "hog_only": 0, "both_failed": 0, "both_success": 1}
    images = sorted(
        (root / "visual_samples").rglob("*.jpg"),
        key=lambda path: (priority.get(path.parent.name, 2), path.as_posix()),
    )
    result: dict[str, Path] = {}
    for image in images:
        result.setdefault(image.stem, image)
    return result


def _regenerate_image(sample_id: str, config: ComparisonConfig, source: SourceSample) -> np.ndarray:
    """기존 이미지가 없는 경우에만 동일 STEP 2-B pipeline으로 comparison을 재생성한다."""

    frame, failure = _read_exact_frame(source)
    if frame is None:
        raise RuntimeError(f"{sample_id} frame 재생성 실패: {failure}")
    hog = HogFaceDetector(int(config.values["hog_upsample"]))
    yunet_config = config.step2a_values["yunet"]
    yunet = YuNetFaceDetector(
        str(config.path("yunet_model")), float(yunet_config["score_threshold"]),
        float(yunet_config["nms_threshold"]), int(yunet_config["top_k"]),
    )
    predictor = Dlib68Predictor(str(config.path("dlib_predictor")))
    _record, hog_path, yunet_path = _compare_sample(source, frame, hog, yunet, predictor, config)
    return _comparison_image(frame, source, hog_path, yunet_path)


def _padding_summary(rows: Sequence[dict[str, str]], key: str) -> dict[str, Any]:
    """detector별 crop padding 발생 수와 분포를 계산한다."""

    values = np.asarray(
        [value for row in rows if (value := _float(row, key)) is not None],
        dtype=np.float64,
    )
    return {
        "frame_count": int(values.size), "padding_frame_count": int(np.sum(values > 0.0)),
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "max": float(np.max(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """review artifact를 고정 컬럼 순서의 UTF-8 CSV로 기록한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_repeated_outliers(
    path: Path, rows: Sequence[dict[str, str]], outlier_reasons: Mapping[str, set[str]]
) -> list[dict[str, Any]]:
    """여러 timestamp에서 반복된 video별 Top-8 outlier flag를 집계한다."""

    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    output: list[dict[str, Any]] = []
    for video_id, video_rows in by_video.items():
        reasons = [outlier_reasons.get(row["sample_id"], set()) for row in video_rows]
        item = {
            "video_id": video_id, "sample_count": len(video_rows),
            "landmark_outlier_count": sum("LARGE_LANDMARK_DIFF" in value for value in reasons),
            "ear_outlier_count": sum("LARGE_EAR_DIFF" in value for value in reasons),
            "mar_outlier_count": sum("LARGE_MAR_DIFF" in value for value in reasons),
            "pose_outlier_count": sum(bool(value & {"LARGE_PITCH_DIFF", "LARGE_YAW_DIFF", "LARGE_ROLL_DIFF"}) for value in reasons),
            "crop_outlier_count": sum(bool(value & {"LARGE_CROP_DIFF", "HOG_CROP_PADDING", "YUNET_CROP_PADDING"}) for value in reasons),
        }
        item["total_outlier_flags"] = sum(len(value) for value in reasons)
        if item["total_outlier_flags"]:
            output.append(item)
    output.sort(key=lambda item: (-item["total_outlier_flags"], -item["sample_count"], item["video_id"]))
    columns = (
        "video_id", "sample_count", "landmark_outlier_count", "ear_outlier_count",
        "mar_outlier_count", "pose_outlier_count", "crop_outlier_count", "total_outlier_flags",
    )
    _write_csv(path, columns, output)
    return output


def _labeled_tile(image: np.ndarray, sample_id: str, reasons: str) -> np.ndarray:
    """고해상도 comparison image 위에 sample ID와 선정 이유를 표시한다."""

    resized = cv2.resize(image, (960, 240), interpolation=cv2.INTER_AREA)
    tile = np.zeros((290, 960, 3), dtype=np.uint8)
    tile[50:, :] = resized
    cv2.putText(tile, sample_id, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    text = reasons if len(reasons) <= 120 else reasons[:117] + "..."
    cv2.putText(tile, text, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 255), 1, cv2.LINE_AA)
    return tile


def _write_review_sheets(review_dir: Path, selected: Sequence[SelectedSample]) -> list[Path]:
    """그룹별 최대 12개 타일을 담은 고해상도 검토 시트를 만든다."""

    group_names = {
        "detector_failures": "01_detector_failures",
        "landmark_ear_mar": "02_landmark_ear_mar",
        "pose_crop": "03_pose_crop",
        "normal_reference": "04_normal_reference",
    }
    sheets: list[Path] = []
    for group, name in group_names.items():
        items = [item for item in selected if item.group == group and item.image_path is not None]
        for page_start in range(0, len(items), 12):
            page = items[page_start : page_start + 12]
            rows = (len(page) + 1) // 2
            canvas = np.zeros((rows * 290, 2 * 960, 3), dtype=np.uint8)
            for index, item in enumerate(page):
                image = cv2.imread(str(item.image_path))
                if image is None:
                    raise OSError(f"review image 읽기 실패: {item.image_path}")
                tile = _labeled_tile(image, item.row["sample_id"], ";".join(sorted(item.reasons)))
                row_index, column_index = divmod(index, 2)
                canvas[row_index * 290 : (row_index + 1) * 290, column_index * 960 : (column_index + 1) * 960] = tile
            suffix = f"_{page_start // 12 + 1:02d}" if len(items) > 12 else ""
            sheet_path = review_dir / f"review_sheet_{name}{suffix}.jpg"
            if not cv2.imwrite(str(sheet_path), canvas):
                raise OSError(f"review sheet 저장 실패: {sheet_path}")
            for item in page:
                item.review_sheet = sheet_path.name
            sheets.append(sheet_path)
    return sheets


def _numeric_summary(
    summary: Mapping[str, Any], selected: Sequence[SelectedSample],
    padding: Mapping[str, Any], repeated: Sequence[Mapping[str, Any]],
) -> str:
    """자동 수치를 요약하되 의미적 품질 판정을 유보하는 한국어 문서를 만든다."""

    detection = summary["detector_detection"]
    outcome = summary["outcome_matrix"]
    landmark = summary["landmark"]
    ear, mar, pose = summary["ear"], summary["mar"], summary["head_pose"]
    latency = summary["latency"]
    reason_counts = Counter(reason for item in selected for reason in item.reasons)
    repeated_names = ", ".join(item["video_id"] for item in repeated[:10]) or "없음"
    return f"""# STEP 2-B Visual Review Numeric Summary

이 문서는 visual sample 선정을 위한 자동 metric 요약입니다. 이미지의 의미적 품질이나 primary detector를 자동 판정하지 않습니다.

## Detector 및 outcome

- HOG: {detection['hog']['success']}/{detection['hog']['attempted']} ({detection['hog']['success_rate']:.2%})
- YuNet: {detection['yunet']['success']}/{detection['yunet']['attempted']} ({detection['yunet']['success_rate']:.2%})
- Both success: {outcome['both_success']['count']}
- HOG only: {outcome['hog_only']['count']}
- YuNet only: {outcome['yunet_only']['count']}
- Both failed: {outcome['both_failed']['count']}

## Latency

- HOG detector: mean {latency['hog_detector_ms']['mean']:.2f} ms, median {latency['hog_detector_ms']['median']:.2f} ms, P95 {latency['hog_detector_ms']['p95']:.2f} ms
- YuNet detector: mean {latency['yunet_detector_ms']['mean']:.2f} ms, median {latency['yunet_detector_ms']['median']:.2f} ms, P95 {latency['yunet_detector_ms']['p95']:.2f} ms
- HOG behavior pipeline: mean {latency['hog_behavior_pipeline_ms']['mean']:.2f} ms
- YuNet behavior pipeline: mean {latency['yunet_behavior_pipeline_ms']['mean']:.2f} ms

## Landmark 및 EAR/MAR

- HOG+Dlib68: {landmark['hog']['success']}/{landmark['hog']['attempted']}
- YuNet+Dlib68: {landmark['yunet']['success']}/{landmark['yunet']['attempted']}
- Normalized landmark diff: median {landmark['normalized_difference']['median']:.6f}, P95 {landmark['normalized_difference']['p95']:.6f}, max {landmark['normalized_difference']['max']:.6f}
- EAR: Pearson {ear['correlation']['pearson']:.4f}, Spearman {ear['correlation']['spearman']:.4f}, abs diff median {ear['absolute_difference']['median']:.6f}, P95 {ear['absolute_difference']['p95']:.6f}
- MAR: Pearson {mar['correlation']['pearson']:.4f}, Spearman {mar['correlation']['spearman']:.4f}, abs diff median {mar['absolute_difference']['median']:.6f}, P95 {mar['absolute_difference']['p95']:.6f}

EAR/MAR correlation이 충분히 높지 않다는 사실은 어느 detector가 잘못됐다는 증거가 아닙니다. 동일 Dlib68 predictor라도 detector bbox가 landmark fitting과 파생 geometry에 영향을 줄 수 있다는 신호이므로 실제 눈·입 위치를 visual review해야 합니다.

## Circular pose difference

- Pitch: median {pose['pitch_circular_diff']['median']:.3f}°, P95 {pose['pitch_circular_diff']['p95']:.3f}°, max {pose['pitch_circular_diff']['max']:.3f}°
- Yaw: median {pose['yaw_circular_diff']['median']:.3f}°, P95 {pose['yaw_circular_diff']['p95']:.3f}°, max {pose['yaw_circular_diff']['max']:.3f}°
- Roll: median {pose['roll_circular_diff']['median']:.3f}°, P95 {pose['roll_circular_diff']['p95']:.3f}°, max {pose['roll_circular_diff']['max']:.3f}°

## Crop padding

- HOG: {padding['hog']['padding_frame_count']}/{padding['hog']['frame_count']}, mean {padding['hog']['mean']:.6f}, median {padding['hog']['median']:.6f}, max {padding['hog']['max']:.6f}, P90 {padding['hog']['p90']:.6f}, P95 {padding['hog']['p95']:.6f}
- YuNet: {padding['yunet']['padding_frame_count']}/{padding['yunet']['frame_count']}, mean {padding['yunet']['mean']:.6f}, median {padding['yunet']['median']:.6f}, max {padding['yunet']['max']:.6f}, P90 {padding['yunet']['p90']:.6f}, P95 {padding['yunet']['p95']:.6f}

Padding 빈도와 크기는 숫자로만 보고하며 품질의 좋고 나쁨을 자동 판정하지 않습니다.

## Review pack

- 최종 sample: {len(selected)}
- Normal reference: {reason_counts['NORMAL_REFERENCE']}
- 반복 outlier 상위 video: {repeated_names}
- Decision: `WAITING_FOR_MANUAL_PRIMARY_DETECTOR_REVIEW`
"""


def _relative(path: Path, project_root: Path) -> str:
    """프로젝트 루트를 기준으로 이식 가능한 POSIX 경로를 만든다."""

    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _manifest_rows(selected: Sequence[SelectedSample], project_root: Path) -> list[dict[str, Any]]:
    """선택된 표본과 병합된 선정 사유를 manifest 행으로 변환한다."""

    output: list[dict[str, Any]] = []
    for order, item in enumerate(selected, start=1):
        row: dict[str, Any] = dict(item.row)
        row.update(
            review_order=order,
            selection_reasons=";".join(sorted(item.reasons)),
            comparison_image_path=_relative(item.image_path, project_root) if item.image_path else "",
            review_sheet=item.review_sheet,
        )
        output.append(row)
    return output


def _manual_rows(manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """자동 판정 없이 사람이 직접 채울 빈 검토 양식을 만든다."""

    context = {
        "review_order", "sample_id", "video_id", "timestamp_sec",
        "comparison_image_path", "selection_reasons",
    }
    return [
        {column: row.get(column, "") if column in context else "" for column in MANUAL_COLUMNS}
        for row in manifest
    ]


def _create_zip(review_dir: Path, zip_path: Path) -> None:
    """검토 폴더 전체를 같은 디렉터리 구조를 유지한 ZIP으로 묶는다."""

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(review_dir.parent))


def build_review_pack(
    project_root: Path,
    config_path: Path,
    top_count: int = 8,
    maximum_total: int = 45,
    normal_target: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """완료된 STEP 2-B 수치에서 수동 시각 검토 팩을 구성한다."""

    project_root = project_root.resolve()
    config = load_comparison_config(config_path.resolve(), project_root)
    result_dir = config.path("output_dir")
    csv_path = result_dir / "paired_primary_results.csv"
    summary_path = result_dir / "comparison_summary.json"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected, outlier_reasons = select_review_samples(
        rows, top_count=top_count, maximum_total=maximum_total, normal_target=normal_target,
    )
    reason_counts = Counter(reason for item in selected for reason in item.reasons)
    outcome_counts = Counter(_outcome(item.row) for item in selected)
    metric_counts = {
        reason: sum(reason in item.reasons for item in selected) for _metric, reason in METRICS
    }
    report: dict[str, Any] = {
        "selected_count": len(selected),
        "outcome_counts": dict(outcome_counts),
        "metric_counts": metric_counts,
        "normal_reference_count": reason_counts["NORMAL_REFERENCE"],
        "review_dir": str(result_dir / "review_pack"),
        "zip_path": str(result_dir / "step2b_visual_review_pack.zip"),
    }
    if dry_run:
        return report

    review_dir = result_dir / "review_pack"
    zip_path = result_dir / "step2b_visual_review_pack.zip"
    if review_dir.exists() or zip_path.exists():
        raise FileExistsError(
            "기존 review pack을 덮어쓰지 않습니다. review_pack 폴더와 ZIP을 확인해 주세요."
        )
    selected_dir = review_dir / "selected_images"
    selected_dir.mkdir(parents=True)
    existing = _index_existing_images(result_dir)
    source_by_id: dict[str, SourceSample] | None = None
    regenerated = 0
    copied = 0
    for order, item in enumerate(selected, start=1):
        sample_id = item.row["sample_id"]
        destination = selected_dir / f"{order:02d}_{sample_id}.jpg"
        source_image = existing.get(sample_id)
        if source_image is not None:
            shutil.copy2(source_image, destination)
            copied += 1
        else:
            if source_by_id is None:
                source_by_id = {sample.sample_id: sample for sample in load_step2a_samples(config)}
            image = _regenerate_image(sample_id, config, source_by_id[sample_id])
            if not cv2.imwrite(str(destination), image):
                raise OSError(f"비교 이미지 저장 실패: {destination}")
            regenerated += 1
        item.image_path = destination

    sheets = _write_review_sheets(review_dir, selected)
    manifest = _manifest_rows(selected, project_root)
    _write_csv(review_dir / "review_manifest.csv", MANIFEST_COLUMNS, manifest)
    _write_csv(review_dir / "review_manual.csv", MANUAL_COLUMNS, _manual_rows(manifest))
    repeated = _write_repeated_outliers(
        review_dir / "repeated_outlier_videos.csv", rows, outlier_reasons,
    )
    padding = {
        "hog": _padding_summary(rows, "hog_crop_padding_fraction"),
        "yunet": _padding_summary(rows, "yunet_crop_padding_fraction"),
    }
    (review_dir / "review_numeric_summary.md").write_text(
        _numeric_summary(summary, selected, padding, repeated), encoding="utf-8",
    )
    _create_zip(review_dir, zip_path)
    report.update(
        copied_image_count=copied,
        regenerated_image_count=regenerated,
        sheet_count=len(sheets),
        repeated_outlier_video_count=len(repeated),
        repeated_outlier_top=[item["video_id"] for item in repeated[:10]],
        crop_padding=padding,
    )
    return report
