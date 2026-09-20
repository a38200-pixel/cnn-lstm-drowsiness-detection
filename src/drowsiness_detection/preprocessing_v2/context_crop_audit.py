"""저장된 YuNet bbox에서 Context CNN용 얼굴 crop 후보만 평가한다."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .detector_primary_comparison import _distribution, _read_exact_frame, load_comparison_config, load_step2a_samples
from .detectors import BoundingBox


DECISION = "WAITING_FOR_MANUAL_CONTEXT_CROP_REVIEW"
CANDIDATES = ("RAW_RESIZE", "SQUARE_0", "SQUARE_M10", "SQUARE_M20")
SQUARE_MARGINS = {"SQUARE_0": 0.0, "SQUARE_M10": 0.10, "SQUARE_M20": 0.20}
MANUAL_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec", "selection_reasons",
    "comparison_image_path", "raw_resize_face_complete", "square0_face_complete",
    "square10_face_complete", "square20_face_complete", "raw_resize_distortion_ok",
    "square0_context_ok", "square10_context_ok", "square20_context_ok",
    "square0_padding_ok", "square10_padding_ok", "square20_padding_ok",
    "preferred_context_crop", "review_decision", "review_note",
)
BASE_COLUMNS = (
    "sample_id", "video_id", "split", "label", "timestamp_sec", "source_frame_index",
    "frame_width", "frame_height", "yunet_bbox_x1", "yunet_bbox_y1", "yunet_bbox_x2",
    "yunet_bbox_y2", "yunet_bbox_width", "yunet_bbox_height", "yunet_bbox_aspect_ratio",
    "raw_crop_width", "raw_crop_height", "raw_distortion_ratio",
    "raw_padding_left", "raw_padding_top", "raw_padding_right", "raw_padding_bottom",
    "raw_padding_fraction", "raw_padding_required", "raw_face_area_fraction",
)
SQUARE_FIELDS = (
    "x1", "y1", "x2", "y2", "side", "padding_left", "padding_top", "padding_right",
    "padding_bottom", "padding_fraction", "padding_required", "face_area_fraction",
    "face_width_fraction", "face_height_fraction",
)
RESULT_COLUMNS = BASE_COLUMNS + tuple(
    f"{prefix}_{field}" for prefix in ("square0", "square10", "square20") for field in SQUARE_FIELDS
)


@dataclass(frozen=True)
class ContextCropConfig:
    """STEP 2-D 설정과 프로젝트 기준 경로를 보관한다."""

    project_root: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        """프로젝트 기준으로 상대 경로를 해석한다."""

        path = Path(self.values[key])
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True)
class CropGeometry:
    """frame 밖 영역도 유지한 반개방 정수 crop ROI와 padding 메타데이터다."""

    roi: BoundingBox
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    padding_fraction: float
    face_area_fraction: float
    face_width_fraction: float
    face_height_fraction: float

    @property
    def padding_required(self) -> bool:
        """frame 밖 pixel을 채워야 하는지 반환한다."""

        return any((self.padding_left, self.padding_top, self.padding_right, self.padding_bottom))


def load_context_crop_config(path: Path, project_root: Path) -> ContextCropConfig:
    """후보·padding·출력 크기 설정을 검증해 읽는다."""

    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    required = {"source_results_dir", "step2b_config", "train_metadata", "val_metadata", "output_size", "candidates", "padding", "visual_audit", "output_dir"}
    if not isinstance(values, dict) or required - values.keys():
        raise ValueError(f"STEP 2-D 설정 필수 항목 누락: {sorted(required - set(values or {}))}")
    if "test.csv" in f"{values['train_metadata']} {values['val_metadata']}".casefold():
        raise ValueError("test split은 사용할 수 없습니다")
    if int(values["output_size"]) != 224:
        raise ValueError("audit preview는 RGB 224×224여야 합니다")
    candidates = values["candidates"]
    if candidates.get("raw_resize") is not True or {key: float(candidates[key]["margin_ratio"]) for key in ("square_0", "square_m10", "square_m20")} != {"square_0": 0.0, "square_m10": 0.10, "square_m20": 0.20}:
        raise ValueError("crop 후보는 RAW_RESIZE/SQUARE_0/M10/M20으로 고정합니다")
    if values["padding"]["mode"] != "imagenet_mean" or list(values["padding"]["rgb_mean"]) != [0.485, 0.456, 0.406]:
        raise ValueError("ImageNet RGB mean padding 정책이 필요합니다")
    return ContextCropConfig(project_root.resolve(), values)


def square_crop_roi(bbox: BoundingBox, margin_ratio: float) -> BoundingBox:
    """bbox 각 축에 margin을 더한 후 긴 축을 줄이지 않고 중심 정사각형으로 확장한다."""

    if not bbox.valid or margin_ratio < 0:
        raise ValueError("유효한 bbox와 음수가 아닌 margin이 필요합니다")
    width = bbox.width * (1.0 + 2.0 * margin_ratio)
    height = bbox.height * (1.0 + 2.0 * margin_ratio)
    side = int(np.ceil(max(width, height)))
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    x1, y1 = int(np.floor(center_x - side / 2.0)), int(np.floor(center_y - side / 2.0))
    roi = BoundingBox(x1, y1, x1 + side, y1 + side)
    if roi.x1 > bbox.x1 or roi.y1 > bbox.y1 or roi.x2 < bbox.x2 or roi.y2 < bbox.y2:
        raise AssertionError("square ROI가 검출 얼굴 bbox를 줄였습니다")
    return roi


def crop_geometry(bbox: BoundingBox, roi: BoundingBox, frame_width: int, frame_height: int) -> CropGeometry:
    """의도한 ROI는 보존하고 frame 밖 면적을 padding으로 계산한다."""

    if not bbox.valid or not roi.valid or frame_width <= 0 or frame_height <= 0:
        raise ValueError("유효한 bbox/ROI/frame 크기가 필요합니다")
    left, top = max(0, -roi.x1), max(0, -roi.y1)
    right, bottom = max(0, roi.x2 - frame_width), max(0, roi.y2 - frame_height)
    visible_width = max(0, min(frame_width, roi.x2) - max(0, roi.x1))
    visible_height = max(0, min(frame_height, roi.y2) - max(0, roi.y1))
    area = roi.area
    return CropGeometry(
        roi, left, top, right, bottom, 1.0 - visible_width * visible_height / area,
        bbox.area / area, bbox.width / roi.width, bbox.height / roi.height,
    )


def imagenet_padding_bgr(rgb_mean: Sequence[float] = (0.485, 0.456, 0.406)) -> tuple[int, int, int]:
    """ImageNet RGB mean을 8-bit로 반올림한 뒤 OpenCV BGR 순서로 바꾼다."""

    rgb = np.rint(np.asarray(rgb_mean, dtype=np.float64) * 255).astype(np.uint8)
    return tuple(int(value) for value in rgb[::-1])


def resize_interpolation(source_width: int, source_height: int, output_size: int) -> int:
    """양 축 모두 축소면 AREA, 한 축이라도 확대면 LINEAR를 선택한다."""

    return cv2.INTER_AREA if source_width >= output_size and source_height >= output_size else cv2.INTER_LINEAR


def crop_preview(frame_bgr: np.ndarray, geometry: CropGeometry, output_size: int, padding_bgr: tuple[int, int, int]) -> np.ndarray:
    """ROI를 줄이지 않고 frame 교집합을 복사·padding한 뒤 RGB로 변환한다."""

    roi = geometry.roi
    canvas = np.empty((roi.height, roi.width, 3), dtype=np.uint8)
    canvas[:] = padding_bgr
    frame_height, frame_width = frame_bgr.shape[:2]
    x1, y1 = max(0, roi.x1), max(0, roi.y1)
    x2, y2 = min(frame_width, roi.x2), min(frame_height, roi.y2)
    if x2 > x1 and y2 > y1:
        canvas[y1 - roi.y1:y2 - roi.y1, x1 - roi.x1:x2 - roi.x1] = frame_bgr[y1:y2, x1:x2]
    resized = cv2.resize(canvas, (output_size, output_size), interpolation=resize_interpolation(roi.width, roi.height, output_size))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def raw_distortion_ratio(bbox: BoundingBox) -> float:
    """비정사각형 RAW bbox를 1:1로 resize할 때 축 비율 변형 배수를 계산한다."""

    if not bbox.valid:
        raise ValueError("유효한 bbox가 필요합니다")
    return max(bbox.width / bbox.height, bbox.height / bbox.width)


def _source_rows(config: ContextCropConfig) -> list[dict[str, str]]:
    """STEP 2-B 저장 결과를 읽고 동일 train/val frame만 허용한다."""

    with (config.path("source_results_dir") / "paired_primary_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 160 or len({row["sample_id"] for row in rows}) != 160 or any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("중복 없는 STEP 2-B train/val 160개 결과가 필요합니다")
    return rows


def _dimensions(config: ContextCropConfig) -> dict[tuple[str, str], tuple[int, int]]:
    """train/val metadata에서 영상 frame 크기만 읽는다."""

    output: dict[tuple[str, str], tuple[int, int]] = {}
    for split, key in (("train", "train_metadata"), ("val", "val_metadata")):
        with config.path(key).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                output[(split, row["video_id"])] = int(row["width"]), int(row["height"])
    return output


def describe_context_crop_dry_run(config: ContextCropConfig) -> dict[str, Any]:
    """영상 decode 없이 저장 bbox와 처리 범위만 확인한다."""

    source = _source_rows(config)
    usable = [row for row in source if row["yunet_success"].casefold() == "true"]
    return {
        "mode": "DRY_RUN", "source_frame_count": len(source), "yunet_success_frame_count": len(usable),
        "unique_video_count": len({row["video_id"] for row in usable}),
        "split_counts": dict(Counter(row["split"] for row in usable)),
        "test_split_used": False, "new_random_sampling": False,
        "detector_rerun": False, "landmark_rerun": False,
        "stored_yunet_bbox_reused": True, "candidates": list(CANDIDATES),
        "output_dir": str(config.path("output_dir")), "decision_after_run": DECISION,
    }


def _record(row: Mapping[str, str], frame_width: int, frame_height: int) -> dict[str, Any]:
    """한 YuNet bbox에서 네 후보의 geometry diagnostic을 계산한다."""

    bbox = BoundingBox(*(int(float(row[f"yunet_bbox_{axis}"])) for axis in ("x1", "y1", "x2", "y2")))
    if not bbox.valid:
        raise ValueError(f"유효하지 않은 YuNet bbox: {row['sample_id']}")
    output: dict[str, Any] = {
        "sample_id": row["sample_id"], "video_id": row["video_id"], "split": row["split"],
        "label": row["label"], "timestamp_sec": row["timestamp_sec"],
        "source_frame_index": row["source_frame_index"], "frame_width": frame_width,
        "frame_height": frame_height,
        **{f"yunet_bbox_{axis}": value for axis, value in zip(("x1", "y1", "x2", "y2"), bbox.as_tuple())},
        "yunet_bbox_width": bbox.width, "yunet_bbox_height": bbox.height,
        "yunet_bbox_aspect_ratio": bbox.width / bbox.height,
        "raw_crop_width": bbox.width, "raw_crop_height": bbox.height,
        "raw_distortion_ratio": raw_distortion_ratio(bbox),
    }
    raw_geometry = crop_geometry(bbox, bbox, frame_width, frame_height)
    output.update({
        "raw_padding_left": raw_geometry.padding_left,
        "raw_padding_top": raw_geometry.padding_top,
        "raw_padding_right": raw_geometry.padding_right,
        "raw_padding_bottom": raw_geometry.padding_bottom,
        "raw_padding_fraction": raw_geometry.padding_fraction,
        "raw_padding_required": raw_geometry.padding_required,
        "raw_face_area_fraction": raw_geometry.face_area_fraction,
    })
    for candidate, margin in SQUARE_MARGINS.items():
        geometry = crop_geometry(bbox, square_crop_roi(bbox, margin), frame_width, frame_height)
        prefix = {"SQUARE_0": "square0", "SQUARE_M10": "square10", "SQUARE_M20": "square20"}[candidate]
        output.update({f"{prefix}_{axis}": value for axis, value in zip(("x1", "y1", "x2", "y2"), geometry.roi.as_tuple())})
        output.update({
            f"{prefix}_side": geometry.roi.width,
            f"{prefix}_padding_left": geometry.padding_left,
            f"{prefix}_padding_top": geometry.padding_top,
            f"{prefix}_padding_right": geometry.padding_right,
            f"{prefix}_padding_bottom": geometry.padding_bottom,
            f"{prefix}_padding_fraction": geometry.padding_fraction,
            f"{prefix}_padding_required": geometry.padding_required,
            f"{prefix}_face_area_fraction": geometry.face_area_fraction,
            f"{prefix}_face_width_fraction": geometry.face_width_fraction,
            f"{prefix}_face_height_fraction": geometry.face_height_fraction,
        })
    return output


def _select_visual_rows(config: ContextCropConfig, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """기존 STEP 2-B review 후보에서 YuNet 성공 표본만 순서대로 재사용한다."""

    by_id = {str(row["sample_id"]): row for row in records}
    path = Path(config.values["visual_audit"]["source_manifest"])
    if not path.is_absolute():
        path = config.project_root / path
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            if item["sample_id"] in by_id and item["sample_id"] not in seen:
                selected.append({"record": by_id[item["sample_id"]], "reason": item["selection_reasons"]})
                seen.add(item["sample_id"])
    maximum = int(config.values["visual_audit"]["max_samples"])
    if not 35 <= len(selected) <= maximum:
        raise ValueError(f"기존 review 대상이 35~{maximum}개 범위가 아닙니다: {len(selected)}")
    return selected


def _visual_image(frame: np.ndarray, bbox: BoundingBox, record: Mapping[str, Any], panel_size: int, output_size: int, padding_bgr: tuple[int, int, int]) -> np.ndarray:
    """원본 bbox와 네 RGB crop을 5-panel로 표시한다."""

    original = frame.copy()
    cv2.rectangle(original, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 255, 0), 2)
    scale = min(panel_size / original.shape[1], panel_size / original.shape[0])
    preview_width = max(1, int(round(original.shape[1] * scale)))
    preview_height = max(1, int(round(original.shape[0] * scale)))
    resized_original = cv2.resize(
        original, (preview_width, preview_height),
        interpolation=resize_interpolation(original.shape[1], original.shape[0], max(preview_width, preview_height)),
    )
    preview = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    offset_x = (panel_size - preview_width) // 2
    offset_y = (panel_size - preview_height) // 2
    preview[offset_y:offset_y + preview_height, offset_x:offset_x + preview_width] = resized_original
    panels = [("ORIGINAL / YUNET", preview)]
    for candidate, prefix in (("RAW_RESIZE", "raw"), ("SQUARE_0", "square0"), ("SQUARE_M10", "square10"), ("SQUARE_M20", "square20")):
        roi = bbox if prefix == "raw" else BoundingBox(*(int(record[f"{prefix}_{axis}"]) for axis in ("x1", "y1", "x2", "y2")))
        geometry = crop_geometry(bbox, roi, frame.shape[1], frame.shape[0])
        rgb = crop_preview(frame, geometry, output_size, padding_bgr)
        if panel_size != output_size:
            rgb = cv2.resize(rgb, (panel_size, panel_size), interpolation=resize_interpolation(output_size, output_size, panel_size))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        title = f"{candidate} pad={geometry.padding_fraction:.3f} face={geometry.face_area_fraction:.3f}"
        panels.append((title, image))
    labeled: list[np.ndarray] = []
    for title, panel in panels:
        tile = np.zeros((panel_size + 40, panel_size, 3), dtype=np.uint8)
        tile[40:, :] = panel
        cv2.putText(tile, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(tile)
    return np.hstack(labeled)


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """고정 열 순서로 UTF-8 CSV를 저장한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _temporal(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """동일 영상의 인접 audit timestamp에서 얼굴 점유율 변화만 진단한다."""

    videos: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        videos[str(row["video_id"])].append(row)
    result: dict[str, Any] = {}
    for prefix in ("square0", "square10", "square20"):
        values: list[float] = []
        for rows in videos.values():
            ordered = sorted(rows, key=lambda row: float(row["timestamp_sec"]))
            values.extend(abs(float(a[f"{prefix}_face_area_fraction"]) - float(b[f"{prefix}_face_area_fraction"])) for a, b in zip(ordered, ordered[1:]))
        result[prefix] = _distribution(values)
    return result


def _summary(records: Sequence[Mapping[str, Any]], visual_count: int, sheet_count: int) -> dict[str, Any]:
    """수치만으로 crop 우열을 선언하지 않는 audit 요약을 만든다."""

    candidate_stats: dict[str, Any] = {
        "raw_resize": {
            "distortion_ratio": _distribution([row["raw_distortion_ratio"] for row in records]),
            "padding_fraction": _distribution([row["raw_padding_fraction"] for row in records]),
            "padding_required_count": sum(bool(row["raw_padding_required"]) for row in records),
        },
    }
    for prefix in ("square0", "square10", "square20"):
        candidate_stats[prefix] = {
            "side": _distribution([row[f"{prefix}_side"] for row in records]),
            "face_area_fraction": _distribution([row[f"{prefix}_face_area_fraction"] for row in records]),
            "padding_fraction": _distribution([row[f"{prefix}_padding_fraction"] for row in records]),
            "padding_required_count": sum(bool(row[f"{prefix}_padding_required"]) for row in records),
        }
    return {
        "scope": {"source_frames": 160, "yunet_success_frames": len(records),
                  "unique_videos": len({row["video_id"] for row in records}),
                  "split_counts": dict(Counter(str(row["split"]) for row in records)),
                  "test_split_used": False, "detector_rerun": False, "landmark_rerun": False},
        "candidates": candidate_stats, "temporal_face_area_change": _temporal(records),
        "visual_review": {"sample_count": visual_count, "contact_sheet_count": sheet_count},
        "decision": DECISION, "automatic_best_crop": None,
    }


def _report(summary: Mapping[str, Any]) -> str:
    """crop 정의, 통계와 미확정 결정을 한국어 보고서로 기록한다."""

    lines = ["STEP 2-D Context CNN Crop Policy Audit", "", "[Scope]",
             f"YuNet success: {summary['scope']['yunet_success_frames']}/{summary['scope']['source_frames']}",
             f"Train/val: {summary['scope']['split_counts']}", "Test split 미사용; detector/Dlib68 재실행 없음.",
             "", "[Candidates]", "RAW_RESIZE: 원본 bbox를 RGB 224×224로 바로 resize.",
             "SQUARE_0: 긴 축을 줄이지 않고 중심 기준 정사각형으로 확장.",
             "SQUARE_M10/M20: bbox width와 height 각각 양쪽 10%/20% margin을 더한 뒤 정사각형 확장.",
             "", "[Padding & Resize]", "frame 밖 intended ROI는 줄이지 않고 ImageNet RGB mean [124,116,104]로 padding.",
             "OpenCV BGR canvas padding은 [104,116,124]. 양 축 축소는 INTER_AREA, 확대가 있으면 INTER_LINEAR.",
             "", "[Diagnostics]", "RAW distortion_ratio=max(width/height,height/width); 1은 변형 없음."]
    for name, item in summary["candidates"].items():
        lines.append(f"{name}: {item}")
    lines.extend(["", "[Temporal Consistency]", str(summary["temporal_face_area_change"]),
                  "실제 얼굴 움직임이 있으므로 낮은 변동이 반드시 더 좋은 crop은 아님.",
                  "", "[Visual Review]", f"{summary['visual_review']['sample_count']} samples, {summary['visual_review']['contact_sheet_count']} sheets",
                  "원본 얼굴 bbox의 완전성, 배경, padding, RAW 왜곡과 pose별 일관성을 수동 확인.",
                  "", "[Decision]", DECISION, "자동 best crop 선택 없음. CNN 학습 전 수동 검토 필요."])
    return "\n".join(lines) + "\n"


def run_context_crop_audit(config: ContextCropConfig) -> dict[str, Any]:
    """YuNet 저장 bbox로만 진단을 계산하고 선택 표본의 crop preview를 만든다."""

    output = config.path("output_dir")
    if output.exists():
        raise FileExistsError(f"기존 STEP 2-D 출력을 덮어쓰지 않습니다: {output}")
    source = [row for row in _source_rows(config) if row["yunet_success"].casefold() == "true"]
    dimensions = _dimensions(config)
    records = [_record(row, *dimensions[(row["split"], row["video_id"])]) for row in source]
    selected = _select_visual_rows(config, records)
    selected_ids = {str(item["record"]["sample_id"]) for item in selected}
    comparison = load_comparison_config(Path(config.values["step2b_config"]), config.project_root)
    samples = {sample.sample_id: sample for sample in load_step2a_samples(comparison)}
    output.mkdir(parents=True)
    _write_csv(output / "context_crop_results.csv", RESULT_COLUMNS, records)
    images_dir, sheets_dir = output / "visual_samples", output / "contact_sheets"
    images_dir.mkdir()
    sheets_dir.mkdir()
    image_paths: list[Path] = []
    padding_bgr = imagenet_padding_bgr(config.values["padding"]["rgb_mean"])
    panel_size = int(config.values["visual_audit"]["panel_size"])
    for item in selected:
        record = item["record"]
        sample_id = str(record["sample_id"])
        frame, failure = _read_exact_frame(samples[sample_id])
        if frame is None:
            raise RuntimeError(f"{sample_id} frame decode 실패: {failure}")
        if frame.shape[1] != int(record["frame_width"]) or frame.shape[0] != int(record["frame_height"]):
            raise ValueError(f"metadata와 실제 frame 크기 불일치: {sample_id}")
        bbox = BoundingBox(*(int(record[f"yunet_bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2")))
        image = _visual_image(frame, bbox, record, panel_size, int(config.values["output_size"]), padding_bgr)
        path = images_dir / f"{sample_id}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"visual sample 저장 실패: {path}")
        image_paths.append(path)
    rows_per_sheet = int(config.values["visual_audit"]["contact_sheet_rows"])
    sheets: list[Path] = []
    for start in range(0, len(image_paths), rows_per_sheet):
        images = [cv2.imread(str(path)) for path in image_paths[start:start + rows_per_sheet]]
        if any(image is None for image in images):
            raise OSError("contact sheet 입력 이미지 읽기 실패")
        sheet = sheets_dir / f"context_crop_{start // rows_per_sheet + 1:02d}.jpg"
        if not cv2.imwrite(str(sheet), np.vstack(images)):
            raise OSError(f"contact sheet 저장 실패: {sheet}")
        sheets.append(sheet)
    manual = []
    context = {"sample_id", "video_id", "split", "label", "timestamp_sec"}
    for item in selected:
        record = item["record"]
        row = {column: record.get(column, "") if column in context else "" for column in MANUAL_COLUMNS}
        row["selection_reasons"] = item["reason"]
        row["comparison_image_path"] = (images_dir / f"{record['sample_id']}.jpg").relative_to(config.project_root).as_posix()
        manual.append(row)
    _write_csv(output / "context_crop_manual_review.csv", MANUAL_COLUMNS, manual)
    summary = _summary(records, len(selected_ids), len(sheets))
    (output / "context_crop_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "context_crop_report.txt").write_text(_report(summary), encoding="utf-8")
    (output / "MANUAL_REVIEW_GUIDE.md").write_text(
        "# STEP 2-D 수동 검토\n\nRAW_RESIZE 왜곡, 정사각형 후보의 얼굴 완전성·배경·padding을 비교합니다. "
        "`preferred_context_crop`에는 RAW_RESIZE, SQUARE_0, SQUARE_M10, SQUARE_M20, TIE, UNCERTAIN 중 하나를 기록합니다. "
        "수치나 label만으로 best crop을 정하지 않습니다.\n", encoding="utf-8",
    )
    return summary
