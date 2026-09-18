"""완료된 STEP 2-C artifact만 사용해 compact visual review pack을 만든다."""

from __future__ import annotations

import csv
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .angle_utils import circular_angle_difference


DECISION = "WAITING_FOR_MANUAL_LANDMARK_ROI_REVIEW"
SHEET_DEFINITIONS = (
    ("overview", "review_sheet_01_overview.jpg", "STEP 2-C - OVERVIEW / NORMAL REFERENCE"),
    ("eye", "review_sheet_02_eye_ear.jpg", "STEP 2-C - EYE / EAR REVIEW"),
    ("mouth", "review_sheet_03_mouth_mar.jpg", "STEP 2-C - MOUTH / MAR REVIEW"),
    ("pose", "review_sheet_04_pose_landmark.jpg", "STEP 2-C - HEAD POSE / LANDMARK CHANGE"),
    ("difficult", "review_sheet_05_difficult_cases.jpg", "STEP 2-C - ROI / EDGE / DIFFICULT CASES"),
)
MANIFEST_COLUMNS = (
    "review_order", "sheet_name", "sample_id", "video_id", "split", "label",
    "timestamp_sec", "selection_reason", "source_image_path",
    "raw_m05_ear_diff", "raw_m10_ear_diff", "raw_m15_ear_diff",
    "raw_m05_mar_diff", "raw_m10_mar_diff", "raw_m15_mar_diff",
    "raw_m05_landmark_diff", "raw_m10_landmark_diff", "raw_m15_landmark_diff",
    "raw_m05_pitch_diff", "raw_m10_pitch_diff", "raw_m15_pitch_diff",
    "raw_m05_yaw_diff", "raw_m10_yaw_diff", "raw_m15_yaw_diff",
    "raw_m05_roll_diff", "raw_m10_roll_diff", "raw_m15_roll_diff",
)
MANUAL_COLUMNS = (
    "review_order", "sheet_name", "sample_id", "video_id", "timestamp_sec",
    "source_image_path", "preferred_landmark_roi", "eye_comment", "mouth_comment",
    "pose_comment", "overall_comment",
)


@dataclass
class ReviewSample:
    """기존 35개 표본의 수치, 배정 sheet와 선정 사유를 보관한다."""

    row: dict[str, str]
    metrics: dict[str, float]
    group: str = ""
    reason: str = ""
    review_order: int = 0
    copied_image: Path | None = None


def _number(row: Mapping[str, str], key: str) -> float:
    """CSV의 유효한 수치를 float로 읽고 빈 값은 0으로 처리한다."""

    value = row.get(key, "").strip()
    try:
        number = float(value)
        return number if np.isfinite(number) else 0.0
    except ValueError:
        return 0.0


def _flag(row: Mapping[str, str], key: str) -> bool:
    """CSV boolean 문자열을 해석한다."""

    return row.get(key, "").strip().casefold() == "true"


def _feature_diff(row: Mapping[str, str], candidate: str, feature: str) -> float:
    """RAW와 한 margin 후보 사이 feature 차이를 계산한다."""

    left = _number(row, f"raw_{feature}")
    right = _number(row, f"{candidate}_{feature}")
    if feature in {"pitch_raw", "yaw", "roll"}:
        return circular_angle_difference(left, right)
    return abs(left - right)


def compute_review_metrics(row: Mapping[str, str]) -> dict[str, float]:
    """기존 CSV 열만으로 review 표본 선정용 변화량을 계산한다."""

    metrics: dict[str, float] = {}
    for candidate in ("m05", "m10", "m15"):
        metrics[f"raw_{candidate}_ear_diff"] = _feature_diff(row, candidate, "ear")
        metrics[f"raw_{candidate}_mar_diff"] = _feature_diff(row, candidate, "mar")
        metrics[f"raw_{candidate}_landmark_diff"] = _number(
            row, f"raw_{candidate}_normalized_landmark_diff"
        )
        for feature in ("pitch_raw", "yaw", "roll"):
            short = "pitch" if feature == "pitch_raw" else feature
            metrics[f"raw_{candidate}_{short}_diff"] = _feature_diff(row, candidate, feature)
    metrics["eye_score"] = max(
        metrics[f"raw_{candidate}_ear_diff"] for candidate in ("m05", "m10", "m15")
    ) + max(metrics[f"raw_{candidate}_landmark_diff"] for candidate in ("m05", "m10", "m15"))
    metrics["mouth_score"] = max(
        metrics[f"raw_{candidate}_mar_diff"] for candidate in ("m05", "m10", "m15")
    ) + max(metrics[f"raw_{candidate}_landmark_diff"] for candidate in ("m05", "m10", "m15"))
    metrics["pose_score"] = max(
        metrics[f"raw_{candidate}_{feature}_diff"]
        for candidate in ("m05", "m10", "m15") for feature in ("pitch", "yaw", "roll")
    ) + 20.0 * max(metrics[f"raw_{candidate}_landmark_diff"] for candidate in ("m05", "m10", "m15"))
    clipped_count = sum(_flag(row, f"{candidate}_roi_clipped") for candidate in ("m05", "m10", "m15"))
    clipped_fraction = max(_number(row, f"{candidate}_clipped_fraction") for candidate in ("m05", "m10", "m15"))
    metrics["difficult_score"] = (
        clipped_count + 20.0 * clipped_fraction
        + abs(_number(row, "raw_yaw")) / 45.0
        + max(0.0, 1.0 - _number(row, "yunet_confidence"))
    )
    return metrics


def _normalize(samples: Sequence[ReviewSample], key: str) -> dict[str, float]:
    """표본 집합 안에서 점수를 0~1 범위로 바꾼다."""

    values = np.asarray([sample.metrics[key] for sample in samples], dtype=np.float64)
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        return {sample.row["sample_id"]: 0.0 for sample in samples}
    return {
        sample.row["sample_id"]: float((sample.metrics[key] - low) / (high - low))
        for sample in samples
    }


def assign_review_groups(samples: Sequence[ReviewSample]) -> list[ReviewSample]:
    """35개를 중복 없이 다섯 sheet에 7개씩 배정한다."""

    if len(samples) != 35 or len({sample.row["sample_id"] for sample in samples}) != 35:
        raise ValueError("중복 없는 기존 STEP 2-C review 표본 35개가 필요합니다")
    by_id = {sample.row["sample_id"]: sample for sample in samples}
    normalized = {
        key: _normalize(samples, key)
        for key in ("eye_score", "mouth_score", "pose_score", "difficult_score")
    }
    # 정상 참조는 네 split/label 층을 순환하며 전체 변화량이 작은 표본을 고른다.
    strata: dict[tuple[str, str], list[ReviewSample]] = defaultdict(list)
    for sample in samples:
        strata[(sample.row["split"], sample.row["label"])].append(sample)
    for values in strata.values():
        values.sort(key=lambda sample: (
            sum(normalized[key][sample.row["sample_id"]] for key in normalized),
            sample.row["sample_id"],
        ))
    overview: list[ReviewSample] = []
    index = 0
    keys = sorted(strata)
    while len(overview) < 7:
        progressed = False
        for key in keys:
            if index < len(strata[key]) and len(overview) < 7:
                candidate = strata[key][index]
                if candidate not in overview:
                    overview.append(candidate)
                progressed = True
        if not progressed:
            break
        index += 1
    for sample in overview:
        sample.group = "overview"
        sample.reason = "LOW_MULTI_METRIC_CHANGE;BALANCED_NORMAL_REFERENCE"

    # 나머지는 category 점수가 가장 높은 조합부터 quota를 채워 극단 사례를 분산한다.
    remaining = [sample for sample in samples if not sample.group]
    capacities = {"eye": 7, "mouth": 7, "pose": 7, "difficult": 7}
    pairs = sorted(
        [
            normalized[f"{group}_score"][sample.row["sample_id"]],
            sample.row["sample_id"], group,
        ]
        for sample in remaining for group in capacities
    )
    pairs.reverse()
    assigned: set[str] = set()
    for _score, sample_id, group in pairs:
        if sample_id in assigned or capacities[group] <= 0:
            continue
        sample = by_id[sample_id]
        sample.group = group
        capacities[group] -= 1
        assigned.add(sample_id)
    if any(capacities.values()) or len(assigned) != 28:
        raise RuntimeError(f"review group quota 배정 실패: {capacities}")

    reason_keys = {
        "eye": ("EAR_CHANGE", "eye_score"),
        "mouth": ("MAR_CHANGE", "mouth_score"),
        "pose": ("POSE_OR_LANDMARK_CHANGE", "pose_score"),
        "difficult": ("ROI_EDGE_OR_DIFFICULT", "difficult_score"),
    }
    for sample in remaining:
        label, key = reason_keys[sample.group]
        sample.reason = f"{label};score={sample.metrics[key]:.6f};STEP2C_EXISTING_SELECTION"
    ordered: list[ReviewSample] = []
    order = {name: index for index, (name, _file, _title) in enumerate(SHEET_DEFINITIONS)}
    for sample in sorted(samples, key=lambda item: (order[item.group], -item.metrics.get(f"{item.group}_score", 0.0), item.row["sample_id"])):
        sample.review_order = len(ordered) + 1
        ordered.append(sample)
    return ordered


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """고정된 열 순서로 UTF-8 CSV를 기록한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fit_text(text: str, maximum: int = 180) -> str:
    """sheet label이 이미지 너비를 넘지 않도록 문자열을 제한한다."""

    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def _write_sheet(path: Path, title: str, samples: Sequence[ReviewSample]) -> None:
    """기존 5-panel 이미지를 확대·축소하지 않고 세로형 고해상도 sheet로 묶는다."""

    width, header_height, label_height = 2400, 112, 64
    blocks: list[np.ndarray] = []
    for sample in samples:
        if sample.copied_image is None:
            raise RuntimeError(f"복사 이미지가 없습니다: {sample.row['sample_id']}")
        image = cv2.imread(str(sample.copied_image))
        if image is None or image.shape[:2] != (360, width):
            raise OSError(f"예상하지 못한 5-panel 이미지: {sample.copied_image}")
        label = np.zeros((label_height, width, 3), dtype=np.uint8)
        first = f"#{sample.review_order:02d} | {sample.row['sample_id']} | video={sample.row['video_id']} | t={sample.row['timestamp_sec']}s"
        second = f"reason: {sample.reason}"
        cv2.putText(label, first, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(label, _fit_text(second), (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        blocks.extend((label, image))
    header = np.zeros((header_height, width, 3), dtype=np.uint8)
    cv2.putText(header, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "YuNet detection bbox fixed | Dlib68 ROI = RAW / +5% / +10% / +15%", (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f"samples={len(samples)} | metrics select review targets only; no automatic best margin", (18, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)
    canvas = np.vstack([header, *blocks])
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"review sheet 저장 실패: {path}")


def _manifest_row(sample: ReviewSample, sheet_name: str, project_root: Path) -> dict[str, Any]:
    """표본과 실제 계산된 차이 지표를 manifest 행으로 변환한다."""

    row: dict[str, Any] = {
        "review_order": sample.review_order, "sheet_name": sheet_name,
        "sample_id": sample.row["sample_id"], "video_id": sample.row["video_id"],
        "split": sample.row["split"], "label": sample.row["label"],
        "timestamp_sec": sample.row["timestamp_sec"], "selection_reason": sample.reason,
        "source_image_path": sample.copied_image.resolve().relative_to(project_root.resolve()).as_posix(),
    }
    for column in MANIFEST_COLUMNS:
        if column in sample.metrics:
            row[column] = sample.metrics[column]
    return row


def _readme_text(
    group_counts: Mapping[str, int], source_sheet_count: int, clipping_inconsistency_count: int,
) -> str:
    """검토 방법과 알려진 주의를 설명하는 한국어 README를 만든다."""

    return f"""# STEP 2-C Compact Visual Review Pack

## 목적

YuNet detection bbox를 고정한 상태에서 Dlib68 fitting ROI만 RAW, +5%, +10%, +15%로 변경했을 때 눈·입·윤곽과 pose axis가 어떻게 달라지는지 사람이 검토하기 위한 팩입니다.

- 기존 contact sheet: {source_sheet_count}장
- 기존 review sample: {sum(group_counts.values())}개
- 최종 중복 제거 sample: {sum(group_counts.values())}개
- 최종 compact sheet: {len(SHEET_DEFINITIONS)}장
- Decision: `{DECISION}`

## 후보 정의

- RAW: YuNet detection bbox 그대로 사용
- M05: bbox width와 height에 각각 5% 확장
- M10: 각각 10% 확장
- M15: 각각 15% 확장

모든 panel의 YuNet detection bbox는 동일하며 Dlib68 fitting ROI만 변경됩니다. 강제 square 변환은 사용하지 않았습니다.

## Sheet 목적

- Overview / Normal Reference ({group_counts['overview']}개): 변화가 작은 train/val, drowsy/not_drowsy 균형 참조
- Eye / EAR ({group_counts['eye']}개): 눈 landmark와 EAR 변화가 큰 사례
- Mouth / MAR ({group_counts['mouth']}개): 입 landmark와 MAR 변화가 큰 사례
- Head Pose / Landmark ({group_counts['pose']}개): circular pose 및 전체 landmark 변화가 큰 사례
- ROI / Edge / Difficult ({group_counts['difficult']}개): clipping flag, 큰 yaw, 낮은 confidence 등을 우선한 난례

자동 metric은 사람이 볼 표본을 분류하는 데만 사용했습니다. ground-truth landmark가 없으므로 RAW나 특정 margin을 best로 자동 결정하지 않습니다. `review_manual_compact.csv`의 입력 필드는 모두 비워 두었습니다.

## Known Issues

기존 audit에서 `roi_clipped` flag와 `clipped_fraction` 값의 일관성을 별도로 확인할 필요가 있습니다. YuNet 성공 {clipping_inconsistency_count}개 행에서 M05/M10/M15 중 하나 이상이 clipping flag=True이지만 기록된 fraction은 0.0입니다. 이 후처리에서는 기존 값을 수정하거나 재해석하지 않았습니다.

Landmark ROI margin과 Context CNN crop margin은 별도 정책입니다. 이 팩은 Dlib68 fitting ROI만 검토합니다.
"""


def build_review_pack(project_root: Path, result_dir: Path) -> dict[str, Any]:
    """기존 STEP 2-C 결과를 변경하지 않고 review_pack과 ZIP만 생성한다."""

    project_root, result_dir = project_root.resolve(), result_dir.resolve()
    required = (
        "roi_margin_report.txt", "roi_margin_summary.json", "roi_margin_results.csv",
        "roi_margin_manual_review.csv", "visual_samples", "contact_sheets",
    )
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"STEP 2-C 필수 artifact 누락: {missing}")
    review_dir = result_dir / "review_pack"
    zip_path = result_dir / "step2c_visual_review_pack.zip"
    if review_dir.exists() or zip_path.exists():
        raise FileExistsError("기존 STEP 2-C review pack을 덮어쓰지 않습니다")
    with (result_dir / "roi_margin_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        result_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with (result_dir / "roi_margin_manual_review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        manual_rows = list(csv.DictReader(handle))
    samples = [ReviewSample(result_rows[row["sample_id"]], compute_review_metrics(result_rows[row["sample_id"]])) for row in manual_rows]
    selected = assign_review_groups(samples)
    review_dir.mkdir()
    selected_dir = review_dir / "selected_images"
    selected_dir.mkdir()
    for sample in selected:
        source = result_dir / "visual_samples" / f"{sample.row['sample_id']}.jpg"
        destination = selected_dir / f"{sample.review_order:02d}_{sample.row['sample_id']}.jpg"
        shutil.copy2(source, destination)
        sample.copied_image = destination
    sheet_names = {group: file_name for group, file_name, _title in SHEET_DEFINITIONS}
    for group, file_name, title in SHEET_DEFINITIONS:
        _write_sheet(review_dir / file_name, title, [sample for sample in selected if sample.group == group])
    manifest = [_manifest_row(sample, sheet_names[sample.group], project_root) for sample in selected]
    _write_csv(review_dir / "review_manifest.csv", MANIFEST_COLUMNS, manifest)
    compact = [
        {
            "review_order": row["review_order"], "sheet_name": row["sheet_name"],
            "sample_id": row["sample_id"], "video_id": row["video_id"],
            "timestamp_sec": row["timestamp_sec"], "source_image_path": row["source_image_path"],
            "preferred_landmark_roi": "", "eye_comment": "", "mouth_comment": "",
            "pose_comment": "", "overall_comment": "",
        }
        for row in manifest
    ]
    _write_csv(review_dir / "review_manual_compact.csv", MANUAL_COLUMNS, compact)
    group_counts = Counter(sample.group for sample in selected)
    source_sheet_count = len(list((result_dir / "contact_sheets").glob("*.jpg")))
    clipping_inconsistency_count = sum(
        any(
            _flag(row, f"{candidate}_roi_clipped")
            and _number(row, f"{candidate}_clipped_fraction") == 0.0
            for candidate in ("m05", "m10", "m15")
        )
        for row in result_rows.values()
    )
    (review_dir / "README.md").write_text(
        _readme_text(group_counts, source_sheet_count, clipping_inconsistency_count),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(result_dir))
    return {
        "source_contact_sheet_count": source_sheet_count,
        "source_review_sample_count": len(manual_rows),
        "unique_final_sample_count": len({sample.row["sample_id"] for sample in selected}),
        "review_sheet_count": len(SHEET_DEFINITIONS),
        "group_counts": dict(group_counts),
        "clipping_inconsistency_count": clipping_inconsistency_count,
        "zip_path": str(zip_path), "decision": DECISION,
        "post_processing_only": True,
    }
