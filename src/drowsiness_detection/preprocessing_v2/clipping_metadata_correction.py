"""기존 STEP 2-C 결과를 보존하며 ROI clipping 메타데이터만 재계산한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .detectors import BoundingBox
from .landmark_roi_margin_audit import expand_landmark_roi


CANDIDATES = {"raw": 0.0, "m05": 0.05, "m10": 0.10, "m15": 0.15}
RESULT_COLUMNS = (
    "sample_id", "video_id", "split", "frame_width", "frame_height", "candidate",
    "old_roi_clipped", "old_clipped_fraction", "corrected_roi_clipped",
    "corrected_clipped_fraction", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
    "stored_roi_matches_corrected", "flag_changed",
)


def _metadata_dimensions(project_root: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """기존 train/val metadata의 frame 크기를 읽고 test split은 제외한다."""

    dimensions: dict[tuple[str, str], tuple[int, int]] = {}
    for split in ("train", "val"):
        with (project_root / "data" / "metadata" / f"{split}.csv").open(
            "r", encoding="utf-8-sig", newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                key = (split, row["video_id"])
                dimensions[key] = (int(row["width"]), int(row["height"]))
    return dimensions


def _statistics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """후보별 clipping 횟수와 fraction 분포를 계산한다."""

    output: dict[str, Any] = {}
    for candidate in CANDIDATES:
        subset = [row for row in rows if row["candidate"] == candidate]
        fractions = np.asarray([float(row[f"{prefix}_clipped_fraction"]) for row in subset])
        count = sum(bool(row[f"{prefix}_roi_clipped"]) for row in subset)
        output[candidate] = {
            "frame_count": len(subset), "clipped_count": count,
            "clipped_rate": count / len(subset),
            "clipped_fraction": {
                "mean": float(np.mean(fractions)), "median": float(np.median(fractions)),
                "max": float(np.max(fractions)), "p90": float(np.percentile(fractions, 90)),
                "p95": float(np.percentile(fractions, 95)),
            },
        }
    return output


def calculate_correction(project_root: Path, result_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """저장된 bbox와 영상 크기로 ROI 좌표 및 clipping metadata를 교차 검증한다."""

    dimensions = _metadata_dimensions(project_root)
    with (result_dir / "roi_margin_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        sources = list(csv.DictReader(handle))
    if len(sources) != 157 or any(row["split"] not in {"train", "val"} for row in sources):
        raise ValueError("기존 STEP 2-C YuNet 성공 157개 행이 필요합니다")
    rows: list[dict[str, Any]] = []
    for source in sources:
        width, height = dimensions[(source["split"], source["video_id"])]
        bbox = BoundingBox(*(int(source[f"det_bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2")))
        for candidate, margin in CANDIDATES.items():
            corrected = expand_landmark_roi(bbox, margin, width, height)
            stored = BoundingBox(*(int(source[f"{candidate}_roi_{axis}"]) for axis in ("x1", "y1", "x2", "y2")))
            old_flag = source[f"{candidate}_roi_clipped"].casefold() == "true"
            rows.append({
                "sample_id": source["sample_id"], "video_id": source["video_id"],
                "split": source["split"], "frame_width": width, "frame_height": height,
                "candidate": candidate, "old_roi_clipped": old_flag,
                "old_clipped_fraction": float(source[f"{candidate}_clipped_fraction"]),
                "corrected_roi_clipped": corrected.clipped,
                "corrected_clipped_fraction": corrected.clipped_fraction,
                "roi_x1": corrected.bbox.x1, "roi_y1": corrected.bbox.y1,
                "roi_x2": corrected.bbox.x2, "roi_y2": corrected.bbox.y2,
                "stored_roi_matches_corrected": stored == corrected.bbox,
                "flag_changed": old_flag != corrected.clipped,
            })
    matches = sum(row["stored_roi_matches_corrected"] for row in rows)
    impact = "REPORTING_ONLY_BUG" if matches == len(rows) else "PROCESSING_AFFECTED"
    summary = {
        "source_frame_count": len(sources), "candidate_row_count": len(rows),
        "frame_dimension_source": "data/metadata/train.csv, data/metadata/val.csv",
        "original_statistics": _statistics(rows, "old"),
        "corrected_statistics": _statistics(rows, "corrected"),
        "flag_changed_count": sum(row["flag_changed"] for row in rows),
        "stored_roi_coordinate_match_count": matches,
        "impact": impact,
        "roi_audit_rerun_required": impact == "PROCESSING_AFFECTED",
        "detector_landmark_feature_rerun_performed": False,
    }
    return rows, summary


def _report(summary: dict[str, Any]) -> str:
    """기존 값과 보정값, 원인, 처리 영향 판정을 보고서로 구성한다."""

    lines = ["STEP 2-C Clipping Metadata Correction", "", "[Original Statistics]"]
    for candidate, item in summary["original_statistics"].items():
        lines.append(f"{candidate.upper()}: {item['clipped_count']}/{item['frame_count']}, fraction={item['clipped_fraction']}")
    lines.extend([
        "", "[Root Cause]",
        "기존 roi_clipped는 margin을 적용한 실수 ROI와 floor/ceil로 정수화한 최종 ROI를 직접 비교했다.",
        "따라서 frame 내부에서 발생한 실수/정수 반올림 차이를 경계 clipping으로 잘못 판정했다.",
        "clipped_fraction은 면적 비율을 계산하여 0 미만을 0으로 제한했으므로 flag와 불일치했다.",
        "Dlib inclusive 끝점 변환은 predictor에서 이후에 수행되며 원인이 아니다.",
        "", "[Corrected Definition]",
        "반개방 정수 intended ROI와 같은 좌표계의 frame-clipped ROI가 다를 때만 clipped=True.",
        "clipped_fraction=1-clipped_area/intended_area; 정수 반개방 영역의 면적 손실 비율.",
        "", "[Corrected Statistics]",
    ])
    for candidate, item in summary["corrected_statistics"].items():
        lines.append(
            f"{candidate.upper()}: {item['clipped_count']}/{item['frame_count']} "
            f"({item['clipped_rate']:.2%}), fraction={item['clipped_fraction']}"
        )
    lines.extend([
        "", "[Impact Assessment]",
        f"저장된 ROI 좌표와 보정 코드 계산값 일치: {summary['stored_roi_coordinate_match_count']}/{summary['candidate_row_count']}",
        f"영향 판정: {summary['impact']}",
        "저장된 ROI가 모두 일치하면 Dlib에 전달된 ROI와 기존 landmark/feature/visual 결과는 변경되지 않는다.",
        f"ROI audit 재실행 필요: {summary['roi_audit_rerun_required']}",
        "기존 STEP 2-C 원본 artifact는 수정하지 않았다.",
    ])
    return "\n".join(lines) + "\n"


def write_correction(project_root: Path, result_dir: Path) -> dict[str, Any]:
    """보정 결과를 새 하위 디렉터리에만 저장하고 원본은 보존한다."""

    output_dir = result_dir / "clipping_metadata_correction"
    if output_dir.exists():
        raise FileExistsError(f"기존 clipping correction을 덮어쓰지 않습니다: {output_dir}")
    rows, summary = calculate_correction(project_root, result_dir)
    output_dir.mkdir()
    with (output_dir / "clipping_correction_results.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "clipping_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / "clipping_correction_report.txt").write_text(_report(summary), encoding="utf-8")
    return summary
