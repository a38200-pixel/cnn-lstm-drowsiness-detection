"""STEP 3-B Context crop 결측만 추출해 원본 frame 검토 자료를 만든다."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


LIST_COLUMNS = (
    "review_id", "split", "label", "video_id", "canonical_index", "context_slot",
    "source_frame_index", "target_timestamp_sec", "actual_timestamp_sec",
    "decode_ok", "decode_status", "yunet_success", "yunet_confidence",
    "yunet_detection_count", "detector_status", "landmark_success",
    "landmark_status", "pose_success", "pose_status", "context_status",
    "context_crop_available", "failure_flags", "bbox_x1", "bbox_y1", "bbox_x2",
    "bbox_y2", "frame_width", "frame_height", "source_video_path", "bundle_path",
    "policy_hash", "suspected_reason_auto", "neighbor_hint", "image_path",
)

MANUAL_COLUMNS = (
    "review_id", "split", "label", "video_id", "canonical_index", "context_slot",
    "source_frame_index", "detector_success", "landmark_success",
    "context_crop_available", "suspected_reason_auto", "manual_category",
    "detector_miss_plausible", "crop_storage_bug_suspected", "bbox_visible_to_human",
    "face_near_edge", "extreme_pose", "occlusion", "low_light", "blur",
    "review_decision", "review_note",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """BOM을 허용하면서 기존 CSV를 읽기 전용으로 연다."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """고정 컬럼 순서로 새 review CSV만 작성한다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _yes(value: Any) -> bool:
    """CSV의 문자열 bool과 테스트용 bool을 같은 의미로 해석한다."""

    return value is True or str(value).casefold() == "true"


def _number(value: Any) -> float:
    """없는 수치는 NaN으로 두고 실제 값이 있는지 별도로 판단한다."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def select_missing_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Train/val의 선택 Context slot 중 crop이 없는 행만 고른다."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") not in {"train", "val"}:
            if _yes(row.get("context_selected")) and not _yes(row.get("context_crop_available")):
                raise ValueError("test 또는 알 수 없는 split의 missing row 발견")
            continue
        if _yes(row.get("context_selected")) and not _yes(row.get("context_crop_available")):
            selected.append(dict(row))
    return sorted(selected, key=lambda row: (row["split"], row["video_id"], int(row["canonical_index"])))


def suspected_reason(row: Mapping[str, Any]) -> str:
    """수동 판정을 대신하지 않는 기계적 결측 원인 후보만 반환한다."""

    if not _yes(row.get("decode_ok")):
        return "SOURCE_FRAME_UNAVAILABLE"
    if not _yes(row.get("yunet_success")):
        return "DETECTOR_MISS"
    if _yes(row.get("context_selected")) and not _yes(row.get("context_crop_available")):
        return "PIPELINE_OR_STORAGE_SUSPECT"
    return "UNCLASSIFIED"


def _neighbor_hint(row: Mapping[str, Any], video_rows: Sequence[Mapping[str, Any]]) -> str:
    """근처 검출 성공 slot의 pose·경계만 참고하며 현재 frame 판단으로 오인하지 않는다."""

    current = int(row["canonical_index"])
    nearby = [candidate for candidate in video_rows
              if _yes(candidate.get("yunet_success")) and abs(int(candidate["canonical_index"]) - current) <= 5]
    if not nearby:
        return "NO_NEARBY_DETECTION_WITHIN_5_SLOTS"
    nearest = min(nearby, key=lambda candidate: abs(int(candidate["canonical_index"]) - current))
    hints: list[str] = []
    yaw = _number(nearest.get("yaw"))
    if math.isfinite(yaw) and abs(yaw) >= 30:
        hints.append("NEARBY_LARGE_YAW")
    edges = (_number(nearest.get("bbox_x1")), _number(nearest.get("bbox_y1")),
             _number(nearest.get("frame_width")) - _number(nearest.get("bbox_x2")),
             _number(nearest.get("frame_height")) - _number(nearest.get("bbox_y2")))
    if all(math.isfinite(value) for value in edges) and min(edges) < 12:
        hints.append("NEARBY_FRAME_EDGE_FACE")
    if not hints:
        hints.append("NEARBY_FACE_DETECTED")
    return ";".join(hints) + f"@k{nearest['canonical_index']}"


def _source_frame(path: Path, source_indices: set[int]) -> dict[int, np.ndarray]:
    """한 영상에서 필요한 원본 index까지 순차 decode하고 그 frame만 보존한다."""

    if not source_indices:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"원본 영상 열기 실패: {path}")
    frames: dict[int, np.ndarray] = {}
    try:
        for index in range(max(source_indices) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index in source_indices:
                frames[index] = frame.copy()
    finally:
        capture.release()
    return frames


def _letterbox(frame: np.ndarray, size: int = 320) -> np.ndarray:
    """원본 비율을 보존해 review panel 왼쪽에 배치한다."""

    output = np.full((size, size, 3), 36, dtype=np.uint8)
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    x = (size - resized.shape[1]) // 2
    y = (size - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def render_missing_panel(record: Mapping[str, Any], frame: np.ndarray | None) -> np.ndarray:
    """없는 crop을 생성하지 않고 원본·bbox 상태와 결측 메타데이터만 그린다."""

    canvas = np.full((438, 780, 3), 24, dtype=np.uint8)
    row = record
    if frame is None:
        left = np.full((320, 320, 3), 36, dtype=np.uint8)
        cv2.putText(left, "SOURCE FRAME UNAVAILABLE", (8, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    else:
        original = frame.copy()
        bbox = [_number(row.get(key)) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        if _yes(row["yunet_success"]) and all(math.isfinite(value) for value in bbox):
            cv2.rectangle(original, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        left = _letterbox(original)
        if not _yes(row["yunet_success"]):
            cv2.putText(left, "NO YUNET BBOX", (58, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    canvas[118:438, 10:330] = left
    cv2.putText(canvas, "CROP MISSING", (408, 267), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    lines = [
        f"{row['review_id']}  {row['split']}/{row['label']}  {row['video_id']}",
        f"context_selected=True  ctx={int(row['context_slot']):02d}  k={int(row['canonical_index']):03d}  src={int(row['source_frame_index'])}",
        f"YuNet={row['yunet_success']}  Dlib={row['landmark_success']}  Pose={row['pose_success']}",
        f"decode={row['decode_status']}  detector={row['detector_status']}  context={row['context_status']}",
        f"auto={row['suspected_reason_auto']}  hint={row['neighbor_hint'][:45]}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(canvas, line[:102], (10, 19 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _summary(records: Sequence[Mapping[str, Any]], selected_count: int, pilot_root: Path,
             policy_hashes: set[str], image_count: int, sheet_count: int,
             source_decode_missing: int) -> dict[str, Any]:
    """원본 artifact 기반 분포와 자동 추정의 한계를 구조화한다."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(pilot_root.resolve()),
        "target_definition": "split in {train,val} AND context_selected=True AND context_crop_available=False",
        "total_context_selected": selected_count,
        "total_context_missing": len(records),
        "missing_rate": len(records) / selected_count if selected_count else 0.0,
        "by_split": dict(Counter(str(row["split"]) for row in records)),
        "by_label": dict(Counter(str(row["label"]) for row in records)),
        "by_video_desc": [{"video_id": video_id, "count": count} for video_id, count in
                          sorted(Counter(str(row["video_id"]) for row in records).items(),
                                 key=lambda item: (-item[1], item[0]))],
        "detector_success": dict(Counter(str(_yes(row["yunet_success"])) for row in records)),
        "landmark_success": dict(Counter(str(_yes(row["landmark_success"])) for row in records)),
        "pose_success": dict(Counter(str(_yes(row["pose_success"])) for row in records)),
        "decode_status": dict(Counter(str(row["decode_status"]) for row in records)),
        "context_status": dict(Counter(str(row["context_status"]) for row in records)),
        "missing_reason": dict(Counter(str(row["suspected_reason_auto"]) for row in records)),
        "canonical_index": {"min": min((int(row["canonical_index"]) for row in records), default=None),
                            "max": max((int(row["canonical_index"]) for row in records), default=None),
                            "counts": dict(sorted(Counter(int(row["canonical_index"]) for row in records).items()))},
        "context_slot": {"min": min((int(row["context_slot"]) for row in records), default=None),
                         "max": max((int(row["context_slot"]) for row in records), default=None),
                         "counts": dict(sorted(Counter(int(row["context_slot"]) for row in records).items()))},
        "policy_hashes": sorted(policy_hashes),
        "visual_artifacts": {"sample_count": image_count, "sheet_count": sheet_count,
                             "source_decode_missing": source_decode_missing},
        "test_split_used": False,
        "pilot_artifacts_modified": False,
        "manual_interpretation": "WAITING",
    }


def _report(summary: Mapping[str, Any]) -> str:
    """자동 원인과 아직 필요한 사람의 이미지 판단을 분리해 기록한다."""

    reasons = summary["missing_reason"]
    only_detector_miss = bool(summary["total_context_missing"]) and reasons == {"DETECTOR_MISS": summary["total_context_missing"]}
    if summary["total_context_missing"] == 0:
        interpretation = "결측이 없어 검토 대상이 없습니다."
    elif only_detector_miss:
        interpretation = "모든 결측 row는 정상 decode 후 YuNet 미검출과 일치합니다. 저장/경로 실패의 자동 증거는 없습니다. 실제 얼굴이 보이는 false negative인지는 수동 확인이 필요합니다."
    else:
        interpretation = "검출 성공 상태의 crop 결측 등 추가 조사 대상이 포함됩니다. 자동 원인 분포를 확인하십시오."
    lines = [
        "STEP 3-B Missing Context Crop Review", "",
        "목적: 선택된 Context slot의 crop 결측만 분리해 detector miss·false negative·저장/파이프라인 문제를 시각 점검한다.",
        f"출처: {summary['source_directory']}",
        f"대상: {summary['target_definition']}",
        f"결측: {summary['total_context_missing']}/{summary['total_context_selected']} ({summary['missing_rate']:.2%})",
        f"split: {summary['by_split']}", f"label: {summary['by_label']}",
        f"video: {summary['by_video_desc']}",
        f"YuNet: {summary['detector_success']} / Dlib68: {summary['landmark_success']} / Pose: {summary['pose_success']}",
        f"decode: {summary['decode_status']} / context: {summary['context_status']}",
        f"자동 원인 후보: {reasons}",
        f"canonical index: {summary['canonical_index']['min']}..{summary['canonical_index']['max']}; "
        f"context slot: {summary['context_slot']['min']}..{summary['context_slot']['max']}",
        f"이미지: {summary['visual_artifacts']['sample_count']}개, sheet: {summary['visual_artifacts']['sheet_count']}장, 원본 frame decode 누락: {summary['visual_artifacts']['source_decode_missing']}개",
        "", "대표 패턴 및 구현 문제 의심 여부:", interpretation,
        "`DETECTOR_MISS`는 저장된 stage 상태에 따른 기계적 분류일 뿐 정상 miss와 false negative를 구분하지 않습니다. "
        "근처 frame의 yaw/edge 힌트도 현재 frame의 사람 판정이 아닙니다.",
        "", "다음 액션: 모든 원본 frame을 보고 얼굴·profile·edge·가림·조명·흐림을 분류하고, 검출 성공인데 crop이 없는 사례가 있으면 우선 구현/경로 문제로 조사합니다. "
        "STEP 2 정책은 바꾸지 않고 full preprocessing은 시작하지 않습니다.",
        "", "DECISION:", "MISSING_CONTEXT_REVIEW_COMPLETE", "", "PRELIMINARY INTERPRETATION:",
        f"- {interpretation}", "", "STEP 3-B STATUS:", "AUTOMATIC CHECK COMPLETE",
        "MISSING-ONLY VISUAL REVIEW GENERATED", "MANUAL INTERPRETATION REQUIRED", "",
    ]
    return "\n".join(lines)


def _guide() -> str:
    """사람이 표본별 관찰을 입력할 때의 범주와 체크 포인트를 설명한다."""

    return """# STEP 3-B Context Missing-Only 수동 검토 가이드

이 pack은 `context_selected=True`이고 `context_crop_available=False`인 pilot row만 담습니다. 왼쪽은 저장된 source index를 순차 decode한 원본 frame이고, 녹색 사각형은 YuNet bbox가 **실제로 저장된 경우에만** 그립니다. 오른쪽 `CROP MISSING`은 누락을 표시할 뿐 crop을 재생성하지 않습니다.

1. 실제 운전자 얼굴이 보이는지, 다른 물체·동승자와 혼동하지 않았는지 확인합니다.
2. YuNet bbox가 없다면 detector miss가 타당한 상황인지 판단합니다. 얼굴이 선명한데 bbox가 없다면 false negative 후보입니다.
3. bbox가 있는데도 crop이 없으면 `context_status`, 파일 경로, 저장 단계 문제를 우선 조사합니다.
4. Profile/큰 yaw, frame 가장자리, 가림·선글라스·마스크, 저조도·과노출, blur를 체크합니다.
5. 이미지의 `neighbor_hint`는 근처 *다른* frame의 yaw/edge 정보입니다. 현재 frame의 pose 정답으로 사용하지 마십시오.

`missing_context_manual_review.csv`에서 `manual_category`는 `EXPECTED_DETECTOR_MISS`, `LIKELY_FALSE_NEGATIVE`, `POSSIBLE_STORAGE_OR_PIPELINE_ISSUE`, `UNCERTAIN` 중 하나를 입력합니다. `review_decision`은 `PASS`, `CHECK_NEEDED`, `INVESTIGATE` 중 하나를 입력합니다. `detector_miss_plausible` 등 관찰 칸과 `review_note`는 사람이 직접 채웁니다. 자동 추정값을 수동 칸으로 복사하지 마십시오.

검토 목적은 STEP 3 구현·missing pattern 확인이며 YuNet threshold, SQUARE_M10, fallback 또는 졸음 행동 임계값을 재선택하는 것이 아닙니다. 수동 검토 전까지 판정은 대기 상태입니다.
"""


def collect_records(pilot_root: Path) -> tuple[list[dict[str, Any]], int, set[str]]:
    """Global CSV와 완료 bundle을 대조하고 missing 행의 provenance를 모은다."""

    global_rows = _read_csv(pilot_root / "metadata" / "canonical_frames.csv")
    videos = _read_csv(pilot_root / "metadata" / "canonical_videos.csv")
    manifest = _read_csv(pilot_root / "metadata" / "input_manifest.csv")
    if any(row["split"] not in {"train", "val"} for row in global_rows + videos):
        raise ValueError("pilot 결과에 test 또는 알 수 없는 split이 있습니다")
    if any(row["split"] not in {"train", "val"} for row in manifest):
        raise ValueError("pilot 입력 manifest에 test 또는 알 수 없는 split이 있습니다")
    selected_count = sum(_yes(row["context_selected"]) for row in global_rows)
    missing = select_missing_rows(global_rows)
    by_video = {row["video_id"]: row for row in videos}
    by_manifest = {row["video_id"]: row for row in manifest}
    if len(by_video) != len(videos):
        raise ValueError("pilot video_id가 중복됐습니다")
    if len(by_manifest) != len(manifest) or set(by_video) != set(by_manifest):
        raise ValueError("pilot manifest와 완료 영상 목록이 일치하지 않습니다")
    for video_id, video in by_video.items():
        expected = by_manifest[video_id]
        if video["split"] != expected["split"] or video["label"] != expected["label"] or video["source_path"] != expected["source_path"]:
            raise ValueError(f"pilot manifest의 원본 경로·identity 불일치: {video_id}")
    video_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in global_rows:
        video_rows[row["video_id"]].append(row)
    local_missing: set[tuple[str, str, int]] = set()
    for video in videos:
        bundle = pilot_root / "per_video" / video["split"] / video["video_id"]
        if not (bundle / "COMPLETE.json").is_file():
            raise ValueError(f"완료 marker가 없습니다: {bundle}")
        for row in select_missing_rows(_read_csv(bundle / "frames.csv")):
            local_missing.add((row["split"], row["video_id"], int(row["canonical_index"])))
    global_missing = {(row["split"], row["video_id"], int(row["canonical_index"])) for row in missing}
    if local_missing != global_missing:
        raise ValueError("global/per-video missing row가 일치하지 않습니다")
    records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for index, row in enumerate(missing):
        video = by_video[row["video_id"]]
        if (row["split"], row["label"]) != (video["split"], video["label"]):
            raise ValueError("frame/video identity가 일치하지 않습니다")
        bundle = pilot_root / "per_video" / row["split"] / row["video_id"]
        marker = json.loads((bundle / "COMPLETE.json").read_text(encoding="utf-8"))
        hashes.add(marker["policy_hash"])
        record = {column: row.get(column, "") for column in LIST_COLUMNS}
        record.update(review_id=f"{index:03d}", source_video_path=video["source_path"],
                      bundle_path=str(bundle.resolve()), policy_hash=marker["policy_hash"],
                      suspected_reason_auto=suspected_reason(row),
                      neighbor_hint=_neighbor_hint(row, video_rows[row["video_id"]]),
                      image_path=f"samples/{index:03d}_{row['split']}_{row['video_id']}_ctx{int(row['context_slot']):02d}_k{int(row['canonical_index']):03d}.jpg")
        records.append(record)
    return records, selected_count, hashes


def generate_review(pilot_root: Path, output_dir: Path, expected_count: int | None = None) -> dict[str, Any]:
    """기존 pilot 출력은 읽기만 하고 새 디렉터리를 완성한 뒤 발행한다."""

    pilot_root = pilot_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(pilot_root):
        raise ValueError("review 출력을 기존 pilot canonical root 안에 만들 수 없습니다")
    if output_dir.exists():
        raise FileExistsError(f"기존 missing review 디렉터리 덮어쓰기 금지: {output_dir}")
    records, selected_count, hashes = collect_records(pilot_root)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"예상 missing {expected_count}건과 실제 {len(records)}건이 다릅니다")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = output_dir.parent / f".{output_dir.name}_tmp_{uuid.uuid4().hex}"
    temp.mkdir()
    try:
        samples = temp / "samples"
        samples.mkdir()
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_video[str(record["video_id"])].append(record)
        images: list[np.ndarray] = []
        source_decode_missing = 0
        for items in by_video.values():
            source = Path(str(items[0]["source_video_path"]))
            frames = _source_frame(source, {int(item["source_frame_index"]) for item in items})
            for item in items:
                frame = frames.get(int(item["source_frame_index"]))
                if frame is None:
                    source_decode_missing += 1
                panel = render_missing_panel(item, frame)
                path = temp / str(item["image_path"])
                if not cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"review image 저장 실패: {path}")
                images.append(panel)
        sheet_count = 0
        for start in range(0, len(images), 4):
            sheet = np.vstack(images[start:start + 4])
            path = temp / f"missing_context_sheet_{start // 4 + 1:02d}.jpg"
            if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"contact sheet 저장 실패: {path}")
            sheet_count += 1
        summary = _summary(records, selected_count, pilot_root, hashes, len(images), sheet_count, source_decode_missing)
        _write_csv(temp / "missing_context_rows.csv", LIST_COLUMNS, records)
        manual = [{column: record.get(column, "") if column in {
            "review_id", "split", "label", "video_id", "canonical_index", "context_slot",
            "source_frame_index", "suspected_reason_auto"} else "" for column in MANUAL_COLUMNS}
                  for record in records]
        for review, record in zip(manual, records):
            review["detector_success"] = record["yunet_success"]
            review["landmark_success"] = record["landmark_success"]
            review["context_crop_available"] = record["context_crop_available"]
        _write_csv(temp / "missing_context_manual_review.csv", MANUAL_COLUMNS, manual)
        (temp / "missing_context_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp / "missing_context_report.txt").write_text(_report(summary), encoding="utf-8")
        (temp / "MISSING_CONTEXT_REVIEW_GUIDE.md").write_text(_guide(), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"기존 review 디렉터리 덮어쓰기 금지: {output_dir}")
        os.replace(temp, output_dir)
        return summary
    except Exception:
        if temp.resolve().parent != output_dir.parent.resolve():
            raise RuntimeError("임시 review 경로가 허용 범위를 벗어났습니다")
        shutil.rmtree(temp)
        raise
