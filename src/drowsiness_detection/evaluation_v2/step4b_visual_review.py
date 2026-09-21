"""STEP 4-B 후보 36개의 읽기 전용 원본·저장 crop 시각 검토 팩을 만든다."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


EXPECTED_POLICY_HASH = "29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721"
REASONS = {
    "highest_missing_count", "longest_consecutive_run", "highest_context_missing",
    "isolated_single_miss", "zero_missing_reference",
}
MANUAL_FIELDS = (
    "manual_pattern", "face_visible_during_miss", "profile_or_large_yaw",
    "face_near_frame_edge", "partial_face_clipping", "occlusion", "low_light",
    "blur", "face_too_small", "back_of_head_or_no_face",
    "visual_condition_changes_across_run", "detector_miss_plausible",
    "likely_false_negative", "pipeline_issue_suspected", "sequence_severity",
    "manual_category", "review_decision", "review_note",
)
INDEX_COLUMNS = (
    "review_id", "video_id", "split", "label", "selection_reason",
    "yunet_missing_count", "longest_missing_run", "context_missing_count",
    "selected_canonical_indices", "individual_sheet_path", "contact_sheet_id",
)
MANUAL_AUTO_COLUMNS = (
    "review_id", "video_id", "split", "label", "selection_reason",
    "yunet_missing_count", "yunet_missing_rate", "longest_missing_run",
    "number_of_missing_runs", "context_missing_count", "context_missing_rate",
    "behavior_valid_rate", "selected_frame_indices",
)


def _csv(path: Path) -> list[dict[str, str]]:
    """원본 CSV를 수정하지 않고 읽는다."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """검토 팩 안의 신규 CSV만 기록한다."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _yes(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _runs(indices: Sequence[int]) -> list[tuple[int, int]]:
    """결측 canonical index를 연속 구간으로 묶는다."""
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = last = indices[0]
    for index in indices[1:]:
        if index != last + 1:
            runs.append((start, last))
            start = index
        last = index
    runs.append((start, last))
    return runs


def validate_candidates(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """후보를 재선정하지 않고 STEP 4-A manifest의 순서·범위를 검증한다."""
    if len(rows) != 36:
        raise ValueError(f"candidate rows는 36개여야 합니다: {len(rows)}")
    ids = [str(row.get("video_id", "")) for row in rows]
    if len(set(ids)) != 36 or any(not video_id or Path(video_id).name != video_id for video_id in ids):
        raise ValueError("candidate video_id 중복 또는 잘못된 ID")
    if any(row.get("split") not in {"train", "val"} for row in rows):
        raise ValueError("STEP_4B_BLOCKED_TEST_LEAKAGE: candidate test/unknown split")
    if any(row.get("label") not in {"drowsy", "not_drowsy"} for row in rows):
        raise ValueError("candidate label이 올바르지 않습니다")
    if [int(row["review_order"]) for row in rows] != list(range(36)):
        raise ValueError("candidate review_order는 0~35 순서여야 합니다")
    for row in rows:
        reasons = set(str(row.get("selection_reason", "")).split(";"))
        if not reasons or not reasons <= REASONS:
            raise ValueError(f"selection_reason 누락/오류: {row.get('video_id')}")
        recommended = str(row.get("recommended_frame_indices", ""))
        if recommended and any(not 0 <= int(index) < 100 for index in recommended.split(";")):
            raise ValueError(f"recommended_frame_indices 오류: {row['video_id']}")
    return [dict(row) for row in rows]


def select_frame_indices(candidate: Mapping[str, str], frames: Sequence[Mapping[str, str]]) -> list[int]:
    """선정 사유에 따라 비교에 필요한 canonical frame 최대 5개만 고른다."""
    if len(frames) != 100 or [int(row["canonical_index"]) for row in frames] != list(range(100)):
        raise ValueError(f"candidate canonical frame 100개/순서 오류: {candidate['video_id']}")
    missing = [index for index, row in enumerate(frames) if not _yes(row["yunet_success"])]
    runs = _runs(missing)
    reason = set(candidate["selection_reason"].split(";"))
    if len(missing) == 100:
        selected = [0, 25, 50, 75, 99]
    elif reason & {"longest_consecutive_run", "highest_missing_count"} and runs:
        start, end = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
        before = next((index for index in range(start - 1, -1, -1) if index not in missing), None)
        after = next((index for index in range(end + 1, 100) if index not in missing), None)
        selected = [index for index in (before, start, (start + end) // 2, end, after) if index is not None]
    elif "highest_context_missing" in reason:
        missing_context = [index for index, row in enumerate(frames)
                           if _yes(row["context_selected"]) and not _yes(row["context_crop_available"])]
        if not missing_context:
            raise ValueError(f"Context 결측 후보에 결측 slot이 없습니다: {candidate['video_id']}")
        positions = [0, len(missing_context) // 2, len(missing_context) - 1]
        selected = [missing_context[position] for position in positions]
        valid_context = [index for index, row in enumerate(frames)
                         if _yes(row["context_selected"]) and _yes(row["context_crop_available"])]
        if valid_context:
            selected.append(min(valid_context, key=lambda index: (min(abs(index - miss) for miss in selected), index)))
    elif "isolated_single_miss" in reason:
        isolated = next(((start, end) for start, end in runs if start == end), None)
        if isolated is None:
            raise ValueError(f"고립 결측 후보에 단일 run이 없습니다: {candidate['video_id']}")
        index = isolated[0]
        selected = [near for near in (index - 1, index, index + 1) if 0 <= near < 100]
    elif "zero_missing_reference" in reason and not missing:
        selected = [0, 50, 99]
    else:
        raise ValueError(f"selection_reason과 결측 구조가 일치하지 않습니다: {candidate['video_id']}")
    return sorted(set(selected))[:5]


def load_plan(canonical_root: Path, audit_root: Path) -> list[dict[str, Any]]:
    """Dry-run에서도 원본 영상을 열지 않고 후보·저장 artifact의 정합성을 확인한다."""
    candidates = validate_candidates(_csv(audit_root / "step4b_review_candidates.csv"))
    videos = _csv(canonical_root / "metadata/canonical_videos.csv")
    if any(row["split"] not in {"train", "val"} for row in videos):
        raise ValueError("STEP_4B_BLOCKED_TEST_LEAKAGE: canonical_videos")
    video_map = {row["video_id"]: row for row in videos}
    if len(video_map) != len(videos):
        raise ValueError("canonical_videos video_id 중복")
    if any(row["policy_hash"] != EXPECTED_POLICY_HASH for row in videos):
        raise ValueError("STEP 3-C policy hash 불일치")
    run_hash = json.loads((canonical_root / "metadata/run_metadata.json").read_text(encoding="utf-8"))["policy_hash"]
    if run_hash != EXPECTED_POLICY_HASH:
        raise ValueError("run_metadata policy hash 불일치")
    quality_rows = _csv(audit_root / "per_video_quality.csv")
    quality_map = {row["video_id"]: row for row in quality_rows}
    if len(quality_map) != len(quality_rows):
        raise ValueError("per_video_quality video_id 중복")
    candidate_ids = {row["video_id"] for row in candidates}
    frame_map: dict[str, list[dict[str, str]]] = {video_id: [] for video_id in candidate_ids}
    with (canonical_root / "metadata/canonical_frames.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "val"}:
                raise ValueError("STEP_4B_BLOCKED_TEST_LEAKAGE: canonical_frames")
            if row["video_id"] in candidate_ids:
                frame_map[row["video_id"]].append(row)
    plan: list[dict[str, Any]] = []
    for candidate in candidates:
        video_id = candidate["video_id"]
        video = video_map.get(video_id)
        quality = quality_map.get(video_id)
        if video is None or quality is None:
            raise ValueError(f"후보 video metadata/quality 누락: {video_id}")
        if any(row["split"] != candidate["split"] or row["label"] != candidate["label"]
               for row in (video, quality)):
            raise ValueError(f"후보 split/label 불일치: {video_id}")
        frames = sorted(frame_map[video_id], key=lambda row: int(row["canonical_index"]))
        if any(row["split"] != candidate["split"] or row["label"] != candidate["label"] for row in frames):
            raise ValueError(f"frame split/label 불일치: {video_id}")
        indices = select_frame_indices(candidate, frames)
        missing_count = sum(not _yes(row["yunet_success"]) for row in frames)
        context_missing = sum(_yes(row["context_selected"]) and not _yes(row["context_crop_available"]) for row in frames)
        if missing_count != int(candidate["yunet_missing_count"]) or context_missing != int(candidate["context_missing_count"]):
            raise ValueError(f"STEP 4-A candidate count 불일치: {video_id}")
        if missing_count != int(quality["yunet_missing_count"]) or context_missing != int(quality["context_missing_count"]):
            raise ValueError(f"STEP 4-A quality count 불일치: {video_id}")
        if quality["policy_hash"] != EXPECTED_POLICY_HASH or video["policy_hash"] != EXPECTED_POLICY_HASH:
            raise ValueError(f"후보 policy hash 불일치: {video_id}")
        source = Path(video["source_path"])
        if source.stem != video_id:
            raise ValueError(f"원본 영상 ID/경로 불일치: {video_id}: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"후보 원본 영상 없음: {video_id}: {source}")
        bundle = canonical_root / "per_video" / candidate["split"] / video_id
        selected_rows = [frames[index] for index in indices]
        for row in selected_rows:
            if int(row["frame_width"]) < 1 or int(row["frame_height"]) < 1:
                raise ValueError(f"frame size metadata 오류: {video_id}/k{row['canonical_index']}")
            if not _yes(row["context_selected"]) or not _yes(row["context_crop_available"]):
                continue
            relative = Path(row["context_crop_relpath"])
            crop = (bundle / relative).resolve()
            if relative.is_absolute() or not crop.is_relative_to(bundle.resolve()) or not crop.is_file():
                raise FileNotFoundError(f"PIPELINE REVIEW BLOCKER: 저장된 Context JPEG 누락/경로 오류: {crop}")
        plan.append({"candidate": candidate, "video": video, "quality": quality,
                     "frames": frames, "selected_indices": indices, "selected_rows": selected_rows,
                     "source": source, "bundle": bundle})
    return plan


def _read_source_frames(source: Path, rows: Sequence[Mapping[str, str]]) -> dict[int, np.ndarray]:
    """필요 source index에만 seek하여 decode하고 index·크기를 검증한다."""
    wanted = sorted({int(row["source_frame_index"]) for row in rows})
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"원본 영상 열기 실패: {source}")
    frames: dict[int, np.ndarray] = {}
    try:
        for index in wanted:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"원본 frame seek 실패: {source} #{index}")
            ok, frame = capture.read()
            actual = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if not ok or frame is None or abs(actual - index) > 0.25:
                raise ValueError(f"원본 frame index 검증 실패: {source} expected={index}, got={actual}")
            frames[index] = frame
        for row in rows:
            index = int(row["source_frame_index"])
            frame = frames[index]
            if (frame.shape[1], frame.shape[0]) != (int(row["frame_width"]), int(row["frame_height"])):
                raise ValueError(f"원본 frame size 불일치: {source} #{index}")
            fps = float(row["source_fps"])
            actual_sec = float(row["actual_timestamp_sec"])
            if fps <= 0 or not math.isfinite(actual_sec) or abs(index / fps - actual_sec) > max(0.05, 1 / fps):
                raise ValueError(f"원본 frame timestamp metadata 불일치: {source} #{index}")
    finally:
        capture.release()
    return frames


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """원본 및 저장 crop의 비율을 유지하며 여백에 배치한다."""
    canvas = np.full((height, width, 3), 38, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _line(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int] = (245, 245, 245)) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, color, 1, cv2.LINE_AA)


def render_panel(row: Mapping[str, str], frame: np.ndarray, stored_crop: np.ndarray | None,
                 run_marker: str = "") -> np.ndarray:
    """성공 frame에서만 저장 bbox를 그리고 저장 crop 외의 대체 이미지는 만들지 않는다."""
    panel = np.full((430, 490, 3), 24, dtype=np.uint8)
    original = frame.copy()
    detected = _yes(row["yunet_success"])
    if detected:
        bbox = [float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError(f"YuNet success에 저장 bbox가 없습니다: {row['video_id']}/k{row['canonical_index']}")
        cv2.rectangle(original, (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3])), (0, 220, 0), 3)
    panel[92:332, 5:245] = _fit(original, 240, 240)
    if stored_crop is not None:
        panel[92:332, 245:485] = _fit(stored_crop, 240, 240)
    else:
        _line(panel, "CROP MISSING" if _yes(row["context_selected"]) else "CROP N.A.", 290, 214)
    _line(panel, f"k{int(row['canonical_index']):03d}  t={float(row['actual_timestamp_sec']):.3f}s  src={row['source_frame_index']}", 8, 24)
    _line(panel, f"YUNET {'OK' if detected else 'MISS'}  {run_marker}", 8, 48, (125, 230, 125) if detected else (100, 130, 255))
    _line(panel, "STORED BBOX" if detected else "NO STORED BBOX", 8, 72)
    _line(panel, f"Context: {'SELECTED' if _yes(row['context_selected']) else 'NOT SELECTED'}", 8, 355)
    crop_status = "AVAILABLE" if stored_crop is not None else ("MISSING" if _yes(row["context_selected"]) else "N.A.")
    _line(panel, f"Crop: {crop_status}", 8, 380)
    if detected and row.get("yunet_confidence"):
        _line(panel, f"conf={float(row['yunet_confidence']):.3f}", 8, 405)
    return panel


def _crop_image(bundle: Path, row: Mapping[str, str]) -> np.ndarray | None:
    if not _yes(row["context_selected"]) or not _yes(row["context_crop_available"]):
        return None
    path = (bundle / row["context_crop_relpath"]).resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise FileNotFoundError(f"PIPELINE REVIEW BLOCKER: 저장된 Context JPEG 누락: {path}")
    crop = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if crop is None or crop.shape[:2] != (224, 224):
        raise ValueError(f"PIPELINE REVIEW BLOCKER: Context JPEG 손상/크기 오류: {path}")
    return crop


def _save_jpeg(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError(f"review JPEG encode 실패: {path}")
    path.write_bytes(encoded.tobytes())


def render_individual(item: Mapping[str, Any], raw_frames: Mapping[int, np.ndarray]) -> np.ndarray:
    """영상별 요약과 선택 frame 전체를 한 장에 그린다."""
    candidate, quality = item["candidate"], item["quality"]
    indices = item["selected_indices"]
    sheet = np.full((535, 490 * len(indices), 3), 25, dtype=np.uint8)
    reasons = candidate["selection_reason"]
    header = [
        f"REVIEW {int(candidate['review_order']):02d}  {candidate['video_id']}  {candidate['split']}/{candidate['label']}  {reasons}",
        f"YuNet miss {candidate['yunet_missing_count']}/100 ({float(candidate['yunet_missing_rate']):.1%})  longest run {candidate['longest_missing_run']}  Context miss {candidate['context_missing_count']}/32  Behavior valid {float(candidate['behavior_valid_rate']):.1%}",
    ]
    for offset, line in enumerate(header):
        _line(sheet, line, 10, 30 + 28 * offset)
    missing = [i for i, row in enumerate(item["frames"]) if not _yes(row["yunet_success"])]
    runs = _runs(missing)
    longest = max(runs, key=lambda run: (run[1] - run[0], -run[0])) if runs else None
    for position, row in enumerate(item["selected_rows"]):
        index = int(row["canonical_index"])
        marker = ""
        if longest and longest[0] <= index <= longest[1]:
            marker = "RUN START" if index == longest[0] else "RUN END" if index == longest[1] else "RUN MID"
        panel = render_panel(row, raw_frames[int(row["source_frame_index"])], _crop_image(item["bundle"], row), marker)
        sheet[90:520, position * 490:(position + 1) * 490] = panel
    return sheet


def make_contact_sheets(individual: Sequence[np.ndarray], output_dir: Path, per_sheet: int = 4) -> list[str]:
    """각 후보가 정확히 한 contact sheet에 등장하도록 2×2 묶음을 만든다."""
    names: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for start in range(0, len(individual), per_sheet):
        group = individual[start:start + per_sheet]
        canvas = np.full((2 * 385, 2 * 1280, 3), 24, dtype=np.uint8)
        for local, image in enumerate(group):
            resized = _fit(image, 1280, 385)
            y, x = divmod(local, 2)
            canvas[y * 385:(y + 1) * 385, x * 1280:(x + 1) * 1280] = resized
        name = f"step4b_review_sheet_{len(names) + 1:02d}.jpg"
        _save_jpeg(output_dir / name, canvas)
        names.append(name)
    return names


def _guide() -> str:
    return """# STEP 4-B 수동 시각 검토 안내

## 핵심 질문

**YuNet이 놓친 구간에서 사람이 보기에 얼굴은 얼마나 식별 가능한가?**

얼굴의 profile·큰 yaw, frame edge, 얼굴 잘림, 조명, blur, 가림, 얼굴 크기, 얼굴 부재·뒤통수 여부를 확인하세요. Miss가 고립된 한 frame인지 연속 구간인지, run 전후 pose·얼굴 상태가 달라지는지, 얼굴이 충분히 보여도 miss가 지속되는지도 살펴보세요.

## 이미지 찾기와 비교

`step4b_review_index.csv`의 `review_id`와 `individual_sheet_path`가 1:1로 대응합니다. `contact_sheet_id`는 네 영상씩 묶은 전체 목차입니다. 먼저 contact sheet를 훑고 해당 individual sheet를 확대하세요. Sheet의 왼쪽은 원본 frame이며 초록 상자는 **기존 metadata에 저장된 YuNet bbox**입니다. 오른쪽은 해당 slot에 이미 저장된 SQUARE_M10 JPEG입니다. `YUNET MISS`에는 저장 bbox가 없으며, `CROP MISSING`은 새 crop으로 채우지 않았습니다.

높은 결측·긴 run은 valid-before → run-start/mid/end → valid-after를 비교하세요. 전후 frame이 없으면 표시하지 않습니다. 100/100 miss는 k000·025·050·075·099를 확인하세요. Context 결측 후보는 초·중·후 결측 slot과 가능한 저장 crop 참조를 비교하세요. 고립 miss는 앞·해당·뒤 frame을, 정상 참조는 초·중·후를 비교하세요. 선택된 일부 frame만 보여주므로 표시되지 않은 구간까지 단정하지 마세요.

## 수동 입력

`step4b_manual_review.csv`의 자동 metadata 열은 유지하고, 모든 manual 열은 **실제 이미지를 확인한 후** 직접 입력하세요. 미평가 열은 빈 칸으로 남기세요. 예/아니요 성격의 열은 `YES`/`NO`/`UNCERTAIN`을 권장합니다.

`manual_pattern` 허용 예: `PROFILE_YAW`, `FRAME_EDGE`, `PARTIAL_FACE`, `OCCLUSION`, `LOW_LIGHT`, `BLUR`, `SMALL_FACE`, `NO_FACE_OR_BACK_HEAD`, `MIXED`, `NO_OBVIOUS_DIFFICULTY`, `NORMAL_REFERENCE`, `UNCERTAIN`. 복수는 세미콜론으로 구분할 수 있습니다.

`sequence_severity`: `ISOLATED`, `SHORT_RUN`, `LONG_RUN`, `MOSTLY_MISSING`, `FULL_CLIP_MISSING`, `NORMAL_REFERENCE`. 이 값은 **사람의 시각 판단**이며 자동 run length와 별개입니다.

`manual_category`:

- `EXPECTED_DETECTOR_MISS`: 심한 profile·가림·어둠·얼굴 부재 등 사람이 봐도 검출이 어려운 조건
- `LIKELY_FALSE_NEGATIVE`: 사람이 볼 때 충분히 식별 가능한 얼굴을 YuNet이 놓친 경우
- `MIXED_MISSING_PATTERN`: 같은 영상/구간에 두 유형이 혼재
- `SOURCE_CONTENT_ISSUE`: 운전자/얼굴이 카메라 시야 밖에 있는 등 원본 내용의 문제
- `POSSIBLE_PIPELINE_ISSUE`: frame·영상 불일치, bbox/metadata 불일치, 이미지 손상 의심
- `NORMAL_REFERENCE`: 결측 없는 정상 비교 영상
- `UNCERTAIN`: 근거 부족

`review_decision`: `PASS`는 관찰된 결측이 source/detector 특성으로 설명되거나 정상 참조, `CHECK_NEEDED`는 false negative·혼합 양상 등 STEP 4-C 검토 필요, `INVESTIGATE`는 정책 결정 전 source/pipeline 이상 조사가 필요한 경우입니다. 의심 근거는 `review_note`에 적으세요.

STEP 4-B는 검토 자료와 사람의 판정을 준비하는 단계입니다. **이 CSV는 모두 미판정 상태로 생성되며, 결측 처리 정책·임계값은 여기서 바꾸지 않습니다.** Test split은 sealed입니다.
"""


def dry_run(plan: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    """영상·Context JPEG를 열거나 출력하지 않고 계획을 보여준다."""
    candidates = [item["candidate"] for item in plan]
    return {
        "mode": "DRY_RUN", "candidate_count": len(plan),
        "video_ids": [row["video_id"] for row in candidates],
        "split_counts": dict(Counter(row["split"] for row in candidates)),
        "label_counts": dict(Counter(row["label"] for row in candidates)),
        "selection_reason_counts": dict(Counter(reason for row in candidates for reason in row["selection_reason"].split(";"))),
        "planned_frame_indices": {item["candidate"]["video_id"]: item["selected_indices"] for item in plan},
        "planned_output_files": ["individual/000_...jpg through 035_...jpg", "contact_sheets/step4b_review_sheet_01.jpg through 09.jpg",
                                 "step4b_review_index.csv", "step4b_manual_review.csv", "STEP4B_MANUAL_REVIEW_GUIDE.md",
                                 "step4b_review_pack_summary.json", "step4b_review_pack_report.txt"],
        "test_rows": 0, "manual_fields_filled": 0, "raw_video_opened": False,
        "output_written": False, "output_dir": str(output_dir),
    }


def generate_pack(plan: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    """36개 후보의 필요한 원본 frame만 읽고 신규 디렉터리를 원자적으로 발행한다."""
    if len(plan) != 36 or output_dir.exists():
        raise ValueError("정확히 36개 후보가 필요하며 기존 검토 팩은 덮어쓰지 않습니다")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step4b_staging_", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        individual_dir = staging / "individual"
        individual_dir.mkdir()
        individual_images: list[np.ndarray] = []
        index_rows: list[dict[str, Any]] = []
        manual_rows: list[dict[str, Any]] = []
        frames_extracted = 0
        for item in plan:
            candidate = item["candidate"]
            review_id = int(candidate["review_order"])
            raw = _read_source_frames(item["source"], item["selected_rows"])
            frames_extracted += len(raw)
            sheet = render_individual(item, raw)
            individual_images.append(sheet)
            name = f"{review_id:03d}_{candidate['video_id']}_{candidate['selection_reason'].split(';')[0]}.jpg"
            relative = f"individual/{name}"
            _save_jpeg(individual_dir / name, sheet)
            selected = ";".join(str(index) for index in item["selected_indices"])
            index_rows.append({
                "review_id": review_id, "video_id": candidate["video_id"], "split": candidate["split"],
                "label": candidate["label"], "selection_reason": candidate["selection_reason"],
                "yunet_missing_count": candidate["yunet_missing_count"],
                "longest_missing_run": candidate["longest_missing_run"],
                "context_missing_count": candidate["context_missing_count"],
                "selected_canonical_indices": selected, "individual_sheet_path": relative,
                "contact_sheet_id": f"step4b_review_sheet_{review_id // 4 + 1:02d}.jpg",
            })
            manual_rows.append({
                "review_id": review_id, "video_id": candidate["video_id"], "split": candidate["split"],
                "label": candidate["label"], "selection_reason": candidate["selection_reason"],
                "yunet_missing_count": candidate["yunet_missing_count"],
                "yunet_missing_rate": candidate["yunet_missing_rate"],
                "longest_missing_run": candidate["longest_missing_run"],
                "number_of_missing_runs": item["quality"]["number_of_yunet_missing_runs"],
                "context_missing_count": candidate["context_missing_count"],
                "context_missing_rate": item["quality"]["context_missing_rate"],
                "behavior_valid_rate": candidate["behavior_valid_rate"],
                "selected_frame_indices": selected, **{field: "" for field in MANUAL_FIELDS},
            })
        contact_names = make_contact_sheets(individual_images, staging / "contact_sheets")
        _write_csv(staging / "step4b_review_index.csv", INDEX_COLUMNS, index_rows)
        _write_csv(staging / "step4b_manual_review.csv", MANUAL_AUTO_COLUMNS + MANUAL_FIELDS, manual_rows)
        (staging / "STEP4B_MANUAL_REVIEW_GUIDE.md").write_text(_guide(), encoding="utf-8")
        candidates = [item["candidate"] for item in plan]
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(plan), "individual_sheet_count": len(individual_images),
            "contact_sheet_count": len(contact_names),
            "split_counts": dict(Counter(row["split"] for row in candidates)),
            "label_counts": dict(Counter(row["label"] for row in candidates)),
            "selection_reason_counts": dict(Counter(reason for row in candidates for reason in row["selection_reason"].split(";"))),
            "raw_videos_opened": len(plan), "frames_extracted": frames_extracted,
            "yunet_inference_rerun": False, "dlib_inference_rerun": False,
            "context_crop_regenerated": False, "test_used": False,
            "manual_fields_filled": 0, "decision": "WAITING_FOR_MANUAL_REVIEW",
            "missing_policy_selected": False,
            "n_246_included": "n_246" in {row["video_id"] for row in candidates},
            "zero_missing_reference_count": sum(row["selection_reason"] == "zero_missing_reference" for row in candidates),
        }
        (staging / "step4b_review_pack_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = "\n".join([
            "STEP 4-B Manual Visual Review Pack", "", "[Scope]",
            "STEP 4-A의 기존 36개 deterministic 후보만 사용. raw frame은 시각화용으로만 읽음.",
            "", "[Candidate Validation]", "36개 고유 train/val 영상, test 0개. 후보 재선정 없음.",
            "", "[Selection Reason Distribution]", json.dumps(summary["selection_reason_counts"], ensure_ascii=False),
            "", "[Split / Label Distribution]", f"split={summary['split_counts']}, label={summary['label_counts']}",
            "", "[Frame Selection Rules]", "full miss: k0/25/50/75/99; long/high miss: run 전·시작·중간·끝·후; Context: 초·중·후 결측 및 저장 crop 참조; isolated: 전·결측·후; normal: k0/50/99. 영상당 최대 5 frame.",
            "", "[Generated Images]", f"individual {len(individual_images)}장, contact {len(contact_names)}장. 저장 YuNet bbox와 기존 Context JPEG만 사용.",
            "", "[Raw Video Access]", f"열람 영상 {len(plan)}개, 고유 원본 frame decode {frames_extracted}개. source frame index·크기·timestamp metadata 검증.",
            "", "[No-Inference Verification]", "YuNet/Dlib 재실행 없음; crop 재생성 없음; fake bbox 없음; test 접근 없음.",
            "", "[Manual Review Status]", "manual 필드 18개 모두 빈 값. 사용자 시각 판정 대기.",
            "", "[Limitations]", "선택 frame은 전체 100 frame의 일부이며, 보이지 않은 구간의 원인·결측 처리 정책을 단정하지 않음.",
            "", "[Decision]", "STEP 4-B REVIEW PACK GENERATED", "MANUAL REVIEW: WAITING", "MISSING POLICY: NOT SELECTED", "",
        ])
        (staging / "step4b_review_pack_report.txt").write_text(report, encoding="utf-8")
        staging.rename(output_dir)
    return summary
