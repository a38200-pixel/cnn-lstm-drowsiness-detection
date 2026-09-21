"""사용자가 확정한 STEP 4-B 시각 판정을 자동 metadata를 보존하며 기록한다."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .step4b_visual_review import MANUAL_AUTO_COLUMNS, MANUAL_FIELDS


CATEGORY_IDS = {
    "EXPECTED_DETECTOR_MISS": {0, 1, 3, 6, 7, 11, 14, 15, 17, 19, 20, 23, 24},
    "LIKELY_FALSE_NEGATIVE": {5, 10, 12, 13, 18, 21, 22, 25, 26, 27, 28, 29},
    "MIXED_MISSING_PATTERN": {2, 4, 8, 9, 16},
    "NORMAL_REFERENCE": set(range(30, 36)),
}
BOOLEAN_FIELDS = set(MANUAL_FIELDS) - {
    "manual_pattern", "face_visible_during_miss", "sequence_severity",
    "manual_category", "review_decision", "review_note",
}
BOOLEAN_FIELDS.add("face_visible_during_miss")
EXPECTED_COUNTS = {
    "EXPECTED_DETECTOR_MISS": 13, "LIKELY_FALSE_NEGATIVE": 12,
    "MIXED_MISSING_PATTERN": 5, "NORMAL_REFERENCE": 6,
}


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        return columns, list(reader)


def validate_blank_template(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    """충돌·test leakage를 수정 전에 차단한다."""
    if len(rows) != 36 or len({row.get("review_id") for row in rows}) != 36:
        raise ValueError("STEP 4-B 수동 CSV는 고유 review_id 36행이어야 합니다")
    if set(columns) != set(MANUAL_AUTO_COLUMNS + MANUAL_FIELDS):
        raise ValueError("STEP 4-B 수동 CSV schema가 기존 template와 다릅니다")
    if [int(row["review_id"]) for row in rows] != list(range(36)):
        raise ValueError("review_id는 0~35 순서여야 합니다")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("STEP_4B_BLOCKED_TEST_LEAKAGE: 수동 CSV에 test/unknown split")
    if len({row["video_id"] for row in rows}) != 36:
        raise ValueError("수동 CSV video_id 중복")
    conflicts = [row["review_id"] for row in rows if any(row.get(field, "") != "" for field in MANUAL_FIELDS)]
    if conflicts:
        raise ValueError(f"기존 수동 판정과 충돌하므로 덮어쓰지 않습니다: review_id={','.join(conflicts)}")


def _severity(review_id: int) -> str:
    if review_id == 0:
        return "MOSTLY_MISSING"
    if review_id == 1:
        return "FULL_CLIP_MISSING"
    if 2 <= review_id <= 23:
        return "LONG_RUN"
    if 24 <= review_id <= 29:
        return "ISOLATED"
    return "NORMAL_REFERENCE"


def parse_user_verdicts(source_text: str) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """첨부 본문 REVIEW 블록을 판독하되 시각적 재판단은 하지 않는다."""
    markers = list(re.finditer(r"(?m)^REVIEW (\d{2})(?:~(\d{2}))?\s*$", source_text))
    if len(markers) != 31:
        raise ValueError(f"review 판정 블록은 00~29와 30~35 공통 블록의 31개여야 합니다: {len(markers)}")
    verdicts: dict[int, dict[str, str]] = {}
    conversions: list[dict[str, str | int]] = []
    field_pattern = re.compile(r"(?m)^([a-z][a-z0-9_]*):[ \t]*\r?\n([^\r\n]+)")
    for position, marker in enumerate(markers):
        start = marker.end()
        end = markers[position + 1].start() if position + 1 < len(markers) else source_text.find("\n==================================================", start)
        if end < 0:
            end = len(source_text)
        block = source_text[start:end]
        found = {match.group(1): match.group(2).strip().strip('"') for match in field_pattern.finditer(block)}
        expected_fields = set(MANUAL_FIELDS) - {"sequence_severity"}
        if not expected_fields <= set(found):
            raise ValueError(f"REVIEW {marker.group(1)} 수동 필드 누락: {sorted(expected_fields - set(found))}")
        first = int(marker.group(1))
        last = int(marker.group(2)) if marker.group(2) else first
        if last < first or (first <= 29 and last != first) or (first == 30 and last != 35):
            raise ValueError("REVIEW 블록의 ID 범위가 올바르지 않습니다")
        for review_id in range(first, last + 1):
            if review_id in verdicts:
                raise ValueError(f"REVIEW {review_id} 중복")
            entry = {field: found[field] for field in expected_fields}
            entry["sequence_severity"] = _severity(review_id)
            for field in BOOLEAN_FIELDS:
                if entry[field] == "PARTIAL" and field == "face_visible_during_miss":
                    conversions.append({"review_id": review_id, "field": field, "from": "PARTIAL", "to": "TRUE"})
                    entry[field] = "TRUE"
                elif entry[field] == "N.A.":
                    conversions.append({"review_id": review_id, "field": field, "from": "N.A.", "to": ""})
                    entry[field] = ""
                elif entry[field] not in {"TRUE", "FALSE"}:
                    raise ValueError(f"REVIEW {review_id} boolean field 오류: {field}={entry[field]}")
            verdicts[review_id] = entry
    if set(verdicts) != set(range(36)):
        raise ValueError("REVIEW 00~35 판정이 모두 있어야 합니다")
    for category, ids in CATEGORY_IDS.items():
        if {review_id for review_id, entry in verdicts.items() if entry["manual_category"] == category} != ids:
            raise ValueError(f"사용자 제공 category mapping 불일치: {category}")
    for review_id, entry in verdicts.items():
        category = entry["manual_category"]
        expected_decision = "CHECK_NEEDED" if category in {"LIKELY_FALSE_NEGATIVE", "MIXED_MISSING_PATTERN"} else "PASS"
        if entry["review_decision"] != expected_decision or entry["pipeline_issue_suspected"] != "FALSE":
            raise ValueError(f"REVIEW {review_id} decision/pipeline 판정 불일치")
        if not entry["manual_pattern"] or not entry["review_note"]:
            raise ValueError(f"REVIEW {review_id} 수동 근거 누락")
    return verdicts, {"schema_conversions": conversions}


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    categories = Counter(row["manual_category"] for row in rows)
    decisions = Counter(row["review_decision"] for row in rows)
    if dict(categories) != EXPECTED_COUNTS or dict(decisions) != {"PASS": 19, "CHECK_NEEDED": 17}:
        raise ValueError("수동 category/decision 집계가 요청과 다릅니다")
    pipeline = sum(row["pipeline_issue_suspected"] == "TRUE" for row in rows)
    if pipeline or any(not row[field] for row in rows for field in MANUAL_FIELDS if field not in BOOLEAN_FIELDS):
        raise ValueError("수동 판정이 미완료이거나 pipeline issue 집계 불일치")
    if any(row["manual_category"] != "NORMAL_REFERENCE" for row in rows[30:36]):
        raise ValueError("정상 참조 review_id 30~35 불일치")
    n246 = next((row for row in rows if row["video_id"] == "n_246"), None)
    if n246 is None or n246["review_id"] != "1" or n246["manual_category"] != "EXPECTED_DETECTOR_MISS" or n246["sequence_severity"] != "FULL_CLIP_MISSING":
        raise ValueError("n_246 판정/강도 불일치")
    return {
        "manual_review_complete": True, "reviewed_candidates": 36,
        "expected_detector_miss": categories["EXPECTED_DETECTOR_MISS"],
        "likely_false_negative": categories["LIKELY_FALSE_NEGATIVE"],
        "mixed_missing_pattern": categories["MIXED_MISSING_PATTERN"],
        "normal_reference": categories["NORMAL_REFERENCE"],
        "source_content_issue": 0, "possible_pipeline_issue": 0, "uncertain": 0,
        "pass": decisions["PASS"], "check_needed": decisions["CHECK_NEEDED"],
        "investigate": 0, "pipeline_issue_suspected": pipeline,
        "missing_policy_selected": False, "step4c_started": False,
        "test_used": False,
    }


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "STEP 4-B User-Confirmed Manual Visual Review", "",
        "[Scope]", "STEP 4-A의 목적 선정된 36개 후보와 contact sheet 9장에 대한 사용자·ChatGPT 보조 시각 검토 결과를 기록했다. Codex는 이미지를 재판독하지 않았다.",
        "", "[Category Counts]", "EXPECTED_DETECTOR_MISS: 13", "LIKELY_FALSE_NEGATIVE: 12", "MIXED_MISSING_PATTERN: 5", "NORMAL_REFERENCE: 6", "SOURCE_CONTENT_ISSUE: 0", "POSSIBLE_PIPELINE_ISSUE: 0", "UNCERTAIN: 0",
        "", "[Visual Patterns]", "선정 사례에서 강한 yaw/profile, head pitch·roll, 일부 frame edge·clipping, occlusion, motion blur가 관찰됐다. 얼굴이 충분히 보이는 false-negative 후보도 있다.",
        "", "[Long Consecutive Missing]", "긴 run에서 자세·머리 방향의 어려움이 반복 관찰됐으나, 얼굴이 분명한 상태의 연속 미검출도 있었다. n_246은 100/100 결측이며 FULL_CLIP_MISSING / EXPECTED_DETECTOR_MISS로 수동 판정됐다. 한 사례를 전체 결측 영상에 일반화하지 않는다.",
        "", "[Isolated Missing]", "일부 고립 결측은 일시적 motion blur와 함께 발생했지만, 일부는 앞뒤 frame과 거의 같은 조건에서도 발생해 likely false negative로 판정됐다.",
        "", "[Normal References]", "결측 0개 참조 영상 6개에서 저장 bbox와 Context crop의 명백한 시각적 불일치는 보고되지 않았다.",
        "", "[Pipeline Integrity]", "사용자 판정상 wrong frame/video, JPEG 손상, 명백한 bbox/storage mismatch 의심 0건. 이는 검토한 이미지 범위에 한정된다.",
        "", "[Interpretation]", "선정 사례의 결측은 자세 관련 난이도·납득 가능한 미검출·likely false negative·혼합 구간이 공존한다. 이 36개는 전체 1,763개 영상의 무작위 표본이 아니며 category 비율을 전체 detector 오류 비율로 해석하지 않는다. 단일 global missing rate만으로 sequence 처리 정책을 정하지 않는다.",
        "", "[Decision]", "STEP 4-B: COMPLETE", "MISSING POLICY: NOT SELECTED", "STEP 4-C: NOT STARTED", "TEST SPLIT: SEALED", "",
    ])


def apply_manual_review(csv_path: Path, source_path: Path, output_root: Path) -> dict[str, Any]:
    """충돌 검사 후 manual 18열만 적용하고 해시·집계·출처를 새 파일로 발행한다."""
    if csv_path.parent.resolve() != output_root.resolve():
        raise ValueError("manual CSV와 결과 디렉터리가 다릅니다")
    paths = {name: output_root / name for name in (
        "step4b_manual_review_summary.json", "step4b_manual_review_report.txt",
        "step4b_manual_review_provenance.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("기존 STEP 4-B 수동 결과를 덮어쓰지 않습니다")
    before_bytes = csv_path.read_bytes()
    columns, original = _read_csv(csv_path)
    validate_blank_template(columns, original)
    source_bytes = source_path.read_bytes()
    verdicts, parse_info = parse_user_verdicts(source_bytes.decode("utf-8-sig"))
    revised = []
    for row in original:
        changed = dict(row)
        changed.update(verdicts[int(row["review_id"])])
        revised.append(changed)
    if any(any(before[field] != after[field] for field in columns if field not in MANUAL_FIELDS)
           for before, after in zip(original, revised, strict=True)):
        raise ValueError("automatic metadata 변경 감지")
    summary = _summary(revised)
    after_bytes = _csv_bytes(columns, revised)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manual_review_source": "USER_CONFIRMED_CHATGPT_ASSISTED_VISUAL_REVIEW",
        "manual_review_method": "CONTACT_SHEET_VISUAL_INSPECTION",
        "reviewed_candidates": 36, "contact_sheets": 9,
        "source_attachment_sha256": _sha256(source_bytes),
        "manual_csv_before_sha256": _sha256(before_bytes),
        "manual_csv_after_sha256": _sha256(after_bytes),
        "automatic_metadata_unchanged": True,
        "automatic_inference_rerun": False, "test_used": False,
        "codex_visual_rejudgment": False,
        **parse_info,
    }
    with tempfile.TemporaryDirectory(prefix=".step4b_manual_staging_", dir=output_root) as temporary:
        temp = Path(temporary)
        staged_csv = temp / csv_path.name
        staged_csv.write_bytes(after_bytes)
        (temp / "step4b_manual_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp / "step4b_manual_review_report.txt").write_text(_report(summary), encoding="utf-8")
        (temp / "step4b_manual_review_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name, path in paths.items():
            os.replace(temp / name, path)
        os.replace(staged_csv, csv_path)
    return {"summary": summary, "provenance": provenance}
