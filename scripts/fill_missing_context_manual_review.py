"""사용자 제공 STEP 3-B 판정을 검증하고 수동 검토 칸에만 기록한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


MANUAL_FIELDS = (
    "manual_category", "detector_miss_plausible", "crop_storage_bug_suspected",
    "bbox_visible_to_human", "face_near_edge", "extreme_pose", "occlusion",
    "low_light", "blur", "review_decision", "review_note",
)
REQUIRED_FIELDS = ("review_id", *MANUAL_FIELDS)
REVIEW_SOURCE = "USER_PROVIDED_VISUAL_REVIEW"
REVIEW_METHOD = "DIRECT_VISUAL_INSPECTION"
PROFILE_NOTE = "측면 얼굴이며 코 끝 부분이 일부 잘려 보임. 강한 profile 조건으로 YuNet 미검출이 납득 가능한 사례."
FALSE_NEGATIVE_NOTE = "육안으로 얼굴이 충분히 식별되며 특별한 검출 난이도가 보이지 않음. YuNet false negative 가능성이 높은 사례."


def fill_review(csv_bytes: bytes, review_date: str) -> tuple[bytes, dict]:
    """입력을 모두 검증한 뒤 자동 칸의 값을 보존한 새 CSV와 집계를 반환한다."""
    source_hash = hashlib.sha256(csv_bytes).hexdigest()
    source_text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(source_text, newline=""))
    columns = reader.fieldnames
    if columns is None or len(columns) != len(set(columns)) or any(field not in columns for field in REQUIRED_FIELDS):
        raise ValueError("필수 열이 없거나 중복 열이 있습니다.")
    rows = list(reader)
    if len(rows) != 27:
        raise ValueError(f"수동 검토 대상은 27행이어야 합니다: {len(rows)}행")

    ids = []
    conflicts = []
    for row_number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV 열 개수가 맞지 않습니다: {row_number}행")
        raw_id = row["review_id"]
        if not raw_id or not raw_id.isdecimal():
            raise ValueError(f"review_id가 없거나 숫자가 아닙니다: {row_number}행")
        ids.append(int(raw_id))
        filled = [field for field in MANUAL_FIELDS if row[field].strip()]
        if filled:
            conflicts.append({"review_id": raw_id, "fields": filled})
    if conflicts:
        raise ValueError(f"기존 수동 판정과 충돌합니다. 자동 덮어쓰지 않습니다: {conflicts}")
    if len(set(ids)) != 27 or set(ids) != set(range(27)):
        raise ValueError(f"review_id는 중복 없이 0~26이어야 합니다: {ids}")

    automatic_before = [{key: row[key] for key in columns if key not in MANUAL_FIELDS} for row in rows]
    for row, review_id in zip(rows, ids):
        profile = 12 <= review_id <= 23
        row.update({
            "manual_category": "EXPECTED_DETECTOR_MISS" if profile else "LIKELY_FALSE_NEGATIVE",
            "detector_miss_plausible": str(profile).upper(),
            "crop_storage_bug_suspected": "FALSE",
            "bbox_visible_to_human": "TRUE",
            "face_near_edge": "FALSE",
            "extreme_pose": str(profile).upper(),
            "occlusion": "FALSE",
            "low_light": "FALSE",
            "blur": "FALSE",
            "review_decision": "PASS" if profile else "CHECK_NEEDED",
            "review_note": PROFILE_NOTE if profile else FALSE_NEGATIVE_NOTE,
        })
    automatic_after = [{key: row[key] for key in columns if key not in MANUAL_FIELDS} for row in rows]
    assert automatic_before == automatic_after

    categories = Counter(row["manual_category"] for row in rows)
    decisions = Counter(row["review_decision"] for row in rows)
    if categories != {"EXPECTED_DETECTOR_MISS": 12, "LIKELY_FALSE_NEGATIVE": 15}:
        raise ValueError(f"예상 수동 분류 수가 맞지 않습니다: {categories}")
    if decisions != {"PASS": 12, "CHECK_NEEDED": 15}:
        raise ValueError(f"예상 판정 수가 맞지 않습니다: {decisions}")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    result = output.getvalue().encode("utf-8")
    summary = {
        "manual_review_complete": True,
        "reviewed_cases": 27,
        "expected_detector_miss": 12,
        "likely_false_negative": 15,
        "possible_storage_or_pipeline_issue": 0,
        "uncertain": 0,
        "pass": 12,
        "check_needed": 15,
        "investigate": 0,
        "storage_bug_suspected": 0,
        "review_source": REVIEW_SOURCE,
        "review_method": REVIEW_METHOD,
        "review_date": review_date,
        "review_id_rule": "12-23 inclusive: EXPECTED_DETECTOR_MISS; all other IDs: LIKELY_FALSE_NEGATIVE",
        "original_manual_csv_sha256": source_hash,
        "updated_manual_csv_sha256": hashlib.sha256(result).hexdigest(),
        "automatic_columns_preserved": True,
        "original_automatic_summary_preserved": True,
    }
    return result, summary


def make_report(summary: dict) -> str:
    """수동 판정의 출처와 해석 범위를 명시한 별도 보고서를 만든다."""
    return f"""STEP 3-B Missing Context 수동 검토 보고서

[Scope]
Total missing context cases: 27
Review source: {summary['review_source']}
Review method: {summary['review_method']}
Review date: {summary['review_date']}
Original manual CSV SHA256: {summary['original_manual_csv_sha256']}
기존 자동 열과 자동 summary는 변경하지 않았다.

[Manual Review Result]
EXPECTED_DETECTOR_MISS: 12 / 27
LIKELY_FALSE_NEGATIVE: 15 / 27
POSSIBLE_STORAGE_OR_PIPELINE_ISSUE: 0 / 27
UNCERTAIN: 0 / 27
PASS: 12 / CHECK_NEEDED: 15 / INVESTIGATE: 0

[Expected Detector Miss Pattern]
review_id: 12-23 (CSV의 실제 ID, 행 위치 아님)
사용자 관찰: 측면/profile 얼굴, 코 끝 일부가 잘려 보임. YuNet 미검출이 납득 가능함.
face_near_edge=FALSE는 frame edge 근거가 없다는 뜻이며 코 끝 관찰을 frame edge로 해석하지 않는다.

[Likely False Negative Pattern]
나머지 review_id 15건. 사용자는 얼굴이 뚜렷하고 큰 pose·가림·저조도·blur 문제 없이 YuNet이 놓친 것으로 판정했다.

[Pipeline Integrity]
사용자 검토에서 storage corruption, frame mapping, wrong-person crop, JPEG corruption, context storage failure의 증거가 보고되지 않았다.
Pipeline/storage issue suspected: 0 / 27

[Interpretation]
Pilot Context 결측은 기록상 detector 단계에서 발생했다. 12건은 납득 가능한 miss, 15건은 likely YuNet false negative다.
이는 사용자 제공 시각 판정이며 Codex의 이미지 재판독 결과가 아니다.
기존 6장 pilot 구현 시각 검토에 대한 사용자 제공 결론도 frame/person alignment, RGB/BGR, JPEG, SQUARE_M10 이상 없음이다.
STEP 3-B 수동 시각 검토 완료. STEP 2 frozen policy 변경 없음.
YuNet 교체·fallback·threshold 변경·frame 대체는 하지 않는다.
전체 train/val missing pattern은 STEP 4에서 별도로 정량화한다. STEP 3-C 및 test 처리는 시작하지 않았다.
"""


def apply_review(review_dir: Path, review_date: str | None = None) -> dict:
    """충돌과 기존 결과 파일을 검사한 뒤 CSV·별도 summary·report를 발행한다."""
    csv_path = review_dir / "missing_context_manual_review.csv"
    summary_path = review_dir / "missing_context_manual_review_summary.json"
    report_path = review_dir / "missing_context_manual_review_report.txt"
    for path in (summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"이미 수동 검토 결과가 있습니다: {path}")
    original = csv_path.read_bytes()
    updated, summary = fill_review(original, review_date or datetime.now().astimezone().date().isoformat())
    report = make_report(summary)
    if csv_path.read_bytes() != original:
        raise RuntimeError("검증 중 수동 CSV가 변경되어 작업을 중지합니다.")
    with NamedTemporaryFile(dir=review_dir, prefix=".manual_review_", suffix=".csv", delete=False) as temporary:
        temp_path = Path(temporary.name)
        temporary.write(updated)
    try:
        os.replace(temp_path, csv_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(report, encoding="utf-8")
    finally:
        temp_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    """기본 pilot missing-only 디렉터리에 사용자 판정을 반영한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review")
    args = parser.parse_args()
    print(json.dumps(apply_review(args.review_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
