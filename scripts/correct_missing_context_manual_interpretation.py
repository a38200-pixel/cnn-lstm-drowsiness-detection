"""사용자 후속 시각 검토에 따른 STEP 3-B pose 해석을 정정한다."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


PROFILE_NOTE = "강한 측면/profile pose이며 얼굴이 frame edge 방향에 가깝고 코 끝 부분이 일부 잘려 보인다. YuNet 미검출이 비교적 납득 가능한 어려운 detector 조건."
FALSE_NEGATIVE_NOTE = "Yaw/profile pose가 존재하지만 얼굴 전체와 주요 facial structure는 육안으로 충분히 식별 가능하다. 심각한 frame clipping, blur, occlusion은 관찰되지 않아 YuNet false negative 후보로 분류한다."
MANUAL_FIELDS = {
    "manual_category", "detector_miss_plausible", "crop_storage_bug_suspected",
    "bbox_visible_to_human", "face_near_edge", "extreme_pose", "occlusion",
    "low_light", "blur", "review_decision", "review_note",
}


def corrected_csv(data: bytes) -> tuple[bytes, dict]:
    """실제 ID·기존 분류·자동 열을 검증하고 관찰 해석 열만 바꾼다."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames
    if columns is None or len(columns) != len(set(columns)) or not {"review_id", *MANUAL_FIELDS} <= set(columns):
        raise ValueError("필수 수동 검토 열이 없거나 중복 열이 있습니다.")
    rows = list(reader)
    if len(rows) != 27 or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV는 정확히 27개의 정상 행이어야 합니다.")
    if any(not row["review_id"].isdecimal() for row in rows):
        raise ValueError("숫자가 아닌 review_id가 있습니다.")
    ids = [int(row["review_id"]) for row in rows]
    if set(ids) != set(range(27)) or len(set(ids)) != 27:
        raise ValueError("review_id 0~26이 중복 없이 있어야 합니다.")
    before_auto = [{key: row[key] for key in columns if key not in MANUAL_FIELDS} for row in rows]
    for row, review_id in zip(rows, ids):
        profile = 12 <= review_id <= 23
        expected_category = "EXPECTED_DETECTOR_MISS" if profile else "LIKELY_FALSE_NEGATIVE"
        expected_decision = "PASS" if profile else "CHECK_NEEDED"
        if (row["manual_category"] != expected_category or row["review_decision"] != expected_decision
                or row["detector_miss_plausible"] != ("TRUE" if profile else "FALSE")
                or row["crop_storage_bug_suspected"] != "FALSE"):
            raise ValueError(f"기존 category/decision 또는 핵심 판정과 충돌합니다: review_id={review_id}")
        if profile:
            row["face_near_edge"] = "TRUE"
            row["review_note"] = PROFILE_NOTE
        else:
            # 사용자는 15건 각각의 extreme pose 여부를 확정하지 않았다.
            row["extreme_pose"] = ""
            row["review_note"] = FALSE_NEGATIVE_NOTE
    after_auto = [{key: row[key] for key in columns if key not in MANUAL_FIELDS} for row in rows]
    if before_auto != after_auto:
        raise AssertionError("자동 metadata가 변경되었습니다.")
    categories = Counter(row["manual_category"] for row in rows)
    if categories != {"EXPECTED_DETECTOR_MISS": 12, "LIKELY_FALSE_NEGATIVE": 15}:
        raise ValueError("기존 12/15 category count가 변경되었습니다.")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    result = output.getvalue().encode("utf-8")
    return result, {
        "before_csv_sha256": hashlib.sha256(data).hexdigest(),
        "after_csv_sha256": hashlib.sha256(result).hexdigest(),
        "automatic_columns_preserved": True,
        "expected_detector_miss": 12,
        "likely_false_negative": 15,
        "possible_storage_or_pipeline_issue": 0,
        "uncertain": 0,
    }


def correction_markdown(provenance: dict) -> str:
    """원본·정정본 해시와 사용자 후속 판독 근거를 기록한다."""
    return f"""# STEP 3-B Manual Review Interpretation Correction

## Reason

초기 입력에서 LIKELY_FALSE_NEGATIVE 15건을 '특별한 pose difficulty 없음'으로 일괄 표현한 것은 지나치게 단순했다. 사용자 후속 시각 검토에서는 상당수 결측 frame에 yaw/profile pose가 있었다. Codex가 이미지를 재판독하거나 새 per-frame pose label을 만들지는 않았다.

## Category Counts

변경 없음: EXPECTED_DETECTOR_MISS 12, LIKELY_FALSE_NEGATIVE 15, PIPELINE/STORAGE 0, UNCERTAIN 0. 총 27건.

## Corrected Interpretation

- 실제 `review_id` 12~23: strong profile, frame edge 방향에 가까운 얼굴, 일부 코 끝 clipping. YuNet miss가 납득 가능한 가장 어려운 그룹. `face_near_edge=TRUE`는 사용자 후속 관찰에 근거한다.
- 나머지 15건: 상당수에 yaw/profile pose가 있지만 얼굴과 주요 특징은 육안으로 충분히 보인다. 심한 clipping·blur·occlusion·corruption은 관찰되지 않아 likely false negative 후보를 유지한다. 개별 extreme pose 여부는 확정되지 않아 `extreme_pose`를 빈 값(미평가)으로 둔다. 빈 값은 정면이나 FALSE를 의미하지 않는다.

## Provenance

- Source: `USER_PROVIDED_VISUAL_REVIEW`, 후속 해석 정정
- Method: `DIRECT_VISUAL_INSPECTION`
- Correction date: {provenance['correction_date']}
- Before CSV SHA-256: `{provenance['before_csv_sha256']}`
- After CSV SHA-256: `{provenance['after_csv_sha256']}`
- Automatic metadata preserved: `{str(provenance['automatic_columns_preserved']).lower()}`
- 기존 자동 summary 및 STEP 2 frozen policy는 변경하지 않았다.

## Policy Impact

NONE. YuNet primary와 Dlib68 RAW bbox·SQUARE_M10 정책 유지. Pilot 표본의 관찰을 전체 데이터셋에 일반화하지 않는다.

## Future Use

STEP 3-C 전체 train/val materialization 후 STEP 4에서 missing rate, consecutive misses, pose 관련 패턴, 영상별 집중도를 분석할 때 참고한다. Test는 sealed다.
"""


def correction_section() -> str:
    """기존 수동 보고서에 덧붙일 정정 단락이다."""
    return """
[Interpretation Correction — Revision 2]
기존 12/15 category count는 변경하지 않았다. 이전의 '15건에 큰 pose 문제가 없다'는 일괄 설명은 이 정정이 대체한다.
사용자 후속 시각 검토에 따르면 27건에서 yaw/profile pose가 중요한 공통 패턴이며, 이를 pose와 무관한 무작위 YuNet 실패로 해석하지 않는다.
review_id 12-23: strong profile + frame edge 방향 근접 + 일부 코 끝 clipping. Detector miss가 비교적 납득 가능하다.
나머지 15건: 상당수에 yaw/profile 난이도가 있지만 얼굴·주요 구조가 육안으로 충분히 보이므로 likely false negative 후보를 유지한다.
이 15건의 extreme_pose 빈 값은 per-frame 미평가이며 frontal pose 판정이 아니다.
Pipeline/storage 의심 0건. YuNet frozen policy 변경 없음. 20-video pilot 결과를 전체 데이터셋에 일반화하지 않는다.
"""


def apply_correction(review_dir: Path, correction_date: str | None = None) -> dict:
    """기존 파일을 검증한 뒤 CSV·수동 summary·보고서와 별도 provenance를 갱신한다."""
    csv_path = review_dir / "missing_context_manual_review.csv"
    summary_path = review_dir / "missing_context_manual_review_summary.json"
    report_path = review_dir / "missing_context_manual_review_report.txt"
    provenance_path = review_dir / "manual_review_interpretation_correction.md"
    if provenance_path.exists():
        raise FileExistsError(f"이미 정정 provenance가 있습니다: {provenance_path}")
    original = csv_path.read_bytes()
    result, provenance = corrected_csv(original)
    provenance["correction_date"] = correction_date or datetime.now().astimezone().date().isoformat()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("reviewed_cases") != 27 or summary.get("interpretation_revision", 1) != 1:
        raise ValueError("기존 수동 summary 상태와 충돌합니다.")
    report = report_path.read_text(encoding="utf-8")
    if "[Interpretation Correction" in report:
        raise ValueError("이미 보고서에 정정이 기록되었습니다.")
    summary.update({
        "interpretation_revision": 2,
        "category_counts_changed": False,
        "pose_related_pattern_observed": True,
        "strongest_difficulty_group": {
            "review_ids": list(range(12, 24)),
            "pattern": "strong_profile_near_edge_partial_nose_clipping",
        },
        "likely_false_negative_interpretation": "Mostly pose-related misses where the face remained sufficiently visible to a human.",
        "extreme_pose_remaining_15": "NOT_ASSESSED_PER_FRAME; blank in CSV does not mean frontal pose",
        "policy_changed": False,
        "correction_source": "USER_PROVIDED_VISUAL_REVIEW",
        "correction_date": provenance["correction_date"],
        "pre_correction_manual_csv_sha256": provenance["before_csv_sha256"],
        "updated_manual_csv_sha256": provenance["after_csv_sha256"],
        "automatic_columns_preserved": True,
    })
    # 원본 수동 결과·수치는 유지하되, 오래된 pose 해석 문장만 정정한다.
    report = report.replace(
        "face_near_edge=FALSE는 frame edge 근거가 없다는 뜻이며 코 끝 관찰을 frame edge로 해석하지 않는다.",
        "후속 사용자 판독에서 얼굴이 frame edge 방향에 가깝다고 확인했다. 세부 근거는 아래 정정 단락을 따른다.",
    ).replace(
        "나머지 review_id 15건. 사용자는 얼굴이 뚜렷하고 큰 pose·가림·저조도·blur 문제 없이 YuNet이 놓친 것으로 판정했다.",
        "나머지 review_id 15건. 후속 사용자 판독에서 상당수의 yaw/profile 난이도가 확인됐지만 얼굴 주요 구조는 육안으로 충분히 보였다.",
    ) + correction_section()
    if csv_path.read_bytes() != original:
        raise RuntimeError("검증 중 CSV가 변경되어 중지합니다.")
    csv_path.write_bytes(result)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    provenance_path.write_text(correction_markdown(provenance), encoding="utf-8")
    return provenance


def main() -> None:
    """기본 missing-only review 산출물에 후속 사용자 판정을 적용한다."""
    root = Path(__file__).resolve().parents[1]
    review_dir = root / "outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review"
    print(json.dumps(apply_correction(review_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
