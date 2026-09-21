"""사용자 제공 STEP 4-B 수동 판정의 매핑·충돌 방지·출처 기록을 검증한다."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from drowsiness_detection.evaluation_v2 import step4b_manual_review as manual
from drowsiness_detection.evaluation_v2.step4b_visual_review import MANUAL_AUTO_COLUMNS, MANUAL_FIELDS


def _source() -> str:
    """실제 영상 없이 00~35 형식의 사용자 확정 판정 원문을 만든다."""
    blocks = []
    for first in list(range(30)) + [30]:
        is_reference = first == 30
        category = next(name for name, ids in manual.CATEGORY_IDS.items() if first in ids)
        decision = "CHECK_NEEDED" if category in {"LIKELY_FALSE_NEGATIVE", "MIXED_MISSING_PATTERN"} else "PASS"
        values = {field: "FALSE" for field in manual.BOOLEAN_FIELDS}
        values.update({"manual_pattern": "NORMAL_REFERENCE" if is_reference else "PROFILE_YAW",
                       "face_visible_during_miss": "N.A." if is_reference else "PARTIAL" if first == 1 else "TRUE",
                       "detector_miss_plausible": "N.A." if is_reference else "TRUE",
                       "manual_category": category, "review_decision": decision,
                       "review_note": '"사용자가 확인한 시각 판정."'})
        fields = [field for field in MANUAL_FIELDS if field != "sequence_severity"]
        name = "REVIEW 30~35" if is_reference else f"REVIEW {first:02d}"
        blocks.append(name + "\n" + "\n".join(f"{field}:\n{values[field]}\n" for field in fields))
    return "\n".join(blocks) + "\n==================================================\n"


def _template(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_AUTO_COLUMNS + MANUAL_FIELDS)
        writer.writeheader()
        for review_id in range(36):
            row = {column: "" for column in MANUAL_AUTO_COLUMNS + MANUAL_FIELDS}
            row.update({"review_id": str(review_id), "video_id": "n_246" if review_id == 1 else f"v{review_id}",
                        "split": "train", "label": "drowsy", "selection_reason": "highest_missing_count",
                        "yunet_missing_count": "100" if review_id == 1 else "3",
                        "yunet_missing_rate": "1" if review_id == 1 else "0.03",
                        "longest_missing_run": "100" if review_id == 1 else "1",
                        "number_of_missing_runs": "1", "context_missing_count": "0",
                        "context_missing_rate": "0", "behavior_valid_rate": "0.97",
                        "selected_frame_indices": "0;50;99"})
            writer.writerow(row)
            rows.append(row)
    return rows


def test_parse_category_decision_and_schema_conversions() -> None:
    verdicts, info = manual.parse_user_verdicts(_source())
    assert len(verdicts) == 36
    assert verdicts[1]["manual_category"] == "EXPECTED_DETECTOR_MISS"
    assert verdicts[1]["sequence_severity"] == "FULL_CLIP_MISSING"
    assert verdicts[1]["face_visible_during_miss"] == "TRUE"
    assert verdicts[30]["face_visible_during_miss"] == ""
    assert verdicts[30]["detector_miss_plausible"] == ""
    assert verdicts[29]["sequence_severity"] == "ISOLATED"
    assert len(info["schema_conversions"]) == 13


def test_parse_rejects_missing_or_wrong_mapping() -> None:
    with pytest.raises(ValueError, match="31개"):
        manual.parse_user_verdicts(_source().replace("REVIEW 29", "REVIEW XX"))
    with pytest.raises(ValueError, match="category mapping"):
        manual.parse_user_verdicts(_source().replace("manual_category:\nLIKELY_FALSE_NEGATIVE", "manual_category:\nEXPECTED_DETECTOR_MISS", 1))


def test_template_conflict_and_test_rejection(tmp_path: Path) -> None:
    path = tmp_path / "step4b_manual_review.csv"
    rows = _template(path)
    manual.validate_blank_template(MANUAL_AUTO_COLUMNS + MANUAL_FIELDS, rows)
    rows[0]["manual_category"] = "EXPECTED_DETECTOR_MISS"
    with pytest.raises(ValueError, match="덮어쓰지"):
        manual.validate_blank_template(MANUAL_AUTO_COLUMNS + MANUAL_FIELDS, rows)
    rows[0]["manual_category"] = ""
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="STEP_4B_BLOCKED_TEST_LEAKAGE"):
        manual.validate_blank_template(MANUAL_AUTO_COLUMNS + MANUAL_FIELDS, rows)


def test_apply_preserves_auto_fields_and_generates_provenance(tmp_path: Path) -> None:
    review_csv = tmp_path / "step4b_manual_review.csv"
    before_rows = _template(review_csv)
    before_hash = hashlib.sha256(review_csv.read_bytes()).hexdigest()
    source = tmp_path / "user_review.txt"
    source.write_text(_source(), encoding="utf-8")
    result = manual.apply_manual_review(review_csv, source, tmp_path)
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        after_rows = list(csv.DictReader(handle))
    assert len(after_rows) == 36
    assert all(before[field] == after[field] for before, after in zip(before_rows, after_rows, strict=True)
               for field in MANUAL_AUTO_COLUMNS)
    assert all(after["manual_pattern"] and after["manual_category"] and after["review_decision"]
               for after in after_rows)
    assert result["summary"]["expected_detector_miss"] == 13
    assert result["summary"]["likely_false_negative"] == 12
    assert result["summary"]["mixed_missing_pattern"] == 5
    assert result["summary"]["normal_reference"] == 6
    assert result["summary"]["pass"] == 19
    assert result["summary"]["check_needed"] == 17
    assert result["summary"]["pipeline_issue_suspected"] == 0
    provenance = json.loads((tmp_path / "step4b_manual_review_provenance.json").read_text(encoding="utf-8"))
    assert provenance["manual_csv_before_sha256"] == before_hash
    assert provenance["manual_csv_after_sha256"] == hashlib.sha256(review_csv.read_bytes()).hexdigest()
    assert provenance["automatic_metadata_unchanged"] is True
    assert provenance["codex_visual_rejudgment"] is False
    assert "STEP 4-B: COMPLETE" in (tmp_path / "step4b_manual_review_report.txt").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        manual.apply_manual_review(review_csv, source, tmp_path)
