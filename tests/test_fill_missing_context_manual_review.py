"""사용자 제공 수동 판정의 ID 매핑·충돌·자동 필드 보존을 검증한다."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fill_missing_context_manual_review.py"
SPEC = importlib.util.spec_from_file_location("fill_missing_context_manual_review", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_csv(ids=range(27), *, conflict_id=None) -> bytes:
    """행 순서를 거꾸로 해도 실제 review_id만 사용하도록 합성한다."""
    output = io.StringIO(newline="")
    columns = ["review_id", "split", "detector_success", *MODULE.MANUAL_FIELDS]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for index in ids:
        row = {key: "" for key in columns}
        row.update(review_id=f"{index:03d}", split="train", detector_success="False")
        if index == conflict_id:
            row["review_note"] = "기존 사용자 판정"
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def parsed(data: bytes) -> dict[int, dict]:
    """출력의 ID별 값을 읽는다."""
    return {int(row["review_id"]): row for row in csv.DictReader(io.StringIO(data.decode("utf-8")))}


def test_review_id_mapping_counts_and_automatic_columns() -> None:
    original = source_csv(reversed(range(27)))
    updated, summary = MODULE.fill_review(original, "2026-09-20")
    rows = parsed(updated)
    assert len(rows) == 27
    assert {index for index, row in rows.items() if row["manual_category"] == "EXPECTED_DETECTOR_MISS"} == set(range(12, 24))
    assert {index for index, row in rows.items() if row["manual_category"] == "LIKELY_FALSE_NEGATIVE"} == set(range(27)) - set(range(12, 24))
    assert all(row["review_decision"] == ("PASS" if 12 <= index <= 23 else "CHECK_NEEDED") for index, row in rows.items())
    assert all(row["face_near_edge"] == "FALSE" and row["crop_storage_bug_suspected"] == "FALSE" for row in rows.values())
    assert all(row["split"] == "train" and row["detector_success"] == "False" for row in rows.values())
    assert summary["expected_detector_miss"] == 12
    assert summary["likely_false_negative"] == 15
    assert summary["possible_storage_or_pipeline_issue"] == summary["uncertain"] == 0
    assert summary["pass"] == 12 and summary["check_needed"] == 15
    assert summary["original_manual_csv_sha256"] == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("ids", [range(26), [*range(26), 25], [*range(26), 27]])
def test_invalid_count_duplicate_or_missing_review_id(ids) -> None:
    with pytest.raises(ValueError):
        MODULE.fill_review(source_csv(ids), "2026-09-20")


def test_existing_manual_judgment_is_not_overwritten() -> None:
    with pytest.raises(ValueError, match="충돌"):
        MODULE.fill_review(source_csv(conflict_id=12), "2026-09-20")


def test_summary_report_generation_and_no_reapply(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_context_manual_review.csv"
    original = source_csv()
    csv_path.write_bytes(original)
    summary = MODULE.apply_review(tmp_path, "2026-09-20")
    saved = json.loads((tmp_path / "missing_context_manual_review_summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "missing_context_manual_review_report.txt").read_text(encoding="utf-8")
    assert saved == summary
    assert "EXPECTED_DETECTOR_MISS: 12 / 27" in report
    assert "LIKELY_FALSE_NEGATIVE: 15 / 27" in report
    assert "USER_PROVIDED_VISUAL_REVIEW" in report
    assert parsed(csv_path.read_bytes())[12]["review_decision"] == "PASS"
    with pytest.raises(FileExistsError):
        MODULE.apply_review(tmp_path, "2026-09-20")
