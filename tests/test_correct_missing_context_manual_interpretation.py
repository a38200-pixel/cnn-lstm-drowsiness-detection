"""수동 해석 정정이 분류·자동 metadata·정책을 보존하는지 검증한다."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FILL_SPEC = importlib.util.spec_from_file_location("fill_review_for_correction", ROOT / "scripts/fill_missing_context_manual_review.py")
assert FILL_SPEC and FILL_SPEC.loader
FILL = importlib.util.module_from_spec(FILL_SPEC)
FILL_SPEC.loader.exec_module(FILL)
SPEC = importlib.util.spec_from_file_location("correct_review", ROOT / "scripts/correct_missing_context_manual_interpretation.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reviewed_csv() -> bytes:
    """실제 review_id를 역순으로 둔 기존 수동 판정 형식을 만든다."""
    stream = io.StringIO(newline="")
    columns = ["review_id", "split", "detector_status", *FILL.MANUAL_FIELDS]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for index in reversed(range(27)):
        row = {key: "" for key in columns}
        row.update(review_id=f"{index:03d}", split="train", detector_status="YUNET_FACE_NOT_FOUND")
        writer.writerow(row)
    return FILL.fill_review(stream.getvalue().encode(), "2026-09-20")[0]


def rows_by_id(data: bytes) -> dict[int, dict]:
    """CSV를 행 위치와 무관하게 ID로 색인한다."""
    return {int(row["review_id"]): row for row in csv.DictReader(io.StringIO(data.decode()))}


def test_correction_keeps_categories_and_automatic_metadata() -> None:
    original = reviewed_csv()
    updated, meta = MODULE.corrected_csv(original)
    before, after = rows_by_id(original), rows_by_id(updated)
    assert len(after) == 27
    assert sum(row["manual_category"] == "EXPECTED_DETECTOR_MISS" for row in after.values()) == 12
    assert sum(row["manual_category"] == "LIKELY_FALSE_NEGATIVE" for row in after.values()) == 15
    assert all(after[index]["face_near_edge"] == "TRUE" and after[index]["review_decision"] == "PASS" for index in range(12, 24))
    assert all(after[index]["extreme_pose"] == "" and after[index]["review_decision"] == "CHECK_NEEDED" for index in (*range(12), *range(24, 27)))
    assert all(after[index]["split"] == before[index]["split"] and after[index]["detector_status"] == before[index]["detector_status"] for index in range(27))
    assert meta["automatic_columns_preserved"] is True
    assert meta["before_csv_sha256"] == hashlib.sha256(original).hexdigest()
    assert meta["after_csv_sha256"] == hashlib.sha256(updated).hexdigest()


def test_conflicting_category_and_invalid_total_rejected() -> None:
    original = reviewed_csv()
    bad = original.replace(b"EXPECTED_DETECTOR_MISS", b"UNCERTAIN", 1)
    with pytest.raises(ValueError, match="충돌"):
        MODULE.corrected_csv(bad)
    with pytest.raises(ValueError, match="27"):
        MODULE.corrected_csv(b"\n".join(original.splitlines()[:-1]) + b"\n")


def test_provenance_summary_revision_report_and_no_repeat(tmp_path: Path) -> None:
    original = reviewed_csv()
    (tmp_path / "missing_context_manual_review.csv").write_bytes(original)
    (tmp_path / "missing_context_manual_review_summary.json").write_text(json.dumps({"reviewed_cases": 27}), encoding="utf-8")
    (tmp_path / "missing_context_manual_review_report.txt").write_text("[Manual Review Result]\nEXPECTED_DETECTOR_MISS: 12 / 27\n", encoding="utf-8")
    provenance = MODULE.apply_correction(tmp_path, "2026-09-20")
    summary = json.loads((tmp_path / "missing_context_manual_review_summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "missing_context_manual_review_report.txt").read_text(encoding="utf-8")
    correction = (tmp_path / "manual_review_interpretation_correction.md").read_text(encoding="utf-8")
    assert summary["interpretation_revision"] == 2
    assert summary["category_counts_changed"] is False
    assert summary["policy_changed"] is False
    assert summary["pose_related_pattern_observed"] is True
    assert "[Interpretation Correction — Revision 2]" in report
    assert provenance["before_csv_sha256"] in correction and provenance["after_csv_sha256"] in correction
    with pytest.raises(FileExistsError):
        MODULE.apply_correction(tmp_path, "2026-09-20")
