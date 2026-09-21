"""STEP 5-D3 Context CNN feature 최종 무결성 감사 테스트."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from drowsiness_detection.evaluation_v2 import context_cnn_feature_audit as audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (PROJECT_ROOT / "outputs/features_v2/context_cnn/final_integrity_audit"
               / "step5d3_final_integrity_audit.json")


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _backbone(report: dict, name: str) -> dict:
    return next(item for item in report["backbones"] if item["backbone"] == name)


def _index_rows(tmp_path: Path, count: int = 3) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        path = tmp_path / f"source_{index}.jpg"
        path.touch()
        rows.append({"source_feature_id": str(index),
                     "source_image_relpath": path.relative_to(tmp_path).as_posix()})
    return rows


def test_source_array_shape_mismatch() -> None:
    result = audit.validate_source_array(np.zeros((2, 512), dtype=np.float32))
    assert result["shape_mismatch"] == 1


def test_source_array_dtype_mismatch() -> None:
    result = audit.validate_source_array(np.zeros((audit.EXPECTED["unique_sources"], 512)))
    assert result["dtype_mismatch"] == 1


def test_source_array_nan_detection() -> None:
    array = np.ones((1, 512), dtype=np.float32)
    array[0, 0] = np.nan
    result = audit.validate_source_array(array)
    assert result["nan_count"] == 1


def test_source_array_inf_detection() -> None:
    array = np.ones((1, 512), dtype=np.float32)
    array[0, 0] = np.inf
    result = audit.validate_source_array(array)
    assert result["inf_count"] == 1


def test_source_array_all_zero_detection() -> None:
    result = audit.validate_source_array(np.zeros((1, 512), dtype=np.float32))
    assert result["all_zero_rows"] == 1


def test_source_index_duplicate_id_detection(tmp_path: Path) -> None:
    rows = _index_rows(tmp_path)
    rows[2]["source_feature_id"] = "1"
    assert audit.validate_source_index(rows, tmp_path)["duplicate_source_ids"] == 1


def test_source_index_duplicate_path_detection(tmp_path: Path) -> None:
    rows = _index_rows(tmp_path)
    rows[2]["source_image_relpath"] = rows[1]["source_image_relpath"]
    assert audit.validate_source_index(rows, tmp_path)["duplicate_source_paths"] == 1


def test_source_index_missing_path_detection(tmp_path: Path) -> None:
    rows = _index_rows(tmp_path)
    rows[0]["source_image_relpath"] = "missing.jpg"
    assert audit.validate_source_index(rows, tmp_path)["missing_source_paths"] == 1


def test_source_index_out_of_range_detection(tmp_path: Path) -> None:
    rows = _index_rows(tmp_path)
    rows[0]["source_feature_id"] = str(audit.EXPECTED["unique_sources"])
    assert audit.validate_source_index(rows, tmp_path)["invalid_source_ids"] == 1


def test_test_path_reference_detection(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    path = test_dir / "source.jpg"
    path.touch()
    rows = [{"source_feature_id": "0", "source_image_relpath": "test/source.jpg"}]
    assert audit.validate_source_index(rows, tmp_path)["test_source_references"] == 1


def test_required_artifact_missing_detection(tmp_path: Path) -> None:
    with pytest.raises(audit.IntegrityAuditError, match="missing artifacts"):
        audit._required_files(tmp_path, ("features.npy",))


def test_recorded_hash_match(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    path.write_bytes(b"feature")
    recorded = {path.name: audit.file_sha256(path)}
    assert audit.validate_recorded_hashes(tmp_path, recorded, (path.name,))["anomaly_count"] == 0


def test_recorded_hash_mismatch_detection(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    path.write_bytes(b"feature")
    result = audit.validate_recorded_hashes(tmp_path, {path.name: "0" * 64}, (path.name,))
    assert result["mismatches"] == [path.name]


def test_recorded_hash_missing_entry_detection(tmp_path: Path) -> None:
    (tmp_path / "features.npy").write_bytes(b"feature")
    result = audit.validate_recorded_hashes(tmp_path, {}, ("features.npy",))
    assert result["missing_records"] == ["features.npy"]


def test_report_status_pass(report: dict) -> None:
    assert report["status"] == "STEP_5D3_FINAL_INTEGRITY_AUDIT_PASS"


def test_report_anomaly_count_zero(report: dict) -> None:
    assert report["anomaly_count"] == 0


def test_report_universe_counts(report: dict) -> None:
    assert report["universe"]["videos"] == 1677
    assert report["universe"]["target_rows"] == 53664


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_global_source_shape_and_dtype(report: dict, name: str) -> None:
    source = _backbone(report, name)["source_feature_integrity"]
    assert source["shape"] == [52965, 512]
    assert source["dtype"] == "float32"


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_source_index_uniqueness(report: dict, name: str) -> None:
    index = _backbone(report, name)["source_index_integrity"]
    assert index["unique_source_ids"] == 52965
    assert index["unique_source_paths"] == 52965


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_bundle_shapes_dtypes_and_numeric_integrity(report: dict, name: str) -> None:
    bundles = _backbone(report, name)["video_bundle_integrity"]
    keys = ("feature_shape_mismatch", "feature_dtype_mismatch", "nan_count", "inf_count",
            "all_zero_rows")
    assert all(bundles[key] == 0 for key in keys)


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_per_video_artifacts_and_hashes(report: dict, name: str) -> None:
    bundles = _backbone(report, name)["video_bundle_integrity"]
    assert bundles["bundles"] == 1677
    assert bundles["missing_bundle_files"] == bundles["hash_mismatch"] == 0


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_target_source_mapping_exact(report: dict, name: str) -> None:
    bundles = _backbone(report, name)["video_bundle_integrity"]
    assert bundles["target_source_exact_mismatch"] == 0
    assert bundles["source_path_mapping_mismatch"] == 0


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_imputed_integrity(report: dict, name: str) -> None:
    bundles = _backbone(report, name)["video_bundle_integrity"]
    assert bundles["imputed_count"] == 699
    assert bundles["imputed_feature_mismatch"] == 0


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_mask_integrity(report: dict, name: str) -> None:
    bundles = _backbone(report, name)["video_bundle_integrity"]
    keys = ("mask_shape_mismatch", "mask_dtype_mismatch", "mask_complement_mismatch",
            "mask_mapping_mismatch")
    assert all(bundles[key] == 0 for key in keys)


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_policy_hash_integrity(report: dict, name: str) -> None:
    policy = _backbone(report, name)["policy_integrity"]
    assert policy["expected_cnn_feature_policy_hash"] == audit.POLICY_HASHES[name]
    assert policy["policy_hash_mismatch"] == 0


@pytest.mark.parametrize("name", audit.BACKBONES)
def test_checkpoint_provenance(report: dict, name: str) -> None:
    checkpoint = _backbone(report, name)["checkpoint_provenance"]
    assert checkpoint["recorded_mismatch"] == checkpoint["cache_mismatch"] == 0


def test_cross_backbone_mapping_equality(report: dict) -> None:
    cross = report["cross_backbone_integrity"]
    assert cross["video_id_mismatch"] == cross["semantic_mapping_mismatch"] == 0


def test_known_n311_exception_is_verified(report: dict) -> None:
    regression = report["d1_d2_regression"]
    assert regression["known_exception_verified"] is True
    assert regression["unexpected_mismatched_videos"] == []
    assert all(item["exact_equal_videos"] == 15 for item in regression["details"])


def test_unknown_regression_mismatch_is_rejected(report: dict) -> None:
    d2r_path = (PROJECT_ROOT / "outputs/features_v2/context_cnn/full_regression_audit"
                / "step5d2r_regression_audit.json")
    summary_path = (PROJECT_ROOT / "outputs/features_v2/context_cnn/full"
                    / "step5d2_full_feature_summary.json")
    d2r = json.loads(d2r_path.read_text(encoding="utf-8"))
    full = json.loads(summary_path.read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(d2r)
    corrupted["backbones"][0]["mismatched_video_ids"].append("d_unknown")
    result = audit.validate_known_exception(corrupted, full)
    assert result["known_exception_verified"] is False
    assert result["unexpected_mismatched_videos"] == ["d_unknown"]


def test_test_protection_is_sealed(report: dict) -> None:
    assert report["test_protection"] == {"test_access": 0, "test_rows": 0, "status": "SEALED"}


def test_artifact_immutability(report: dict) -> None:
    immutable = report["artifact_immutability"]
    assert immutable["mutation_count"] == 0
    assert immutable["complete_fingerprints_unchanged"] is True


def test_report_files_exist_and_text_records_exception() -> None:
    text_path = REPORT_PATH.with_name("step5d3_final_integrity_audit_report.txt")
    text = text_path.read_text(encoding="utf-8")
    assert REPORT_PATH.is_file()
    assert "n_311 remains a known numerical exception" in text
    assert "not rewritten as 16/16 PASS" in text


def test_report_writer_creates_json_and_text(tmp_path: Path, report: dict) -> None:
    json_path, text_path = audit.write_audit_report(tmp_path / "audit", report)
    assert json.loads(json_path.read_text(encoding="utf-8"))["anomaly_count"] == 0
    assert "STEP 5 FULLY CLOSED" in text_path.read_text(encoding="utf-8")


def test_report_writer_refuses_existing_output(tmp_path: Path, report: dict) -> None:
    output = tmp_path / "audit"
    audit.write_audit_report(output, report)
    with pytest.raises(FileExistsError, match="final audit"):
        audit.write_audit_report(output, report)
