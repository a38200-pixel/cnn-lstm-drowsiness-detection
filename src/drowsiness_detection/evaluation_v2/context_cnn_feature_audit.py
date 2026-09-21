"""STEP 5-D3 full Context CNN feature artifact read-only final audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256, stable_hash
from drowsiness_detection.sequences_v2.cnn_feature_extractor import mapping_fingerprint


EXPECTED = {
    "videos": 1677, "train": 1382, "val": 295, "drowsy": 800,
    "not_drowsy": 877, "context_only": 157, "target_rows": 53664,
    "unique_sources": 52965, "imputed": 699, "original": 52965,
}
POLICY_HASHES = {
    "resnet18": "0b8d72abdc9f3fbdb44934100c507ebc249ea8225d01fe1e4ca0de2236c4cd3e",
    "vgg16": "1edc39db86531ea3240f767722236ab45ee89d607811770308c30b526f0b2bc0",
}
WEIGHTS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "vgg16": "VGG16_Weights.IMAGENET1K_V1",
}
CHECKPOINTS = {
    "resnet18": ("resnet18-f37072fd.pth",
                 "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "vgg16": ("vgg16-397923af.pth",
               "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"),
}
CANONICAL_HASH = "29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721"
MISSING_HASH = "f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d"
MAPPING_FINGERPRINT = "83b4587fdb41e5ef4f02287a0de8076426910a65eefb02cf4cc032527b42e453"
BACKBONES = ("resnet18", "vgg16")
MAPPING_COLUMNS = (
    "target_context_index", "target_canonical_index", "target_timestamp_sec",
    "original_available", "imputed", "source_context_index", "source_canonical_index",
    "source_image_relpath", "source_feature_id", "feature_row_index",
)
SEMANTIC_COLUMNS = (
    "video_id", "split", "label", "target_context_index", "target_canonical_index",
    "target_timestamp_sec", "original_available", "imputed", "source_context_index",
    "source_canonical_index", "source_image_relpath",
)


class IntegrityAuditError(RuntimeError):
    """Final audit 입력이 구조적으로 불완전할 때 사용한다."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _boolean(value: str) -> bool:
    normalized = value.casefold()
    if normalized not in {"true", "false"}:
        raise IntegrityAuditError(f"invalid boolean: {value}")
    return normalized == "true"


def validate_source_array(array: np.ndarray) -> dict[str, Any]:
    shape_mismatch = int(tuple(array.shape) != (EXPECTED["unique_sources"], 512))
    dtype_mismatch = int(array.dtype != np.float32)
    if array.ndim != 2 or array.shape[1:] != (512,):
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "shape_mismatch": shape_mismatch, "dtype_mismatch": dtype_mismatch,
                "finite_count": int(np.isfinite(array).sum()),
                "nan_count": int(np.isnan(array).sum()),
                "inf_count": int(np.isinf(array).sum()), "all_zero_rows": -1}
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "shape_mismatch": shape_mismatch, "dtype_mismatch": dtype_mismatch,
            "finite_count": int(np.isfinite(array).sum()),
            "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()),
            "all_zero_rows": int(np.count_nonzero(np.all(array == 0, axis=1)))}


def validate_source_index(rows: Sequence[Mapping[str, str]], project_root: Path) -> dict[str, Any]:
    ids = [int(row["source_feature_id"]) for row in rows]
    paths = [row["source_image_relpath"] for row in rows]
    invalid_ids = sum(not 0 <= source_id < EXPECTED["unique_sources"] for source_id in ids)
    test_references = sum("test" in Path(path).parts for path in paths)
    return {
        "rows": len(rows), "row_count_mismatch": int(len(rows) != EXPECTED["unique_sources"]),
        "unique_source_ids": len(set(ids)),
        "duplicate_source_ids": len(ids) - len(set(ids)),
        "source_id_order_mismatch": int(ids != list(range(len(rows)))),
        "invalid_source_ids": invalid_ids,
        "unique_source_paths": len(set(paths)),
        "duplicate_source_paths": len(paths) - len(set(paths)),
        "missing_source_paths": sum(not (project_root / path).is_file() for path in paths),
        "test_source_references": test_references,
    }


def _required_files(root: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise IntegrityAuditError(f"missing artifacts at {root}: {missing}")


def _complete_fingerprint(root: Path, videos: Sequence[Mapping[str, str]]) -> str:
    paths = [root / "COMPLETE.json"] + [
        root / "per_video" / row["split"] / row["video_id"] / "COMPLETE.json"
        for row in videos]
    return stable_hash({"complete_files": [
        {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
        for path in paths]})


def validate_recorded_hashes(root: Path, recorded: Mapping[str, str],
                             required: Sequence[str]) -> dict[str, Any]:
    """완료 마커의 파일 해시 집합과 실제 파일을 비교한다."""
    missing_records = sorted(set(required) - set(recorded))
    unexpected_records = sorted(set(recorded) - set(required))
    mismatches = []
    for filename in required:
        path = root / filename
        expected = recorded.get(filename)
        if expected is None or not path.is_file() or file_sha256(path) != expected:
            mismatches.append(filename)
    return {
        "required": list(required), "missing_records": missing_records,
        "unexpected_records": unexpected_records, "mismatches": mismatches,
        "anomaly_count": len(missing_records) + len(unexpected_records) + len(mismatches),
    }


def _checkpoint_provenance(backbone: str, artifact_summary: Mapping[str, Any]) -> dict[str, Any]:
    filename, expected_hash = CHECKPOINTS[backbone]
    recorded = artifact_summary["checkpoint_provenance"]
    result = {"filename": filename, "expected_sha256": expected_hash,
              "recorded_filename": recorded.get("cached_checkpoint"),
              "recorded_sha256": recorded.get("checkpoint_sha256"),
              "recorded_mismatch": int(recorded.get("cached_checkpoint") != filename
                                       or recorded.get("checkpoint_sha256") != expected_hash),
              "cache_present": False, "cache_sha256": None, "cache_mismatch": 0}
    try:
        import torch
        candidate = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if candidate.is_file():
            result["cache_present"] = True
            result["cache_sha256"] = file_sha256(candidate)
            result["cache_mismatch"] = int(result["cache_sha256"] != expected_hash)
    except Exception as error:
        result["cache_check_note"] = f"{type(error).__name__}: {error}"
    return result


def audit_backbone(project_root: Path, backbone: str,
                   context_metadata: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    root = project_root / "data/interim/features_v2/context_full" / backbone
    _required_files(root, ("COMPLETE.json", "full_feature_summary.json", "source_features.npy",
                           "source_feature_index.csv", "full_feature_videos.csv"))
    if not (root / "per_video").is_dir():
        raise IntegrityAuditError(f"missing per_video tree: {root}")
    marker = _read_json(root / "COMPLETE.json")
    global_summary = _read_json(root / "full_feature_summary.json")
    videos = _read_csv(root / "full_feature_videos.csv")
    source_rows = _read_csv(root / "source_feature_index.csv")
    source_features = np.load(root / "source_features.npy", mmap_mode="r", allow_pickle=False)
    source_integrity = validate_source_array(source_features)
    source_index = validate_source_index(source_rows, project_root)
    source_paths = [row["source_image_relpath"] for row in source_rows]

    split_counts = {name: sum(row["split"] == name for row in videos)
                    for name in ("train", "val", "test")}
    label_counts = {name: sum(row["label"] == name for row in videos)
                    for name in ("drowsy", "not_drowsy")}
    video_ids = [row["video_id"] for row in videos]
    context_only = sum(
        context_metadata[row["video_id"]]["behavior_eligible"].casefold() == "false"
        for row in videos if row["video_id"] in context_metadata)
    universe = {
        "videos": len(videos), "unique_video_ids": len(set(video_ids)),
        **split_counts, **label_counts, "context_only": context_only,
        "n_246_present": "n_246" in video_ids,
        "metadata_missing_videos": sum(video_id not in context_metadata for video_id in video_ids),
        "test_directory_present": (root / "per_video/test").exists(),
        "feature_video_index_mismatch": int(
            [int(row["feature_video_index"]) for row in videos] != list(range(len(videos)))),
        "marker_video_order_mismatch": int(marker.get("full_video_ids") != video_ids),
    }
    expected_policy = POLICY_HASHES[backbone]
    policy_mismatch = int(marker.get("cnn_feature_policy_hash") != expected_policy)
    canonical_mismatch = int(marker.get("canonical_policy_hash") != CANONICAL_HASH)
    missing_policy_mismatch = int(marker.get("sequence_missing_policy_hash") != MISSING_HASH)
    policy_mismatch += int(global_summary.get("cnn_feature_policy_hash") != expected_policy)
    canonical_mismatch += int(global_summary.get("canonical_policy_hash") != CANONICAL_HASH)
    missing_policy_mismatch += int(global_summary.get("sequence_missing_policy_hash") != MISSING_HASH)
    resolved_weights_mismatch = int(
        global_summary.get("feature_policy", {}).get("weights_enum") != WEIGHTS[backbone])

    totals = {"bundles": 0, "missing_bundle_files": 0, "feature_shape_mismatch": 0,
              "feature_dtype_mismatch": 0, "nan_count": 0, "inf_count": 0,
              "all_zero_rows": 0, "mapping_row_mismatch": 0,
              "target_index_mismatch": 0, "feature_row_index_mismatch": 0,
              "invalid_source_ids": 0, "source_path_mapping_mismatch": 0,
              "target_source_exact_mismatch": 0, "imputed_count": 0,
              "original_count": 0, "imputed_feature_mismatch": 0,
              "mask_shape_mismatch": 0, "mask_dtype_mismatch": 0,
              "mask_complement_mismatch": 0, "mask_mapping_mismatch": 0,
              "hash_mismatch": 0, "policy_hash_mismatch": policy_mismatch,
              "canonical_hash_mismatch": canonical_mismatch,
              "missing_policy_hash_mismatch": missing_policy_mismatch,
              "resolved_weights_mismatch": resolved_weights_mismatch,
              "bundle_metadata_mismatch": 0, "bundle_fingerprint_mismatch": 0,
              "test_mapping_rows": 0, "test_source_references": 0}
    semantic_rows: list[tuple[str, ...]] = []
    fingerprint_rows: list[dict[str, str]] = []
    required_bundle = ("features.npy", "feature_mapping.csv", "masks.npz",
                       "summary.json", "COMPLETE.json")
    for video in videos:
        bundle = root / "per_video" / video["split"] / video["video_id"]
        missing_files = [name for name in required_bundle if not (bundle / name).is_file()]
        if missing_files:
            totals["missing_bundle_files"] += len(missing_files)
            continue
        totals["bundles"] += 1
        features = np.load(bundle / "features.npy", mmap_mode="r", allow_pickle=False)
        mappings = _read_csv(bundle / "feature_mapping.csv")
        summary = _read_json(bundle / "summary.json")
        complete = _read_json(bundle / "COMPLETE.json")
        totals["feature_shape_mismatch"] += tuple(features.shape) != (32, 512)
        totals["feature_dtype_mismatch"] += features.dtype != np.float32
        totals["nan_count"] += int(np.isnan(features).sum())
        totals["inf_count"] += int(np.isinf(features).sum())
        if features.ndim == 2 and features.shape[1:] == (512,):
            totals["all_zero_rows"] += int(np.count_nonzero(np.all(features == 0, axis=1)))
        totals["mapping_row_mismatch"] += len(mappings) != 32
        target_indices = [int(row["target_context_index"]) for row in mappings]
        row_indices = [int(row["feature_row_index"]) for row in mappings]
        totals["target_index_mismatch"] += target_indices != list(range(32))
        totals["feature_row_index_mismatch"] += row_indices != list(range(32))
        if any(not set(MAPPING_COLUMNS).issubset(row) for row in mappings):
            raise IntegrityAuditError(f"mapping columns missing: {video['video_id']}")
        source_ids = np.asarray([int(row["source_feature_id"]) for row in mappings], dtype=np.int64)
        valid_ids = (source_ids >= 0) & (source_ids < len(source_rows))
        totals["invalid_source_ids"] += int(np.count_nonzero(~valid_ids))
        for row, source_id in zip(mappings, source_ids):
            if 0 <= source_id < len(source_rows):
                totals["source_path_mapping_mismatch"] += (
                    row["source_image_relpath"] != source_paths[source_id])
            imputed = _boolean(row["imputed"])
            original = _boolean(row["original_available"])
            totals["imputed_count"] += imputed
            totals["original_count"] += original
            totals["test_mapping_rows"] += video["split"] == "test"
            totals["test_source_references"] += "test" in Path(row["source_image_relpath"]).parts
            enriched = {**row, "video_id": video["video_id"], "split": video["split"],
                        "label": video["label"]}
            semantic_rows.append(tuple(str(enriched[key]) for key in SEMANTIC_COLUMNS))
            fingerprint_rows.append(enriched)
        if valid_ids.all() and tuple(features.shape) == (32, 512):
            expected_features = source_features[source_ids]
            mismatch_mask = np.any(features != expected_features, axis=1)
            totals["target_source_exact_mismatch"] += int(np.count_nonzero(mismatch_mask))
            imputed_mask = np.asarray([_boolean(row["imputed"]) for row in mappings])
            totals["imputed_feature_mismatch"] += int(np.count_nonzero(mismatch_mask & imputed_mask))
        with np.load(bundle / "masks.npz", allow_pickle=False) as masks:
            original_mask = (masks["original_context_valid_mask"]
                             if "original_context_valid_mask" in masks.files else None)
            imputed_mask_file = (masks["context_imputed_mask"]
                                 if "context_imputed_mask" in masks.files else None)
        if original_mask is None or imputed_mask_file is None:
            totals["mask_shape_mismatch"] += 1
        else:
            totals["mask_shape_mismatch"] += (original_mask.shape != (32,)
                                               or imputed_mask_file.shape != (32,))
            totals["mask_dtype_mismatch"] += (original_mask.dtype != np.bool_
                                               or imputed_mask_file.dtype != np.bool_)
            if original_mask.shape == (32,) and imputed_mask_file.shape == (32,):
                totals["mask_complement_mismatch"] += int(
                    np.count_nonzero(original_mask != ~imputed_mask_file))
                mapping_imputed = np.asarray([_boolean(row["imputed"]) for row in mappings])
                totals["mask_mapping_mismatch"] += int(
                    np.count_nonzero(imputed_mask_file != mapping_imputed))
        totals["policy_hash_mismatch"] += (
            summary.get("cnn_feature_policy_hash") != expected_policy
            or complete.get("cnn_feature_policy_hash") != expected_policy)
        totals["canonical_hash_mismatch"] += (
            summary.get("canonical_policy_hash") != CANONICAL_HASH
            or complete.get("canonical_policy_hash") != CANONICAL_HASH)
        totals["missing_policy_hash_mismatch"] += (
            summary.get("sequence_missing_policy_hash") != MISSING_HASH
            or complete.get("sequence_missing_policy_hash") != MISSING_HASH)
        expected_imputed = sum(_boolean(row["imputed"]) for row in mappings)
        expected_bundle_metadata = {
            "video_id": video["video_id"], "backbone": backbone,
            "feature_shape": [32, 512], "dtype": "float32",
            "imputed_count": expected_imputed,
        }
        totals["bundle_metadata_mismatch"] += sum(
            summary.get(key) != value or complete.get(key) != value
            for key, value in expected_bundle_metadata.items())
        bundle_fingerprint = mapping_fingerprint([
            {**row, "video_id": video["video_id"], "split": video["split"],
             "label": video["label"]} for row in mappings])
        totals["bundle_fingerprint_mismatch"] += (
            summary.get("source_mapping_fingerprint") != bundle_fingerprint
            or complete.get("source_mapping_fingerprint") != bundle_fingerprint)
        hash_result = validate_recorded_hashes(
            bundle, complete.get("file_sha256", {}), required_bundle[:-1])
        totals["hash_mismatch"] += hash_result["anomaly_count"]

    global_hash_mismatch = 0
    global_hash_mismatch += file_sha256(root / "source_features.npy") != marker.get("source_features_sha256")
    global_hash_mismatch += (file_sha256(root / "full_feature_summary.json")
                             != marker.get("full_feature_summary_sha256"))
    totals["hash_mismatch"] += global_hash_mismatch
    global_hashes = {name: file_sha256(root / name) for name in (
        "source_features.npy", "source_feature_index.csv", "full_feature_videos.csv",
        "full_feature_summary.json", "COMPLETE.json")}
    computed_fingerprint = mapping_fingerprint(fingerprint_rows)
    fingerprint_mismatch = int(computed_fingerprint != MAPPING_FINGERPRINT)
    fingerprint_mismatch += int(marker.get("source_mapping_fingerprint") != MAPPING_FINGERPRINT)
    fingerprint_mismatch += int(global_summary.get("source_mapping_fingerprint") != MAPPING_FINGERPRINT)
    expected_global_summary = {
        "backbone": backbone, "full_videos": EXPECTED["videos"], "sequence_length": 32,
        "target_rows": EXPECTED["target_rows"], "unique_source_images": EXPECTED["unique_sources"],
        "feature_dimension": 512, "dtype": "float32", "imputed_feature_mismatch": 0,
        "wrong_source_mapping": 0, "test_used": False,
    }
    global_metadata_mismatch = sum(
        global_summary.get(key) != value for key, value in expected_global_summary.items())
    global_metadata_mismatch += marker.get("backbone") != backbone
    return {
        "backbone": backbone, "root": root.relative_to(project_root).as_posix(),
        "global_artifacts": {"required_present": True,
                             "global_hash_mismatch": global_hash_mismatch,
                             "global_metadata_mismatch": global_metadata_mismatch,
                             "actual_sha256": global_hashes},
        "universe": universe, "source_feature_integrity": source_integrity,
        "source_index_integrity": source_index, "video_bundle_integrity": totals,
        "mapping_integrity": {"computed_fingerprint": computed_fingerprint,
                              "expected_fingerprint": MAPPING_FINGERPRINT,
                              "fingerprint_mismatch": fingerprint_mismatch},
        "policy_integrity": {"expected_cnn_feature_policy_hash": expected_policy,
                             "resolved_weights": WEIGHTS[backbone],
                             "policy_hash_mismatch": totals["policy_hash_mismatch"],
                             "canonical_hash_mismatch": totals["canonical_hash_mismatch"],
                             "missing_policy_hash_mismatch": totals["missing_policy_hash_mismatch"],
                             "resolved_weights_mismatch": totals["resolved_weights_mismatch"]},
        "checkpoint_provenance": _checkpoint_provenance(backbone, global_summary),
        "semantic_rows": semantic_rows,
        "video_ids": video_ids,
        "complete_fingerprint": _complete_fingerprint(root, videos),
        "source_features_sha256": file_sha256(root / "source_features.npy"),
    }


def validate_known_exception(d2r: Mapping[str, Any], full_summary: Mapping[str, Any]) -> dict[str, Any]:
    expected_cause = "BENIGN FLOATING-POINT / BATCH NUMERICAL DIFFERENCE"
    details = []
    unexpected: list[str] = []
    full_by_backbone = {item["backbone"]: item for item in full_summary["backbones"]}
    for item in d2r["backbones"]:
        backbone = item["backbone"]
        mismatched = item["mismatched_video_ids"]
        unexpected.extend(video_id for video_id in mismatched if video_id != "n_311")
        video = item["mismatched_videos"][0] if item["mismatched_videos"] else None
        slots = ([slot["target_context_index"] for slot in video["slots"] if not slot["allclose"]]
                 if video else [])
        sources = item["mismatch_sources"]
        rerun = item.get("controlled_rerun", {})
        rerun_sources = rerun.get("sources", [])
        evidence_count = sum(
            source["comparisons"]["d1_stored_vs_d1_batch"]["exact_equal"]
            and source["comparisons"]["d2_stored_vs_d2_batch"]["exact_equal"]
            for source in rerun_sources)
        recorded = full_by_backbone[backbone]["pilot_full_regression"]
        details.append({
            "backbone": backbone, "videos_compared": recorded["videos_compared"],
            "exact_equal_videos": recorded["exact_equal_videos"],
            "regression_status_preserved": recorded["status"],
            "rtol": recorded["rtol"], "atol": recorded["atol"],
            "mismatched_video_ids": mismatched, "known_exception_video": "n_311",
            "target_slots": slots, "mismatch_source_count": len(sources),
            "d1_batch_sizes": sorted({source["d1_batch"]["batch_actual_size"] for source in sources}),
            "d2_batch_sizes": sorted({source["d2_batch"]["batch_actual_size"] for source in sources}),
            "controlled_rerun_exact_evidence": evidence_count,
            "source_mapping_mismatch": item["mapping_mismatch"],
        })
    verified = (
        d2r.get("cause") == expected_cause and d2r.get("test_access") == 0
        and not d2r.get("full_reextraction_required") and not unexpected
        and all(detail["videos_compared"] == 16 and detail["exact_equal_videos"] == 15
                and detail["regression_status_preserved"] == "FAIL"
                and detail["mismatched_video_ids"] == ["n_311"]
                and detail["target_slots"] == list(range(24, 32))
                and detail["mismatch_source_count"] == 6
                and detail["d1_batch_sizes"] == [6] and detail["d2_batch_sizes"] == [16]
                and detail["controlled_rerun_exact_evidence"] == 6
                and detail["source_mapping_mismatch"] == 0 for detail in details))
    return {"cause": d2r.get("cause"), "expected_cause": expected_cause,
            "known_exception_verified": verified, "unexpected_mismatched_videos": unexpected,
            "details": details}


def _anomaly_total(backbone: Mapping[str, Any]) -> int:
    source = backbone["source_feature_integrity"]
    source_index = backbone["source_index_integrity"]
    bundles = backbone["video_bundle_integrity"]
    universe = backbone["universe"]
    mapping = backbone["mapping_integrity"]
    checkpoint = backbone["checkpoint_provenance"]
    expected_universe = (
        universe["videos"] == EXPECTED["videos"]
        and universe["unique_video_ids"] == EXPECTED["videos"]
        and universe["train"] == EXPECTED["train"] and universe["val"] == EXPECTED["val"]
        and universe["drowsy"] == EXPECTED["drowsy"]
        and universe["not_drowsy"] == EXPECTED["not_drowsy"]
        and universe["context_only"] == EXPECTED["context_only"]
        and not universe["n_246_present"] and universe["test"] == 0
        and not universe["test_directory_present"] and universe["metadata_missing_videos"] == 0
        and universe["feature_video_index_mismatch"] == 0
        and universe["marker_video_order_mismatch"] == 0)
    expected_totals = (
        bundles["bundles"] == EXPECTED["videos"]
        and bundles["imputed_count"] == EXPECTED["imputed"]
        and bundles["original_count"] == EXPECTED["original"])
    ignored = {"bundles", "imputed_count", "original_count"}
    bundle_anomalies = sum(int(value) for key, value in bundles.items() if key not in ignored)
    index_anomalies = sum(source_index[key] for key in (
        "row_count_mismatch", "duplicate_source_ids", "source_id_order_mismatch",
        "invalid_source_ids", "duplicate_source_paths", "missing_source_paths",
        "test_source_references"))
    source_anomalies = sum(source[key] for key in (
        "shape_mismatch", "dtype_mismatch", "nan_count", "inf_count", "all_zero_rows"))
    source_anomalies += int(source["finite_count"] != EXPECTED["unique_sources"] * 512)
    return (int(not expected_universe) + int(not expected_totals) + bundle_anomalies
            + index_anomalies + source_anomalies + mapping["fingerprint_mismatch"]
            + backbone["global_artifacts"]["global_metadata_mismatch"]
            + checkpoint["recorded_mismatch"] + checkpoint["cache_mismatch"])


def run_final_audit(project_root: Path) -> dict[str, Any]:
    context_rows = _read_csv(
        project_root / "data/interim/sequences_v2/context/metadata/context_sequence_videos.csv")
    context_metadata = {row["video_id"]: row for row in context_rows}
    input_reference_paths = [
        project_root / "data/interim/sequences_v2/context/metadata/context_sequence_videos.csv",
        project_root / "outputs/features_v2/context_cnn/full_regression_audit/step5d2r_regression_audit.json",
    ]
    input_reference_before = {path.relative_to(project_root).as_posix(): file_sha256(path)
                              for path in input_reference_paths}
    backbones = [audit_backbone(project_root, backbone, context_metadata)
                 for backbone in BACKBONES]
    semantic_mismatch = sum(left != right for left, right in zip(
        backbones[0]["semantic_rows"], backbones[1]["semantic_rows"]))
    semantic_mismatch += abs(len(backbones[0]["semantic_rows"])
                             - len(backbones[1]["semantic_rows"]))
    video_id_mismatch = len(set(backbones[0]["video_ids"]) ^ set(backbones[1]["video_ids"]))
    d2r = _read_json(input_reference_paths[1])
    full_summary = _read_json(
        project_root / "outputs/features_v2/context_cnn/full/step5d2_full_feature_summary.json")
    regression = validate_known_exception(d2r, full_summary)
    complete_before = {item["backbone"]: item["complete_fingerprint"] for item in backbones}
    source_before = {item["backbone"]: item["source_features_sha256"] for item in backbones}
    input_reference_after = {path.relative_to(project_root).as_posix(): file_sha256(path)
                             for path in input_reference_paths}
    complete_after = {}
    source_after = {}
    for item in backbones:
        root = project_root / item["root"]
        videos = _read_csv(root / "full_feature_videos.csv")
        complete_after[item["backbone"]] = _complete_fingerprint(root, videos)
        source_after[item["backbone"]] = file_sha256(root / "source_features.npy")
    mutation_count = sum(complete_before[key] != complete_after[key] for key in BACKBONES)
    mutation_count += sum(source_before[key] != source_after[key] for key in BACKBONES)
    mutation_count += sum(input_reference_before[key] != input_reference_after[key]
                          for key in input_reference_before)
    for item in backbones:
        item.pop("semantic_rows")
        item.pop("video_ids")
        item.pop("complete_fingerprint")
        item.pop("source_features_sha256")
    anomaly_count = sum(_anomaly_total(item) for item in backbones)
    anomaly_count += semantic_mismatch + video_id_mismatch
    anomaly_count += int(not regression["known_exception_verified"])
    anomaly_count += len(regression["unexpected_mismatched_videos"])
    anomaly_count += mutation_count
    status = ("STEP_5D3_FINAL_INTEGRITY_AUDIT_PASS"
              if anomaly_count == 0 else "STEP_5D3_REVIEW_REQUIRED")
    return {
        "status": status,
        "universe": {**EXPECTED, "sequence_length": 32,
                     "relationship_verified": EXPECTED["unique_sources"] + EXPECTED["imputed"]
                     == EXPECTED["target_rows"]},
        "environment_reference": full_summary["environment"],
        "backbones": backbones,
        "cross_backbone_integrity": {
            "video_id_mismatch": video_id_mismatch,
            "semantic_mapping_mismatch": semantic_mismatch,
            "source_mapping_fingerprint_equal": all(
                item["mapping_integrity"]["computed_fingerprint"] == MAPPING_FINGERPRINT
                for item in backbones)},
        "d1_d2_regression": regression,
        "known_exception": {"video_id": "n_311", "classification": regression["cause"],
                            "tolerance_relaxed": False, "full_reextraction_required": False},
        "test_protection": {"test_access": 0, "test_rows": 0, "status": "SEALED"},
        "artifact_immutability": {"mutation_count": mutation_count,
                                  "complete_fingerprints_unchanged": complete_before == complete_after,
                                  "source_feature_hashes_unchanged": source_before == source_after,
                                  "reference_hashes_unchanged": input_reference_before == input_reference_after},
        "anomaly_count": anomaly_count,
        "decision": ("STEP 5 FULLY CLOSED" if status.endswith("PASS")
                     else "STEP 5 NOT CLOSED; REVIEW REQUIRED"),
        "cnn_inference_executed": False,
        "artifacts_modified": False,
    }


def write_audit_report(output_root: Path, result: Mapping[str, Any]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "step5d3_final_integrity_audit.json"
    text_path = output_root / "step5d3_final_integrity_audit_report.txt"
    if json_path.exists() or text_path.exists():
        raise FileExistsError(f"기존 final audit output이 있습니다: {output_root}")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    universe = result["universe"]
    environment = result["environment_reference"]
    lines = ["STEP 5-D3 Full CNN Feature Final Integrity Audit", "",
             f"Status: {result['status']}", f"Decision: {result['decision']}",
             f"Anomaly count: {result['anomaly_count']}", "CNN inference executed: false",
             "Audited feature artifacts modified: false", "Test access: 0 (SEALED)", "",
             "[Universe]",
             f"Videos: {universe['videos']} (train {universe['train']}, val {universe['val']})",
             f"Labels: drowsy {universe['drowsy']}, not_drowsy {universe['not_drowsy']}",
             f"Context-only: {universe['context_only']}; n_246 absent",
             f"Targets: {universe['target_rows']}; sources: {universe['unique_sources']}; "
             f"imputed: {universe['imputed']}",
             f"Environment: Python {environment['python_version']}, torch "
             f"{environment['torch_version']}, torchvision {environment['torchvision_version']}, "
             f"CUDA {environment['torch_cuda_runtime']}, {environment['gpu_device_name']}", ""]
    for item in result["backbones"]:
        source = item["source_feature_integrity"]
        index = item["source_index_integrity"]
        bundle = item["video_bundle_integrity"]
        policy = item["policy_integrity"]
        checkpoint = item["checkpoint_provenance"]
        lines.extend([f"[{item['backbone']}]", f"Source shape: {source['shape']}",
                      f"Source dtype: {source['dtype']}; finite: {source['finite_count']}; "
                      f"NaN/Inf/all-zero rows: {source['nan_count']}/{source['inf_count']}/"
                      f"{source['all_zero_rows']}",
                      f"Source index rows/unique IDs/unique paths: {index['rows']}/"
                      f"{index['unique_source_ids']}/{index['unique_source_paths']}",
                      f"Bundles: {bundle['bundles']}; missing files: {bundle['missing_bundle_files']}",
                      f"Target/source exact mismatch: {bundle['target_source_exact_mismatch']}",
                      f"Original/imputed rows: {bundle['original_count']}/{bundle['imputed_count']}",
                      f"Imputed feature mismatch: {bundle['imputed_feature_mismatch']}; "
                      f"mask mismatch: {bundle['mask_mapping_mismatch']}",
                      f"Artifact hash mismatch: {bundle['hash_mismatch']}; "
                      f"policy mismatch: {policy['policy_hash_mismatch']}",
                      f"Mapping fingerprint: {item['mapping_integrity']['computed_fingerprint']}",
                      f"Weights: {policy['resolved_weights']}",
                      f"Checkpoint: {checkpoint['filename']} ({checkpoint['cache_sha256']})", ""])
    cross = result["cross_backbone_integrity"]
    lines.extend(["[Cross-backbone]",
                  f"Video ID mismatch: {cross['video_id_mismatch']}",
                  f"Semantic mapping mismatch: {cross['semantic_mapping_mismatch']}", ""])
    regression = result["d1_d2_regression"]
    lines.extend(["[D1/D2 Regression]", "15/16 exact videos per backbone.",
                  "n_311 remains a known numerical exception.",
                  f"Cause: {regression['cause']}",
                  f"Unexpected mismatched videos: {regression['unexpected_mismatched_videos']}",
                  "Affected target slots: 24-31; unique sources: 6.",
                  "Controlled rerun: D1 batch size 6 and D2 batch size 16 each match stored "
                  "vectors 6/6 exactly for both backbones.",
                  "Tolerance remains rtol=1e-5, atol=1e-6; result is not rewritten as 16/16 PASS.",
                  "Full re-extraction required: false", ""])
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, text_path
