"""STEP 5-D1 pilot과 STEP 5-D2 full feature의 read-only 회귀 감사 도구."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256
from drowsiness_detection.sequences_v2.cnn_feature_extractor import (
    BACKBONES, _extract_unique_sources, build_torch_extractor,
)


RTOL = 1e-5
ATOL = 1e-6
TOLERANCES = ((1e-6, 1e-7), (1e-5, 1e-6), (1e-4, 1e-5))
MAPPING_KEYS = (
    "target_context_index", "target_canonical_index", "source_context_index",
    "source_canonical_index", "source_image_relpath", "imputed",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def difference_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Broadcasting을 허용하지 않고 동일 shape 배열의 오차 통계를 계산한다."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"REGRESSION_SHAPE_MISMATCH: {left.shape} != {right.shape}")
    difference = left.astype(np.float64) - right.astype(np.float64)
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(left.astype(np.float64)), np.finfo(np.float32).tiny)
    close_mask = np.isclose(left, right, rtol=RTOL, atol=ATOL)
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "dtype_equal": left.dtype == right.dtype,
        "exact_equal": bool(np.array_equal(left, right)),
        "allclose": bool(close_mask.all()),
        "max_abs_diff": float(absolute.max(initial=0.0)),
        "mean_abs_diff": float(absolute.mean()),
        "median_abs_diff": float(np.median(absolute)),
        "rmse": float(math.sqrt(np.mean(np.square(difference)))),
        "max_relative_diff": float((absolute / denominator).max(initial=0.0)),
        "nonzero_diff_element_count": int(np.count_nonzero(absolute)),
        "allclose_failed_element_count": int(np.count_nonzero(~close_mask)),
        "tolerance_results": [
            {"rtol": rtol, "atol": atol,
             "pass": bool(np.allclose(left, right, rtol=rtol, atol=atol))}
            for rtol, atol in TOLERANCES
        ],
    }


def batch_position(source_feature_id: int, source_count: int,
                   batch_size: int = 16) -> dict[str, Any]:
    if not 0 <= source_feature_id < source_count:
        raise ValueError("source_feature_id out of range")
    batch_index = source_feature_id // batch_size
    start = batch_index * batch_size
    actual_size = min(batch_size, source_count - start)
    return {
        "source_feature_id": source_feature_id,
        "batch_index": batch_index,
        "position_within_batch": source_feature_id - start,
        "batch_actual_size": actual_size,
        "partial_batch": actual_size < batch_size,
        "batch_start_source_feature_id": start,
        "batch_end_source_feature_id_exclusive": start + actual_size,
    }


def _source_index(root: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = _read_csv(root / "source_feature_index.csv")
    lookup = {row["source_image_relpath"]: int(row["source_feature_id"]) for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate source_image_relpath")
    return rows, lookup


def _video_paths(root: Path, split: str, video_id: str) -> tuple[Path, Path]:
    bundle = root / "per_video" / split / video_id
    return bundle / "features.npy", bundle / "feature_mapping.csv"


def compare_stored_backbone(project_root: Path, backbone: str) -> dict[str, Any]:
    pilot_root = project_root / "data/interim/features_v2/context_pilot" / backbone
    full_root = project_root / "data/interim/features_v2/context_full" / backbone
    manifest = _read_csv(
        project_root / "data/metadata/experiment2_step5d_cnn_feature_pilot_manifest.csv")
    pilot_source_rows, pilot_lookup = _source_index(pilot_root)
    full_source_rows, full_lookup = _source_index(full_root)
    pilot_sources = np.load(pilot_root / "source_features.npy", mmap_mode="r", allow_pickle=False)
    full_sources = np.load(full_root / "source_features.npy", mmap_mode="r", allow_pickle=False)
    if pilot_sources.shape != (454, 512) or full_sources.shape != (52965, 512):
        raise ValueError("unexpected source feature shape")

    videos: list[dict[str, Any]] = []
    mismatched_video_ids: list[str] = []
    source_path_mismatch = 0
    for item in manifest:
        pilot_feature_path, pilot_mapping_path = _video_paths(
            pilot_root, item["split"], item["video_id"])
        full_feature_path, full_mapping_path = _video_paths(
            full_root, item["split"], item["video_id"])
        pilot_features = np.load(pilot_feature_path, allow_pickle=False)
        full_features = np.load(full_feature_path, allow_pickle=False)
        video_stats = difference_stats(pilot_features, full_features)
        pilot_mapping = _read_csv(pilot_mapping_path)
        full_mapping = _read_csv(full_mapping_path)
        if len(pilot_mapping) != 32 or len(full_mapping) != 32:
            raise ValueError("mapping row count must be 32")
        slots = []
        mapping_mismatches = 0
        for index, (left, right) in enumerate(zip(pilot_mapping, full_mapping)):
            differing_keys = [key for key in MAPPING_KEYS if left[key] != right[key]]
            mapping_mismatches += bool(differing_keys)
            source_path_mismatch += left["source_image_relpath"] != right["source_image_relpath"]
            stats = difference_stats(pilot_features[index], full_features[index])
            slots.append({
                "target_context_index": int(left["target_context_index"]),
                "target_canonical_index": int(left["target_canonical_index"]),
                "source_context_index": int(left["source_context_index"]),
                "source_canonical_index": int(left["source_canonical_index"]),
                "source_image_relpath": left["source_image_relpath"],
                "imputed": left["imputed"].casefold() == "true",
                "d1_source_feature_id": int(left["source_feature_id"]),
                "d2_source_feature_id": int(right["source_feature_id"]),
                "mapping_equal": not differing_keys,
                "mapping_differing_keys": differing_keys,
                **{key: stats[key] for key in (
                    "max_abs_diff", "mean_abs_diff", "allclose", "exact_equal",
                    "nonzero_diff_element_count", "allclose_failed_element_count")},
            })
        if not video_stats["allclose"]:
            mismatched_video_ids.append(item["video_id"])
        videos.append({"video_id": item["video_id"], "split": item["split"],
                       "mapping_mismatch_slots": mapping_mismatches,
                       "difference": video_stats, "slots": slots})

    mismatch_sources: list[dict[str, Any]] = []
    exact_sources = 0
    allclose_mismatch_sources = 0
    for path, pilot_id in pilot_lookup.items():
        if path not in full_lookup:
            source_path_mismatch += 1
            mismatch_sources.append({"source_image_relpath": path,
                                     "classification": "MISSING_IN_FULL"})
            continue
        full_id = full_lookup[path]
        stats = difference_stats(pilot_sources[pilot_id], full_sources[full_id])
        exact_sources += stats["exact_equal"]
        allclose_mismatch_sources += not stats["allclose"]
        if not stats["allclose"]:
            jpeg = project_root / path
            mismatch_sources.append({
                "source_image_relpath": path,
                "difference": stats,
                "jpeg_file_size": jpeg.stat().st_size,
                "current_jpeg_sha256": file_sha256(jpeg),
                "historical_jpeg_hash_available": False,
                "jpeg_integrity_note": "D1 당시 JPEG와 동일하다고 증명할 수 없음",
                "d1_batch": batch_position(pilot_id, len(pilot_source_rows)),
                "d2_batch": batch_position(full_id, len(full_source_rows)),
            })
    mismatched_videos = [row for row in videos if not row["difference"]["allclose"]]
    return {
        "backbone": backbone,
        "pilot_unique_sources": len(pilot_source_rows),
        "full_unique_sources": len(full_source_rows),
        "mismatched_video_ids": mismatched_video_ids,
        "mismatched_videos": mismatched_videos,
        "source_path_mismatch": source_path_mismatch,
        "exact_equal_sources": exact_sources,
        "allclose_mismatch_sources": allclose_mismatch_sources,
        "mismatch_sources": mismatch_sources,
        "mapping_mismatch": sum(row["mapping_mismatch_slots"] for row in videos),
        "shape_dtype_equal": all(
            row["difference"]["shape"] == [32, 512]
            and row["difference"]["dtype_equal"] for row in videos),
    }


def _rerun_batch(torch, extractor, project_root: Path, paths: Sequence[str],
                 selected_path: str, device: str) -> np.ndarray:
    features = _extract_unique_sources(
        torch, extractor, paths, project_root, device=device, batch_size=len(paths))
    return features[list(paths).index(selected_path)]


def controlled_rerun(project_root: Path, stored: Mapping[str, Any],
                     device: str = "cuda") -> dict[str, Any]:
    """Mismatch source와 해당 D1/D2 batch만 다시 추론한다."""
    backbone = str(stored["backbone"])
    pilot_root = project_root / "data/interim/features_v2/context_pilot" / backbone
    full_root = project_root / "data/interim/features_v2/context_full" / backbone
    pilot_rows, pilot_lookup = _source_index(pilot_root)
    full_rows, full_lookup = _source_index(full_root)
    pilot_paths = [row["source_image_relpath"] for row in pilot_rows]
    full_paths = [row["source_image_relpath"] for row in full_rows]
    pilot_features = np.load(pilot_root / "source_features.npy", mmap_mode="r", allow_pickle=False)
    full_features = np.load(full_root / "source_features.npy", mmap_mode="r", allow_pickle=False)
    torch, extractor, weights_enum = build_torch_extractor(backbone, device)
    outputs = []
    for source in stored["mismatch_sources"]:
        path = source["source_image_relpath"]
        d1_id, d2_id = pilot_lookup[path], full_lookup[path]
        d1_position = batch_position(d1_id, len(pilot_paths))
        d2_position = batch_position(d2_id, len(full_paths))
        d1_batch = pilot_paths[d1_position["batch_start_source_feature_id"]:
                               d1_position["batch_end_source_feature_id_exclusive"]]
        d2_batch = full_paths[d2_position["batch_start_source_feature_id"]:
                              d2_position["batch_end_source_feature_id_exclusive"]]
        reruns = {
            "single": _rerun_batch(torch, extractor, project_root, [path], path, device),
            "d1_batch": _rerun_batch(torch, extractor, project_root, d1_batch, path, device),
            "d2_batch": _rerun_batch(torch, extractor, project_root, d2_batch, path, device),
        }
        vectors = {"d1_stored": np.asarray(pilot_features[d1_id]),
                   "d2_stored": np.asarray(full_features[d2_id]), **reruns}
        comparisons = {}
        pairs = (("d1_stored", "d2_stored"), ("d1_stored", "single"),
                 ("d2_stored", "single"), ("d1_stored", "d1_batch"),
                 ("d2_stored", "d2_batch"), ("d1_batch", "d2_batch"),
                 ("single", "d1_batch"), ("single", "d2_batch"))
        for left, right in pairs:
            comparisons[f"{left}_vs_{right}"] = difference_stats(vectors[left], vectors[right])
        outputs.append({"source_image_relpath": path, "weights_enum": weights_enum,
                        "d1_batch_paths": d1_batch, "d2_batch_paths": d2_batch,
                        "comparisons": comparisons})
    return {"device": device, "sources_rerun": len(outputs), "sources": outputs}


def determinism_settings() -> dict[str, Any]:
    import torch
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()),
        "extraction_sets_random_seed": False,
        "extraction_enables_deterministic_algorithms": False,
    }


def classify_cause(backbones: Sequence[Mapping[str, Any]]) -> str:
    if any(item["mapping_mismatch"] or item["source_path_mismatch"] for item in backbones):
        return "SOURCE MAPPING BUG"
    if any(not item["shape_dtype_equal"] for item in backbones):
        return "FEATURE EXTRACTION BUG"
    reruns = [source for item in backbones for source in item.get("controlled_rerun", {}).get("sources", [])]
    if not reruns:
        return "UNRESOLVED"
    batch_evidence = all(
        source["comparisons"]["d1_stored_vs_d1_batch"]["allclose"]
        and source["comparisons"]["d2_stored_vs_d2_batch"]["allclose"]
        and not source["comparisons"]["d1_batch_vs_d2_batch"]["exact_equal"]
        and source["comparisons"]["d1_batch_vs_d2_batch"]["max_abs_diff"] < 0.01
        for source in reruns)
    return ("BENIGN FLOATING-POINT / BATCH NUMERICAL DIFFERENCE"
            if batch_evidence else "UNRESOLVED")


def run_audit(project_root: Path, run_inference: bool = False,
              device: str = "cuda") -> dict[str, Any]:
    backbones = [compare_stored_backbone(project_root, backbone) for backbone in BACKBONES]
    if run_inference:
        for item in backbones:
            item["controlled_rerun"] = controlled_rerun(project_root, item, device=device)
    cause = classify_cause(backbones)
    mismatched_sets = [set(item["mismatched_video_ids"]) for item in backbones]
    return {
        "status": "STEP_5D2_REGRESSION_INVESTIGATION_COMPLETE",
        "extraction_status": "COMPLETE",
        "integrity_status": ("READY_FOR_FINAL_INTEGRITY_AUDIT"
                             if cause == "BENIGN FLOATING-POINT / BATCH NUMERICAL DIFFERENCE"
                             else "REVIEW_REQUIRED"),
        "cause": cause,
        "same_mismatched_videos": len(mismatched_sets) == 2
                                    and mismatched_sets[0] == mismatched_sets[1],
        "determinism": determinism_settings(),
        "backbones": backbones,
        "full_reextraction_required": cause in {
            "SOURCE MAPPING BUG", "FEATURE EXTRACTION BUG", "INPUT ARTIFACT CHANGE"},
        "test_access": 0,
        "artifacts_modified": False,
    }


def write_audit_report(output_root: Path, result: Mapping[str, Any]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "step5d2r_regression_audit.json"
    text_path = output_root / "step5d2r_regression_audit_report.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["STEP 5-D2R Pilot/Full Feature Regression Mismatch Investigation", "",
             f"Cause: {result['cause']}",
             f"Extraction status: {result['extraction_status']}",
             f"Integrity status: {result['integrity_status']}",
             f"Same mismatched videos: {result['same_mismatched_videos']}",
             f"cuDNN deterministic: {result['determinism']['cudnn_deterministic']}",
             f"cuDNN benchmark: {result['determinism']['cudnn_benchmark']}",
             f"Deterministic algorithms: {result['determinism']['deterministic_algorithms_enabled']}",
             "Test access: 0 (SEALED)", ""]
    for item in result["backbones"]:
        video = item["mismatched_videos"][0]
        difference = video["difference"]
        mismatch_slots = [str(slot["target_context_index"]) for slot in video["slots"]
                          if not slot["allclose"]]
        lines.extend([f"[{item['backbone']}]",
                      f"Mismatched videos: {', '.join(item['mismatched_video_ids'])}",
                      f"Mismatched target slots: {', '.join(mismatch_slots)}",
                      f"Mapping mismatch: {item['mapping_mismatch']}",
                      f"Shape/dtype equal: {item['shape_dtype_equal']}",
                      f"Max abs diff: {difference['max_abs_diff']}",
                      f"Mean abs diff: {difference['mean_abs_diff']}",
                      f"Median abs diff: {difference['median_abs_diff']}",
                      f"RMSE: {difference['rmse']}",
                      f"Failed elements: {difference['allclose_failed_element_count']}",
                      f"Source allclose mismatch: {item['allclose_mismatch_sources']}",
                      f"Exact equal sources: {item['exact_equal_sources']} / {item['pilot_unique_sources']}", ""])
        for source in item["mismatch_sources"]:
            lines.extend([f"Source: {source['source_image_relpath']}",
                          f"Current JPEG SHA-256: {source['current_jpeg_sha256']}",
                          f"D1 batch: {source['d1_batch']}",
                          f"D2 batch: {source['d2_batch']}",
                          source["jpeg_integrity_note"], ""])
        rerun = item.get("controlled_rerun", {})
        if rerun:
            d1_exact = sum(source["comparisons"]["d1_stored_vs_d1_batch"]["exact_equal"]
                           for source in rerun["sources"])
            d2_exact = sum(source["comparisons"]["d2_stored_vs_d2_batch"]["exact_equal"]
                           for source in rerun["sources"])
            lines.extend([f"Controlled rerun device: {rerun['device']}",
                          f"D1 stored == D1 batch rerun: {d1_exact}/{rerun['sources_rerun']}",
                          f"D2 stored == D2 batch rerun: {d2_exact}/{rerun['sources_rerun']}", ""])
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, text_path
