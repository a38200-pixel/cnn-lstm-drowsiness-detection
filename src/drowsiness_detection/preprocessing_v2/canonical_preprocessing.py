"""STEP 2 고정 helper를 재사용하는 STEP 3 canonical 전처리 진입 로직이다."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from .angle_utils import pitch_centered_candidate
from .canonical_sampling import canonical_timestamps, context_indices, evenly_spaced_positions, timestamp_to_nearest_frame_index
from .canonical_schema import empty_frame_row
from .canonical_writer import build_global_manifests, publish_bundle, resume_status, write_csv
from .context_crop_audit import crop_geometry, crop_preview, imagenet_padding_bgr, square_crop_roi
from .detectors import YuNetFaceDetector
from .geometric_features import calculate_ear, calculate_mar
from .head_pose import estimate_head_pose
from .landmarks import Dlib68Predictor
from .preprocessing_provenance import file_sha256, policy_payload, source_fingerprint, stable_hash


def _path(value: str, root: Path) -> Path:
    return (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()


def load_config(path: Path, project_root: Path) -> dict[str, Any]:
    """STEP 2 확정값과 train/val 보호 조건을 검증한다."""

    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("설정이 mapping이 아닙니다")
    sampling, context, detector, landmark = (values[key] for key in ("sampling", "context", "detector", "landmark"))
    if sampling != {"hz": 10.0, "clip_duration_sec": 10.0, "canonical_slots": 100}:
        raise ValueError("10 Hz × 10초 × 100 slot 정책과 다릅니다")
    expected_context = {"frame_count": 32, "crop_policy": "square_m10", "margin_ratio": 0.10,
                        "output_size": 224, "jpeg_quality": 95, "padding_mode": "imagenet_mean",
                        "rgb_mean": [0.485, 0.456, 0.406]}
    if context != expected_context:
        raise ValueError("STEP 2 SQUARE_M10 정책과 다릅니다")
    if detector.get("face_selection_policy") != "largest_valid_bbox" or detector.get("fallback") != "none":
        raise ValueError("YuNet 단독 최대 유효 bbox 정책이 필요합니다")
    if {key: detector.get(key) for key in ("score_threshold", "nms_threshold", "top_k")} != {"score_threshold": 0.9, "nms_threshold": 0.3, "top_k": 5000}:
        raise ValueError("STEP 2 YuNet 설정과 다릅니다")
    if landmark.get("roi_policy") != "raw_yunet_bbox":
        raise ValueError("Dlib68 ROI는 YuNet RAW bbox여야 합니다")
    if set(values["metadata"]) != {"train", "val"}:
        raise ValueError("metadata split은 train/val만 허용합니다")
    values["_root"] = project_root.resolve()
    values["_config_path"] = path.resolve()
    return values


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_input_rows(config: Mapping[str, Any], manifest: Path | None = None) -> list[dict[str, str]]:
    """통합 metadata라도 split을 먼저 거른 뒤 경로를 해석한다; test는 거부한다."""

    root = config["_root"]
    rows: list[dict[str, str]] = []
    if manifest is None:
        for split in ("train", "val"):
            for raw in _csv_rows(_path(config["metadata"][split], root)):
                if raw.get("split", split) != split:
                    raise ValueError("metadata split 불일치 또는 test row 발견")
                rows.append(raw)
    else:
        for raw in _csv_rows(manifest):
            if raw.get("split") not in {"train", "val"}:
                raise ValueError("manifest에는 train/val만 허용됩니다")
            rows.append(raw)
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not value or Path(value).name != value for value in ids):
        raise ValueError("video_id 중복 또는 잘못된 video_id")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("test row를 처리할 수 없습니다")
    return rows


def source_path(row: Mapping[str, str], config: Mapping[str, Any]) -> Path:
    """split 확인 후 raw root 안의 path 문자열만 허용한다."""

    if row["split"] not in {"train", "val"}:
        raise ValueError("test raw path는 해석하지 않습니다")
    path = _path(row.get("source_path") or row["video_path"], config["_root"])
    raw_root = _path(config["raw_root"], config["_root"])
    if not path.is_relative_to(raw_root):
        raise ValueError("영상 경로가 raw root 밖입니다")
    return path


def pilot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """split×label별 정렬 목록 전역에 걸쳐 5개씩 고정 선택한다."""

    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] not in {"train", "val"}:
            raise ValueError("pilot에 test를 포함할 수 없습니다")
        strata[(row["split"], row["label"])].append(row)
    expected = {(split, label) for split in ("train", "val") for label in ("drowsy", "not_drowsy")}
    if set(strata) != expected:
        raise ValueError(f"pilot stratum 불일치: {set(strata)}")
    selected: list[dict[str, str]] = []
    for split, label in sorted(expected):
        ordered = sorted(strata[(split, label)], key=lambda row: row["video_id"])
        for position in evenly_spaced_positions(len(ordered)):
            selected.append({**ordered[position], "selection_stratum": f"{split}:{label}",
                             "selection_method": "sorted_evenly_spaced_half_up_v1"})
    for order, row in enumerate(selected):
        row["pilot_order"] = str(order)
    return selected


def _hashes(config: Mapping[str, Any], mode: str, manifest: Path | None) -> tuple[str, str, str, str]:
    root = config["_root"]
    yunet_sha = file_sha256(_path(config["detector"]["model_path"], root))
    dlib_sha = file_sha256(_path(config["landmark"]["predictor_path"], root))
    policy_hash = stable_hash(policy_payload(config, yunet_sha, dlib_sha))
    run_config_hash = stable_hash({"policy_hash": policy_hash, "mode": mode,
                                   "outputs": config["outputs"], "manifest": str(manifest) if manifest else None,
                                   "validation": config["validation"]})
    return policy_hash, run_config_hash, yunet_sha, dlib_sha


def dry_run(config: Mapping[str, Any], mode: str, manifest: Path | None = None) -> dict[str, Any]:
    """CSV·설정·모델 바이트만 읽고 영상이나 detector는 열지 않는다."""

    rows = load_input_rows(config, manifest)
    for row in rows:
        source_path(row, config)
    policy_hash, run_hash, _, _ = _hashes(config, mode, manifest)
    counts = Counter(row["split"] for row in rows)
    return {"mode": "DRY_RUN", "planned_mode": mode, "video_count": len(rows),
            "split_counts": dict(counts), "test_split_used": False, "canonical_slots_per_video": 100,
            "planned_frame_rows": len(rows) * 100, "context_slots_per_video": 32,
            "planned_context_slots": len(rows) * 32, "context_indices": context_indices(),
            "video_opened": False, "detector_loaded": False, "policy_hash": policy_hash,
            "run_config_hash": run_hash, "decision": "WAITING_FOR_PILOT_RUN"}


def _process_frame(frame: np.ndarray, row: dict[str, Any], detector: Any, predictor: Any,
                   context: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """검출 1회 결과를 두 분기에 공유하고 단계별 결측을 독립적으로 유지한다."""

    started = perf_counter()
    row.update(decode_ok=True, decode_status="OK", frame_width=frame.shape[1], frame_height=frame.shape[0])
    detection = detector.detect(frame)
    row.update(detector_ms=detection.elapsed_ms, yunet_detection_count=detection.detection_count)
    points: np.ndarray | None = None
    crop: np.ndarray | None = None
    if not detection.success or detection.selected is None:
        row["detector_status"] = detection.failure_reason or "YUNET_FACE_NOT_FOUND"
        row["failure_flags"] = row["detector_status"]
        row["context_status"] = "CONTEXT_NO_FACE" if row["context_selected"] else "NOT_SELECTED"
        row["frame_total_ms"] = (perf_counter() - started) * 1000
        return points, crop
    bbox = detection.selected.bbox
    row.update(yunet_success=True, yunet_confidence=detection.selected.confidence,
               detector_status="OK", bbox_x1=bbox.x1, bbox_y1=bbox.y1, bbox_x2=bbox.x2,
               bbox_y2=bbox.y2, bbox_width=bbox.width, bbox_height=bbox.height)
    landmark = predictor.predict(frame, bbox)
    row["landmark_ms"] = landmark.elapsed_ms
    if landmark.success and landmark.points is not None:
        points = landmark.points.astype(np.float32)
        row.update(landmark_success=True, landmark_status="OK")
        feature_started = perf_counter()
        ear = calculate_ear(points)[2]
        mar = calculate_mar(points)
        if ear.valid:
            row["ear"] = ear.value
        else:
            row["failure_flags"] = "EAR_INVALID"
        if mar.valid:
            row["mar"] = mar.value
        else:
            row["failure_flags"] += ";MAR_INVALID" if row["failure_flags"] else "MAR_INVALID"
        pose = estimate_head_pose(points, frame.shape[1], frame.shape[0])
        if pose.success:
            row.update(pose_success=True, pose_status="OK", pitch_raw=pose.pitch,
                       pitch_centered_candidate=pitch_centered_candidate(pose.pitch),
                       yaw=pose.yaw, roll=pose.roll)
        else:
            row["pose_status"] = "POSE_FAILED"
            row["failure_flags"] += ";POSE_FAILED" if row["failure_flags"] else "POSE_FAILED"
        row["feature_ms"] = (perf_counter() - feature_started) * 1000
    else:
        row["landmark_status"] = "DLIB_LANDMARK_FAILED"
        row["failure_flags"] = "DLIB_LANDMARK_FAILED"
    if row["context_selected"]:
        crop_started = perf_counter()
        try:
            roi = square_crop_roi(bbox, context["margin_ratio"])
            geometry = crop_geometry(bbox, roi, frame.shape[1], frame.shape[0])
            crop = crop_preview(frame, geometry, context["output_size"], imagenet_padding_bgr(context["rgb_mean"]))
            row.update(context_crop_available=True, context_status="OK",
                       context_padding_fraction=geometry.padding_fraction,
                       context_face_area_fraction=geometry.face_area_fraction,
                       context_crop_relpath=f"context_crops/ctx_{row['context_slot']:02d}_k{row['canonical_index']:03d}.jpg")
        except Exception:
            row["context_status"] = "CONTEXT_CROP_FAILED"
            row["failure_flags"] += ";CONTEXT_CROP_FAILED" if row["failure_flags"] else "CONTEXT_CROP_FAILED"
        row["context_crop_ms"] = (perf_counter() - crop_started) * 1000
    row["frame_total_ms"] = (perf_counter() - started) * 1000
    return points, crop


def process_video(path: Path, identity: Mapping[str, str], detector: Any, predictor: Any,
                  context: Mapping[str, Any], debug_preview: bool = False) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, np.ndarray], dict[str, Any], dict[int, np.ndarray]]:
    """한 번 순차 decode하며 중복 source index의 전처리 결과를 재사용한다."""

    started = perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("FAILED_VIDEO_OPEN")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FAILED_INVALID_FPS")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            raise ValueError("FAILED_INVALID_FRAME_COUNT")
        indices = context_indices()
        slots = {canonical: context_slot for context_slot, canonical in enumerate(indices)}
        source_indices = [timestamp_to_nearest_frame_index(t, fps) for t in canonical_timestamps()]
        counts = Counter(source_indices)
        rows = [empty_frame_row(identity["video_id"], identity["split"], identity["label"], i, t,
                                source_indices[i], fps, frame_count, slots.get(i), counts[source_indices[i]] > 1)
                for i, t in enumerate(canonical_timestamps())]
        points = np.full((100, 68, 2), np.nan, dtype=np.float32)
        crops: dict[int, np.ndarray] = {}
        previews: dict[int, np.ndarray] = {}
        by_source: dict[int, list[int]] = defaultdict(list)
        for index, source_index in enumerate(source_indices):
            if source_index >= frame_count:
                rows[index]["decode_status"] = "SOURCE_INDEX_OUT_OF_RANGE"
                rows[index]["failure_flags"] = "SOURCE_INDEX_OUT_OF_RANGE"
            else:
                by_source[source_index].append(index)
        target_indices = set(by_source)
        max_index = max(target_indices, default=-1)
        for source_index in range(max_index + 1):
            decoded, frame = cap.read()
            if not decoded:
                break
            if source_index not in target_indices:
                continue
            first = by_source[source_index][0]
            landmark, crop = _process_frame(frame, rows[first], detector, predictor, context)
            if landmark is not None:
                points[first] = landmark
            if crop is not None:
                crops[first] = crop
                if debug_preview and rows[first]["context_slot"] in {0, 15, 31}:
                    annotated = frame.copy()
                    cv2.rectangle(annotated, (int(rows[first]["bbox_x1"]), int(rows[first]["bbox_y1"])),
                                  (int(rows[first]["bbox_x2"]), int(rows[first]["bbox_y2"])), (0, 255, 0), 2)
                    previews[first] = np.hstack((cv2.resize(annotated, (224, 224)), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)))
            for duplicate in by_source[source_index][1:]:
                preserved = {key: rows[duplicate][key] for key in ("canonical_index", "target_timestamp_sec", "actual_timestamp_sec", "timestamp_error_ms", "context_selected", "context_slot")}
                rows[duplicate].update(rows[first])
                rows[duplicate].update(preserved)
                if landmark is not None:
                    points[duplicate] = landmark
                if rows[duplicate]["context_selected"]:
                    if rows[first]["yunet_success"]:
                        bbox = rows[first]
                        from .detectors import BoundingBox
                        selected_bbox = BoundingBox(*(int(bbox[f"bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2")))
                        geometry = crop_geometry(selected_bbox, square_crop_roi(selected_bbox, context["margin_ratio"]), frame.shape[1], frame.shape[0])
                        crops[duplicate] = crop_preview(frame, geometry, context["output_size"], imagenet_padding_bgr(context["rgb_mean"]))
                        rows[duplicate].update(context_crop_available=True, context_status="OK",
                                               context_padding_fraction=geometry.padding_fraction,
                                               context_face_area_fraction=geometry.face_area_fraction,
                                               context_crop_relpath=f"context_crops/ctx_{rows[duplicate]['context_slot']:02d}_k{duplicate:03d}.jpg")
                        if debug_preview and rows[duplicate]["context_slot"] in {0, 15, 31}:
                            annotated = frame.copy()
                            cv2.rectangle(annotated, (selected_bbox.x1, selected_bbox.y1),
                                          (selected_bbox.x2, selected_bbox.y2), (0, 255, 0), 2)
                            previews[duplicate] = np.hstack((cv2.resize(annotated, (224, 224)),
                                                               cv2.cvtColor(crops[duplicate], cv2.COLOR_RGB2BGR)))
                    else:
                        rows[duplicate].update(context_crop_available=False, context_status="CONTEXT_NO_FACE")
                else:
                    rows[duplicate].update(context_crop_available=False, context_status="NOT_SELECTED", context_crop_relpath="",
                                           context_padding_fraction=math.nan, context_face_area_fraction=math.nan,
                                           context_crop_ms=math.nan)
        for row in rows:
            if row["decode_status"] == "NOT_RUN":
                row["decode_status"] = "FRAME_DECODE_FAILED"
                row["failure_flags"] = "FRAME_DECODE_FAILED"
        summary = {"video_id": identity["video_id"], "split": identity["split"], "label": identity["label"],
                   "source_path": str(path), "source_fps": fps, "reported_source_frame_count": frame_count,
                   "reported_duration_sec": frame_count / fps, "expected_canonical_slots": 100,
                   "decoded_slots": sum(row["decode_ok"] for row in rows),
                   "out_of_range_slots": sum(row["decode_status"] == "SOURCE_INDEX_OUT_OF_RANGE" for row in rows),
                   "decode_failed_slots": sum(row["decode_status"] == "FRAME_DECODE_FAILED" for row in rows),
                   "duplicate_source_index_count": sum(count - 1 for count in counts.values() if count > 1),
                   "yunet_success_count": sum(row["yunet_success"] for row in rows),
                   "yunet_failure_count": sum(row["decode_ok"] and not row["yunet_success"] for row in rows),
                   "landmark_success_count": sum(row["landmark_success"] for row in rows),
                   "landmark_failure_count": sum(row["yunet_success"] and not row["landmark_success"] for row in rows),
                   "ear_valid_count": sum(math.isfinite(row["ear"]) for row in rows),
                   "mar_valid_count": sum(math.isfinite(row["mar"]) for row in rows),
                   "pose_valid_count": sum(row["pose_success"] for row in rows),
                   "context_selected_count": sum(row["context_selected"] for row in rows),
                   "context_crop_available_count": sum(row["context_crop_available"] for row in rows),
                   "context_crop_missing_count": sum(row["context_selected"] and not row["context_crop_available"] for row in rows),
                   "multiple_face_frame_count": sum(row["yunet_detection_count"] > 1 for row in rows),
                   "processing_time_sec": perf_counter() - started, "bundle_status": "COMPLETE"}
        return rows, points, crops, summary, previews
    finally:
        cap.release()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def completed_resume_result(statuses: list[str], first_summary: Mapping[str, Any]) -> dict[str, Any] | None:
    """모든 bundle이 유효하면 원본 report를 유지하며 재추론 없는 resume 결과를 만든다."""

    if not statuses or any(status != "SKIP" for status in statuses):
        return None
    return {"completed_this_run": 0, "skipped": len(statuses), "failed": 0,
            "conflicts": 0, "video_count": first_summary["video_count"],
            "frame_count": first_summary["frame_count"], "test_split_processed": False,
            "policy_hash": first_summary["policy_hash"], "video_opened_for_inference": 0,
            "detector_frames": 0, "original_report_preserved": True}


def execute(config: Mapping[str, Any], mode: str, manifest: Path | None = None, resume: bool = False) -> dict[str, Any]:
    """명시적 실행에서만 영상·모델을 열고 pilot/full 출력을 격리한다."""

    if mode not in {"pilot", "full"}:
        raise ValueError("pilot 또는 full mode가 필요합니다")
    rows = load_input_rows(config, manifest)
    if mode == "pilot" and (manifest is None or len(rows) != 20):
        raise ValueError("pilot은 20개 영상 manifest가 필요합니다")
    if mode == "pilot" and Counter((row["split"], row["label"]) for row in rows) != {
        (split, label): 5 for split in ("train", "val") for label in ("drowsy", "not_drowsy")
    }:
        raise ValueError("pilot은 train/val × drowsy/not_drowsy 각 5개여야 합니다")
    if mode == "full" and manifest is not None:
        raise ValueError("full mode에서 pilot manifest를 사용하지 않습니다")
    if mode == "full" and Counter(row["split"] for row in rows) != {"train": 1452, "val": 311}:
        raise ValueError("frozen train/val 영상 수와 다릅니다")
    policy_hash, run_hash, yunet_sha, dlib_sha = _hashes(config, mode, manifest)
    root = config["_root"]
    output = _path(config["outputs"]["pilot_root" if mode == "pilot" else "full_root"], root)
    metadata = output / "metadata"
    existing_metadata = metadata / "run_metadata.json"
    if existing_metadata.is_file():
        previous = json.loads(existing_metadata.read_text(encoding="utf-8"))
        if previous.get("policy_hash") != policy_hash or previous.get("run_config_hash") != run_hash:
            raise ValueError("기존 run metadata와 정책 또는 실행 설정이 달라 자동 덮어쓰지 않습니다")
    if resume and existing_metadata.is_file():
        statuses = [resume_status(output / "per_video" / row["split"] / row["video_id"],
                                  policy_hash, source_fingerprint(source_path(row, config)), True)
                    for row in rows]
        first_summary_path = _path(config["outputs"]["report_root"], root) / mode / "preprocessing_summary.json"
        if first_summary_path.is_file():
            first_summary = json.loads(first_summary_path.read_text(encoding="utf-8"))
            no_op = completed_resume_result(statuses, first_summary)
            if no_op is not None:
                report_path = first_summary_path.parent / f"{mode}_resume_report.txt"
                if report_path.exists():
                    raise FileExistsError(f"기존 resume report 덮어쓰기 금지: {report_path}")
                report_path.write_text(json.dumps(no_op, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return no_op
    detector_config = config["detector"]
    detector = YuNetFaceDetector(str(_path(detector_config["model_path"], root)), detector_config["score_threshold"], detector_config["nms_threshold"], detector_config["top_k"])
    predictor = Dlib68Predictor(str(_path(config["landmark"]["predictor_path"], root)))
    metadata.mkdir(parents=True, exist_ok=True)
    write_csv(metadata / "input_manifest.csv", ("video_id", "split", "label", "source_path"),
              [{"video_id": row["video_id"], "split": row["split"], "label": row["label"],
                "source_path": str(source_path(row, config))} for row in rows])
    run_info = {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(root),
                "python_version": platform.python_version(), "numpy_version": np.__version__,
                "pandas_version": _version("pandas"), "opencv_version": cv2.__version__,
                "dlib_version": _version("dlib"), "os": platform.platform(),
                "config_path": str(config["_config_path"]), "policy_hash": policy_hash,
                "run_config_hash": run_hash, "yunet_model_path": str(_path(detector_config["model_path"], root)),
                "yunet_model_sha256": yunet_sha, "dlib_predictor_path": str(_path(config["landmark"]["predictor_path"], root)),
                "dlib_predictor_sha256": dlib_sha, "sampling_hz": 10.0, "clip_duration_sec": 10.0,
                "canonical_slots": 100, "context_indices": context_indices(),
                "frozen_policy": "YuNet / Dlib68 RAW bbox / SQUARE_M10 / ImageNet mean / RGB 224x224",
                "processed_split_allowlist": ["train", "val"], "test_split_processed": False}
    (metadata / "run_metadata.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    failures: list[dict[str, str]] = []
    skipped = 0
    completed = 0
    for identity in rows:
        path = source_path(identity, config)
        try:
            fingerprint = source_fingerprint(path)
            final = output / "per_video" / identity["split"] / identity["video_id"]
            status = resume_status(final, policy_hash, fingerprint, resume)
            if status == "SKIP":
                skipped += 1
                continue
            if status != "PROCESS":
                failures.append({"video_id": identity["video_id"], "status": status})
                continue
            frame_rows, landmarks, crops, summary, previews = process_video(
                path, identity, detector, predictor, config["context"], mode == "pilot" and completed < 8)
            summary.update(policy_hash=policy_hash, run_config_hash=run_hash)
            marker = {"video_id": identity["video_id"], "policy_hash": policy_hash, "run_config_hash": run_hash,
                      **fingerprint, "canonical_row_count": 100, "context_selected_count": 32,
                      "landmark_npz_shape": [100, 68, 2], "completed_at": datetime.now(timezone.utc).isoformat()}
            publish_bundle(output, identity["split"], identity["video_id"], frame_rows, landmarks, crops,
                           summary, marker, config["context"]["jpeg_quality"],
                           mode == "pilot" and config["validation"]["strict_pilot_crop_decode"], previews)
            completed += 1
        except Exception as exc:
            failures.append({"video_id": identity["video_id"], "status": f"FAILED:{type(exc).__name__}:{exc}"})
    totals = build_global_manifests(output, mode == "pilot" and config["validation"]["strict_pilot_crop_decode"])
    report_dir = _path(config["outputs"]["report_root"], root) / mode
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "failed_videos.csv", ("video_id", "status"), failures)
    result = {"completed_this_run": completed, "skipped": skipped, "failed": len(failures), **totals,
              "test_split_processed": False, "policy_hash": policy_hash}
    (report_dir / "preprocessing_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "preprocessing_report.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "integrity_report.txt").write_text(f"검증된 완료 bundle: {totals['video_count']}\n", encoding="utf-8")
    return result
