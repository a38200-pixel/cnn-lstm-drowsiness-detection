"""STEP 3-B 결과를 읽기 전용 검증하고 구현 검토 자료를 별도 생성한다."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.canonical_integrity import validate_bundle, validate_global_rows


def _rows(path: Path) -> list[dict[str, str]]:
    """기존 CSV를 UTF-8 BOM 유무와 무관하게 읽는다."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _yes(value: str) -> bool:
    return value.casefold() == "true"


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None, "p95": None}
    return {"count": len(finite), "mean": float(np.mean(finite)), "median": float(np.median(finite)),
            "min": float(np.min(finite)), "max": float(np.max(finite)),
            "p90": float(np.percentile(finite, 90)), "p95": float(np.percentile(finite, 95))}


def _priority(row: Mapping[str, str]) -> tuple[int, str]:
    """정책 선호가 아니라 구현상 특이 frame의 검토 우선순위다."""

    score = 0
    if _number(row["context_padding_fraction"]) > 0:
        score += 100
    if int(row["yunet_detection_count"]) > 1:
        score += 40
    if not _yes(row["context_crop_available"]):
        score += 30
    if abs(_number(row["yaw"])) >= 30:
        score += 15
    if _number(row["yunet_confidence"]) < 0.92:
        score += 10
    if _yes(row["yunet_success"]):
        edges = (_number(row["bbox_x1"]), _number(row["bbox_y1"]),
                 _number(row["frame_width"]) - _number(row["bbox_x2"]),
                 _number(row["frame_height"]) - _number(row["bbox_y2"]))
        if min(edges) < 12:
            score += 8
    return score, row["video_id"]


def _review_selection(frames: list[dict[str, str]], videos: list[dict[str, str]]) -> list[dict[str, Any]]:
    """네 split×label 층에서 3개 영상씩, 각 영상 3개 Context slot을 고른다."""

    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frames:
        if _yes(row["context_selected"]):
            by_video[row["video_id"]].append(row)
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for video in videos:
        strata[(video["split"], video["label"])].append(video)
    selection: list[dict[str, Any]] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda video: video["video_id"])
        if len(members) != 5:
            raise ValueError(f"pilot 층별 5개 invariant 불일치: {stratum}")
        ranked = sorted(members, key=lambda video: (-max(_priority(row)[0] for row in by_video[video["video_id"]]), video["video_id"]))
        chosen = ranked[:2]
        normal = min((video for video in members if video not in chosen),
                     key=lambda video: (max(_priority(row)[0] for row in by_video[video["video_id"]]), video["video_id"]))
        chosen.append(normal)
        for video in chosen:
            available = sorted(by_video[video["video_id"]], key=lambda row: int(row["context_slot"]))
            middle = max(available[1:-1], key=lambda row: (_priority(row)[0], -abs(int(row["context_slot"]) - 15)))
            for row in (available[0], middle, available[-1]):
                selection.append({"video_id": video["video_id"], "split": video["split"], "label": video["label"],
                                  "context_slot": int(row["context_slot"]), "canonical_index": int(row["canonical_index"]),
                                  "source_frame_index": int(row["source_frame_index"]),
                                  "review_priority_score": _priority(row)[0], "crop_available": _yes(row["context_crop_available"]),
                                  "row": row, "source_path": video["source_path"]})
    return selection


def _letterbox(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """원본 비율을 보존해 왼쪽 패널에 표시한다."""

    result = np.full((size, size, 3), 40, dtype=np.uint8)
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    target = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    y, x = (size - target.shape[0]) // 2, (size - target.shape[1]) // 2
    result[y:y + target.shape[0], x:x + target.shape[1]] = target
    return result


def _make_visual_pack(selection: list[dict[str, Any]], pilot_root: Path, visual_dir: Path) -> dict[str, Any]:
    """저장 CSV와 JPEG를 이용해 12개 영상의 원본 bbox↔crop 검토 시트를 만든다."""

    if visual_dir.exists():
        raise FileExistsError(f"기존 visual review pack 덮어쓰기 금지: {visual_dir}")
    visual_dir.mkdir(parents=True)
    samples_dir = visual_dir / "samples"
    samples_dir.mkdir()
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selection:
        by_video[item["video_id"]].append(item)
    panels: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    decode_failures = 0
    for video_id, items in by_video.items():
        path = Path(items[0]["source_path"])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"review 원본 영상 열기 실패: {video_id}")
        targets = {item["source_frame_index"]: item for item in items}
        decoded: dict[int, np.ndarray] = {}
        try:
            for frame_index in range(max(targets) + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index in targets:
                    decoded[frame_index] = frame.copy()
        finally:
            cap.release()
        for item in items:
            row = item["row"]
            index = item["source_frame_index"]
            original = decoded.get(index)
            if original is None:
                decode_failures += 1
                original = np.full((224, 224, 3), 40, dtype=np.uint8)
                cv2.putText(original, "SOURCE DECODE FAILED", (5, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255))
            elif _yes(row["yunet_success"]):
                cv2.rectangle(original, (int(row["bbox_x1"]), int(row["bbox_y1"])),
                              (int(row["bbox_x2"]), int(row["bbox_y2"])), (0, 255, 0), 2)
            original_panel = _letterbox(original)
            crop_panel = np.full((224, 224, 3), 40, dtype=np.uint8)
            if item["crop_available"]:
                crop_path = pilot_root / "per_video" / item["split"] / video_id / row["context_crop_relpath"]
                crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                if crop is None or crop.shape != (224, 224, 3):
                    raise ValueError(f"저장 crop 손상: {crop_path}")
                crop_panel = crop
            else:
                cv2.putText(crop_panel, "CROP MISSING", (38, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255))
            header = np.zeros((35, 448, 3), dtype=np.uint8)
            title = f"{item['split']} {video_id} ctx{item['context_slot']:02d} k{item['canonical_index']:03d} score={item['review_priority_score']}"
            cv2.putText(header, title, (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.vstack((header, np.hstack((original_panel, crop_panel))))
            name = f"{item['split']}_{video_id}_ctx{item['context_slot']:02d}_k{item['canonical_index']:03d}.jpg"
            if not cv2.imwrite(str(samples_dir / name), panel, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"review sample 저장 실패: {name}")
            panels.append(panel)
            manifest.append({key: item[key] for key in ("video_id", "split", "label", "context_slot", "canonical_index",
                                                       "source_frame_index", "review_priority_score", "crop_available")} | {"image_path": f"samples/{name}"})
    sheet_count = 0
    for start in range(0, len(panels), 6):
        sheet = np.vstack(panels[start:start + 6])
        path = visual_dir / f"pilot_context_review_sheet_{start // 6 + 1:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"review sheet 저장 실패: {path}")
        sheet_count += 1
    with (visual_dir / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (visual_dir / "PILOT_VISUAL_REVIEW_GUIDE.md").write_text(
        "# STEP 3-B Pilot 구현 시각 검토\n\n"
        "왼쪽은 저장된 canonical source index를 순차 decode한 원본 frame이며 녹색은 저장된 YuNet bbox입니다. 오른쪽은 저장된 SQUARE_M10 JPEG입니다. "
        "crop이 없으면 `CROP MISSING`으로 표시합니다. 우선순위 점수는 padding·검출 누락·큰 yaw·낮은 confidence·multiple face·frame edge를 찾는 용도이며 정책 선호 점수가 아닙니다.\n\n"
        "확인: (1) 동일 인물·동일 frame, (2) bbox 대응, (3) RGB/BGR 피부색, (4) SQUARE_M10 geometry, "
        "(5) ImageNet mean padding, (6) JPEG 손상 여부. 이 pack으로 M10/M20 등 정책을 다시 선택하지 않습니다. "
        "사람이 직접 확인하기 전 상태는 `WAITING_FOR_MANUAL_PILOT_VISUAL_REVIEW`입니다.\n",
        encoding="utf-8")
    return {"selected_videos": len(by_video), "sample_count": len(manifest), "sheet_count": sheet_count,
            "source_decode_failures": decode_failures, "review_status": "WAITING_FOR_MANUAL_PILOT_VISUAL_REVIEW"}


def audit(pilot_root: Path, report_dir: Path, full_root: Path) -> dict[str, Any]:
    """원본 pilot bundle을 수정하지 않고 통계·엄격한 무결성·review pack을 생성한다."""

    expected_outputs = ["pilot_preprocessing_report.txt", "pilot_preprocessing_summary.json",
                        "pilot_integrity_report.txt", "pilot_failed_videos.csv", "pilot_processing.log"]
    if any((report_dir / name).exists() for name in expected_outputs) or (report_dir / "visual_review").exists():
        raise FileExistsError("기존 STEP 3-B audit report 또는 review pack 덮어쓰기 금지")
    frames = _rows(pilot_root / "metadata" / "canonical_frames.csv")
    videos = _rows(pilot_root / "metadata" / "canonical_videos.csv")
    manifest = _rows(PROJECT_ROOT / "data/metadata/experiment2_step3_pilot_manifest.csv")
    run_meta = json.loads((pilot_root / "metadata/run_metadata.json").read_text(encoding="utf-8"))
    first_run = json.loads((report_dir / "preprocessing_summary.json").read_text(encoding="utf-8"))
    resume = json.loads((report_dir / "pilot_resume_report.txt").read_text(encoding="utf-8"))
    integrity_errors = validate_global_rows(frames, videos)
    marker_hashes: set[str] = set()
    npz_mismatches = 0
    jpeg_invalid = 0
    jpeg_count = 0
    fingerprint_missing = 0
    markers = []
    for video in videos:
        bundle = pilot_root / "per_video" / video["split"] / video["video_id"]
        integrity_errors.extend(f"{video['video_id']}:{error}" for error in validate_bundle(bundle, True))
        marker = json.loads((bundle / "COMPLETE.json").read_text(encoding="utf-8"))
        markers.append(marker)
        marker_hashes.add(marker["policy_hash"])
        if not all(marker.get(key) is not None for key in ("source_path", "source_file_size", "source_mtime_ns")):
            fingerprint_missing += 1
        rows = _rows(bundle / "frames.csv")
        with np.load(bundle / "landmarks.npz") as data:
            points, valid, indices = data["landmarks"], data["landmark_valid"], data["canonical_index"]
            if points.shape != (100, 68, 2) or points.dtype != np.float32 or valid.shape != (100,) or indices.tolist() != list(range(100)):
                npz_mismatches += 1
            for index, row in enumerate(rows):
                if bool(valid[index]) != _yes(row["landmark_success"]):
                    npz_mismatches += 1
                if not valid[index] and not np.isnan(points[index]).all() or valid[index] and not np.isfinite(points[index]).all():
                    npz_mismatches += 1
        expected_crops = [row for row in rows if _yes(row["context_crop_available"])]
        actual_crops = list((bundle / "context_crops").glob("ctx_*.jpg"))
        jpeg_count += len(actual_crops)
        if len(expected_crops) != len(actual_crops):
            integrity_errors.append(f"{video['video_id']}:JPEG_COUNT_MISMATCH")
        for row in expected_crops:
            filename = f"ctx_{int(row['context_slot']):02d}_k{int(row['canonical_index']):03d}.jpg"
            path = bundle / row["context_crop_relpath"]
            if path.name != filename:
                integrity_errors.append(f"{video['video_id']}:JPEG_FILENAME_MISMATCH")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape != (224, 224, 3) or image.dtype != np.uint8:
                jpeg_invalid += 1
    if npz_mismatches:
        integrity_errors.append(f"NPZ_CROSS_ARTIFACT_MISMATCH:{npz_mismatches}")
    if jpeg_invalid:
        integrity_errors.append(f"JPEG_INVALID:{jpeg_invalid}")
    if fingerprint_missing:
        integrity_errors.append(f"FINGERPRINT_MISSING:{fingerprint_missing}")
    if marker_hashes != {run_meta["policy_hash"]}:
        integrity_errors.append("POLICY_HASH_MISMATCH")
    if len(manifest) != 20 or len({row["video_id"] for row in manifest}) != 20 or any(row["split"] == "test" for row in manifest):
        integrity_errors.append("MANIFEST_SCOPE")
    if len(frames) != 2000 or len(videos) != 20:
        integrity_errors.append("PILOT_EXPECTED_SIZE")
    if sum(_yes(row["context_selected"]) for row in frames) != 640:
        integrity_errors.append("CONTEXT_SELECTED_COUNT")
    if first_run.get("failed") != 0 or resume.get("skipped") != 20 or resume.get("completed_this_run") != 0:
        integrity_errors.append("RUN_OR_RESUME_STATUS")
    tmp = pilot_root / "_tmp"
    orphan_count = len(list(tmp.iterdir())) if tmp.is_dir() else 0
    if orphan_count:
        integrity_errors.append(f"ORPHAN_TMP:{orphan_count}")
    if full_root.exists():
        integrity_errors.append("FULL_ROOT_TOUCHED")
    started = datetime.fromisoformat(run_meta["created_at"])
    completed = max(datetime.fromisoformat(marker["completed_at"]) for marker in markers)
    wall_sec = (completed - started).total_seconds()
    video_times = [_number(row["processing_time_sec"]) for row in videos]
    context_rows = [row for row in frames if _yes(row["context_crop_available"])]
    decoded = [row for row in frames if _yes(row["decode_ok"])]
    detector_attempted = [row for row in decoded if math.isfinite(_number(row["detector_ms"]))]
    summary: dict[str, Any] = {
        "status": "AUTOMATIC_PILOT_RUN_COMPLETE_WAITING_FOR_MANUAL_PILOT_VISUAL_REVIEW" if not integrity_errors else "PILOT_INTEGRITY_FAILED",
        "manifest": {"path": "data/metadata/experiment2_step3_pilot_manifest.csv", "video_count": len(manifest),
                     "strata": dict(Counter(f"{row['split']}:{row['label']}" for row in manifest)),
                     "unique_video_ids": len({row["video_id"] for row in manifest}),
                     "source_paths_existing": sum(Path(row["source_path"]).is_file() for row in manifest), "test_rows": sum(row["split"] == "test" for row in manifest)},
        "bundle": {"complete": len(videos), "failed": first_run["failed"], "canonical_rows": len(frames),
                   "rows_per_video": sorted(Counter(row["video_id"] for row in frames).values()),
                   "orphan_temp_entries": orphan_count, "policy_hashes": sorted(marker_hashes)},
        "decode": {"decoded_slots": len(decoded), "out_of_range_slots": sum(row["decode_status"] == "SOURCE_INDEX_OUT_OF_RANGE" for row in frames),
                   "decode_failed_slots": sum(row["decode_status"] == "FRAME_DECODE_FAILED" for row in frames),
                   "duplicate_source_index_count": sum(int(row["duplicate_source_index_count"]) for row in videos),
                   "source_fps": _distribution([_number(row["source_fps"]) for row in videos]),
                   "reported_frame_count": _distribution([_number(row["reported_source_frame_count"]) for row in videos]),
                   "reported_duration_sec": _distribution([_number(row["reported_duration_sec"]) for row in videos]),
                   "timestamp_error_ms": _distribution([_number(row["timestamp_error_ms"]) for row in frames]),
                   "last_slot_out_of_range": sum(row["canonical_index"] == "99" and row["decode_status"] == "SOURCE_INDEX_OUT_OF_RANGE" for row in frames)},
        "detector": {"attempted_frames": len(detector_attempted), "success": sum(_yes(row["yunet_success"]) for row in frames),
                     "failure": sum(row["decode_status"] == "OK" and not _yes(row["yunet_success"]) for row in frames),
                     "multiple_face_frames": sum(int(row["yunet_detection_count"]) > 1 for row in frames),
                     "latency_ms": _distribution([_number(row["detector_ms"]) for row in frames])},
        "behavior": {"landmark_success": sum(_yes(row["landmark_success"]) for row in frames),
                     "landmark_failure_after_face": sum(_yes(row["yunet_success"]) and not _yes(row["landmark_success"]) for row in frames),
                     "ear_finite": sum(math.isfinite(_number(row["ear"])) for row in frames),
                     "mar_finite": sum(math.isfinite(_number(row["mar"])) for row in frames),
                     "pose_success": sum(_yes(row["pose_success"]) for row in frames),
                     "pitch_raw": _distribution([_number(row["pitch_raw"]) for row in frames]),
                     "pitch_centered_candidate": _distribution([_number(row["pitch_centered_candidate"]) for row in frames]),
                     "yaw": _distribution([_number(row["yaw"]) for row in frames]),
                     "roll": _distribution([_number(row["roll"]) for row in frames])},
        "context": {"selected": sum(_yes(row["context_selected"]) for row in frames),
                    "available": len(context_rows), "missing": sum(_yes(row["context_selected"]) and not _yes(row["context_crop_available"]) for row in frames),
                    "padding_required_count": sum(_number(row["context_padding_fraction"]) > 0 for row in context_rows),
                    "padding_fraction": _distribution([_number(row["context_padding_fraction"]) for row in context_rows]),
                    "face_area_fraction": _distribution([_number(row["context_face_area_fraction"]) for row in context_rows]),
                    "jpeg_file_count": jpeg_count, "jpeg_invalid_count": jpeg_invalid},
        "timing": {"pilot_wall_sec_from_metadata_to_last_bundle": wall_sec,
                   "processing_sec_per_video": _distribution(video_times),
                   "detector_ms": _distribution([_number(row["detector_ms"]) for row in frames]),
                   "landmark_ms": _distribution([_number(row["landmark_ms"]) for row in frames]),
                   "feature_ms": _distribution([_number(row["feature_ms"]) for row in frames]),
                   "context_crop_ms": _distribution([_number(row["context_crop_ms"]) for row in frames]),
                   "context_write_ms": _distribution([_number(row["context_write_ms"]) for row in frames]),
                   "full_1763_video_linear_projection_sec_from_mean": float(np.mean(video_times) * 1763),
                   "full_1763_video_linear_projection_sec_from_median": float(median(video_times) * 1763)},
        "integrity": {"error_count": len(integrity_errors), "errors": integrity_errors,
                      "npz_cross_artifact_mismatch_count": npz_mismatches, "fingerprint_missing_count": fingerprint_missing,
                      "test_frame_rows": sum(row["split"] == "test" for row in frames), "test_video_rows": sum(row["split"] == "test" for row in videos),
                      "full_root_exists": full_root.exists()},
        "resume": resume, "policy_hash": run_meta["policy_hash"],
        "manual_visual_review": "WAITING_FOR_MANUAL_PILOT_VISUAL_REVIEW",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    selection = _review_selection(frames, videos)
    summary["visual_review"] = _make_visual_pack(selection, pilot_root, report_dir / "visual_review")
    (report_dir / "pilot_preprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["STEP 3-B 20-Video Canonical Preprocessing Pilot", "", f"Status: {summary['status']}",
              "Frozen policy unchanged: YuNet / Dlib68 RAW bbox / SQUARE_M10 RGB 224x224 / ImageNet mean.",
              f"Manifest: {len(manifest)} videos; {summary['manifest']['strata']}; test=0.",
              f"Bundles: {len(videos)} complete; failures={first_run['failed']}; rows={len(frames)}; orphan temp={orphan_count}.",
              f"Decode: {len(decoded)} valid; out-of-range={summary['decode']['out_of_range_slots']}; failed={summary['decode']['decode_failed_slots']}.",
              f"YuNet: {summary['detector']['success']} success / {summary['detector']['failure']} miss; Dlib68: {summary['behavior']['landmark_success']} success.",
              f"EAR/MAR/Pose: {summary['behavior']['ear_finite']} / {summary['behavior']['mar_finite']} / {summary['behavior']['pose_success']}.",
              f"Context: 640 selected; {len(context_rows)} available; {summary['context']['missing']} missing; JPEG invalid={jpeg_invalid}.",
              f"NPZ cross-artifact mismatch={npz_mismatches}; policy hashes={len(marker_hashes)}; test rows=0; full root untouched={not full_root.exists()}.",
              f"Resume: new={resume['completed_this_run']}, skipped={resume['skipped']}, conflicts={resume['conflicts']}, detector frames={resume['detector_frames']}.",
              f"Pilot wall time={wall_sec:.2f} sec; mean/video={np.mean(video_times):.2f} sec; full linear projection={np.mean(video_times) * 1763 / 3600:.2f} h (not a guarantee).",
              f"Visual pack: {summary['visual_review']['selected_videos']} videos, {summary['visual_review']['sample_count']} samples, {summary['visual_review']['sheet_count']} sheets.",
              "Manual visual review is pending; STEP 3-C full run not executed."]
    (report_dir / "pilot_preprocessing_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (report_dir / "pilot_integrity_report.txt").write_text(
        "PASS\n" if not integrity_errors else "FAIL\n" + "\n".join(integrity_errors) + "\n", encoding="utf-8")
    with (report_dir / "pilot_failed_videos.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video_id", "status"))
        writer.writeheader()
        writer.writerows(_rows(report_dir / "failed_videos.csv"))
    (report_dir / "pilot_processing.log").write_text(
        "Pilot first run: 20 completed, 0 failed. Resume: 0 new, 20 skipped, 0 detector frames. "
        "This log summarizes existing run artifacts; it is not a live frame-level log.\n", encoding="utf-8")
    return summary


def main() -> int:
    """pilot 루트만 분석하고 full root는 읽기 전용 존재 확인만 한다."""

    parser = argparse.ArgumentParser(description="STEP 3-B pilot result audit and visual implementation review pack")
    parser.add_argument("--pilot-root", type=Path, default=PROJECT_ROOT / "data/interim/preprocessing_v2/canonical_pilot")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/pilot")
    args = parser.parse_args()
    result = audit(args.pilot_root, args.report_dir, PROJECT_ROOT / "data/interim/preprocessing_v2/canonical")
    print(json.dumps({"status": result["status"], "bundle_count": result["bundle"]["complete"],
                      "context_available": result["context"]["available"], "integrity_errors": result["integrity"]["error_count"],
                      "visual_review": result["visual_review"]}, ensure_ascii=False, indent=2))
    return 0 if not result["integrity"]["error_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
