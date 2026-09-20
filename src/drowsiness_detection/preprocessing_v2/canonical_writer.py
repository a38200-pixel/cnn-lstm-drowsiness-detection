"""영상별 atomic bundle 저장·resume 및 완료 bundle 집계를 담당한다."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .canonical_integrity import validate_bundle, validate_global_rows
from .canonical_schema import FRAME_COLUMNS, VIDEO_COLUMNS


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """정해진 컬럼 순서로 UTF-8 CSV를 쓴다."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_rgb_jpeg(path: Path, rgb: np.ndarray, quality: int = 95) -> None:
    """RGB 메모리 배열을 BGR OpenCV JPEG로 명시적으로 변환한다."""

    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError("Context crop은 uint8 RGB 224×224여야 합니다")
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"JPEG 저장 실패: {path}")


def resume_status(final: Path, policy_hash: str, fingerprint: Mapping[str, Any], resume: bool) -> str:
    """완전하고 동일한 bundle만 skip하며 나머지 기존 경로는 충돌로 처리한다."""

    if not final.exists():
        return "PROCESS"
    marker_path = final / "COMPLETE.json"
    if not marker_path.is_file():
        return "CONFLICT_INCOMPLETE"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "CONFLICT_INCOMPLETE"
    if marker.get("policy_hash") != policy_hash or any(marker.get(key) != value for key, value in fingerprint.items()):
        return "CONFLICT_PROVENANCE"
    if validate_bundle(final):
        return "CONFLICT_INVALID_BUNDLE"
    return "SKIP" if resume else "CONFLICT_EXISTS"


def publish_bundle(root: Path, split: str, video_id: str, rows: list[dict[str, Any]],
                   landmarks: np.ndarray, crops: Mapping[int, np.ndarray], summary: Mapping[str, Any],
                   marker: Mapping[str, Any], quality: int = 95, strict_crop_decode: bool = False,
                   debug_previews: Mapping[int, np.ndarray] | None = None) -> Path:
    """임시 디렉터리에서 검증한 뒤 동일 파일시스템의 최종 경로로 이동한다."""

    if split not in {"train", "val"} or not video_id or Path(video_id).name != video_id or video_id in {".", ".."}:
        raise ValueError("허용되지 않은 split 또는 video_id입니다")
    final = root / "per_video" / split / video_id
    if final.exists():
        raise FileExistsError(f"기존 bundle 자동 덮어쓰기 금지: {final}")
    temp_root = root / "_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp = temp_root / f"{video_id}_{uuid.uuid4().hex}"
    temp.mkdir()
    try:
        (temp / "context_crops").mkdir()
        if debug_previews:
            (temp / "debug_previews").mkdir()
            for index, preview in debug_previews.items():
                path = temp / "debug_previews" / f"k{index:03d}.jpg"
                if not cv2.imwrite(str(path), preview, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise OSError(f"debug preview 저장 실패: {path}")
        for index, rgb in crops.items():
            started = perf_counter()
            write_rgb_jpeg(temp / rows[index]["context_crop_relpath"], rgb, quality)
            rows[index]["context_write_ms"] = (perf_counter() - started) * 1000.0
        write_csv(temp / "frames.csv", FRAME_COLUMNS, rows)
        np.savez_compressed(temp / "landmarks.npz", landmarks=landmarks.astype(np.float32),
                            landmark_valid=np.asarray([row["landmark_success"] for row in rows], dtype=bool),
                            canonical_index=np.arange(100, dtype=np.int32),
                            source_frame_index=np.asarray([row["source_frame_index"] for row in rows], dtype=np.int32))
        (temp / "video_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (temp / "COMPLETE.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        errors = validate_bundle(temp, strict_crop_decode)
        if errors:
            raise ValueError(f"bundle 무결성 오류: {errors}")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"기존 bundle 자동 덮어쓰기 금지: {final}")
        os.replace(temp, final)
        return final
    except Exception:
        if temp.resolve().parent != temp_root.resolve():
            raise RuntimeError("임시 bundle 경로가 허용 범위를 벗어났습니다")
        shutil.rmtree(temp)
        raise


def build_global_manifests(root: Path, strict_crop_decode: bool = False) -> dict[str, int]:
    """검증된 COMPLETE bundle만 읽고 전체 CSV를 재구성한다."""

    frames: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in ("train", "val"):
        parent = root / "per_video" / split
        for bundle in sorted(parent.iterdir()) if parent.is_dir() else []:
            if not bundle.is_dir() or not (bundle / "COMPLETE.json").is_file():
                continue
            errors = validate_bundle(bundle, strict_crop_decode)
            if errors:
                raise ValueError(f"완료 bundle 무결성 오류 {bundle}: {errors}")
            if bundle.name in seen:
                raise ValueError(f"video_id 중복: {bundle.name}")
            seen.add(bundle.name)
            with (bundle / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
                frames.extend(csv.DictReader(handle))
            videos.append(json.loads((bundle / "video_summary.json").read_text(encoding="utf-8")))
    errors = validate_global_rows(frames, videos)
    if errors:
        raise ValueError(f"global 무결성 오류: {errors}")
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    write_csv(metadata / "canonical_frames.csv", FRAME_COLUMNS, frames)
    write_csv(metadata / "canonical_videos.csv", VIDEO_COLUMNS, videos)
    return {"video_count": len(videos), "frame_count": len(frames)}
