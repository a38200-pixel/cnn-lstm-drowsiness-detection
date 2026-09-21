"""STEP 5-A Context RGB sequence를 read-only로 검수하는 시각화 도구."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


CANONICAL_POLICY_HASH = "29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721"
SEQUENCE_MISSING_POLICY_HASH = "f382c7900331c0be2f56b95f8fd95ed77f86dd5d4b06239da078e03d180eee4d"
ALLOWED_SPLITS = ("train", "val")
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class VisualizationBlocked(ValueError):
    """Read-only visualization 안전 조건을 만족하지 못했음을 나타낸다."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in {"true", "false"}:
        raise VisualizationBlocked(f"VISUALIZATION_BLOCKED_INVALID_BOOLEAN: {value}")
    return normalized == "true"


def resolve_sequence_path(project_root: Path, video_id: str,
                          split: str | None = None) -> tuple[str, Path]:
    """Test 경로를 보지 않고 train/val에서만 sequence를 찾는다."""
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise VisualizationBlocked("VISUALIZATION_BLOCKED_INVALID_VIDEO_ID")
    if split == "test":
        raise VisualizationBlocked("VISUALIZATION_BLOCKED_TEST_ACCESS")
    if split is not None and split not in ALLOWED_SPLITS:
        raise VisualizationBlocked(f"VISUALIZATION_BLOCKED_INVALID_SPLIT: {split}")
    context_root = project_root / "data/interim/sequences_v2/context/per_video"
    candidates = [(name, context_root / name / video_id / "sequence.csv")
                  for name in ((split,) if split else ALLOWED_SPLITS)]
    existing = [(name, path) for name, path in candidates if path.is_file()]
    if len(existing) > 1:
        raise VisualizationBlocked(f"VISUALIZATION_BLOCKED_DUPLICATE_SPLIT: {video_id}")
    if not existing:
        raise FileNotFoundError(f"Context sequence를 찾을 수 없습니다: {video_id}")
    return existing[0]


def decode_context_jpeg(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise VisualizationBlocked(f"VISUALIZATION_BLOCKED_UNDECODABLE_JPEG: {path}")
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise VisualizationBlocked(
            f"VISUALIZATION_BLOCKED_IMAGE_SHAPE: {path} {image.shape} {image.dtype}")
    return image


def load_context_sequence(project_root: Path, video_id: str,
                          split: str | None = None) -> tuple[list[dict[str, Any]], list[np.ndarray], Path]:
    resolved_split, sequence_path = resolve_sequence_path(project_root, video_id, split)
    raw_rows = _read_csv(sequence_path)
    if len(raw_rows) != 32:
        raise VisualizationBlocked(f"VISUALIZATION_BLOCKED_SEQUENCE_LENGTH: {len(raw_rows)}")
    indices = [int(row["target_context_index"]) for row in raw_rows]
    if indices != list(range(32)):
        raise VisualizationBlocked("VISUALIZATION_BLOCKED_TARGET_ORDER")
    project_resolved = project_root.resolve()
    rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    for row in raw_rows:
        if row["split"] == "test" or "test" in Path(row["source_image_relpath"]).parts:
            raise VisualizationBlocked("VISUALIZATION_BLOCKED_TEST_ACCESS")
        if row["split"] != resolved_split or row["video_id"] != video_id:
            raise VisualizationBlocked("VISUALIZATION_BLOCKED_SEQUENCE_IDENTITY")
        if (row["canonical_policy_hash"] != CANONICAL_POLICY_HASH
                or row["sequence_missing_policy_hash"] != SEQUENCE_MISSING_POLICY_HASH):
            raise VisualizationBlocked("VISUALIZATION_BLOCKED_POLICY_MISMATCH")
        source_path = (project_root / row["source_image_relpath"]).resolve()
        if not source_path.is_relative_to(project_resolved):
            raise VisualizationBlocked("VISUALIZATION_BLOCKED_SOURCE_PATH")
        if not source_path.is_file():
            raise FileNotFoundError(f"VISUALIZATION_BLOCKED_MISSING_SOURCE: {source_path}")
        normalized = {
            **row,
            "target_context_index": int(row["target_context_index"]),
            "target_canonical_index": int(row["target_canonical_index"]),
            "target_timestamp_sec": float(row["target_timestamp_sec"]),
            "original_available": _as_bool(row["original_available"]),
            "imputed": _as_bool(row["imputed"]),
            "source_context_index": int(row["source_context_index"]),
            "source_canonical_index": int(row["source_canonical_index"]),
            "source_timestamp_sec": float(row["source_timestamp_sec"]),
            "source_path": source_path,
        }
        rows.append(normalized)
        images.append(decode_context_jpeg(source_path))
    return rows, images, sequence_path


def source_reuse(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_image_relpath"])].append(int(row["target_context_index"]))
    return [{"source_image_relpath": path, "target_indices": targets,
             "reuse_count": len(targets)}
            for path, targets in sorted(groups.items()) if len(targets) > 1]


def sequence_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 32:
        raise VisualizationBlocked("VISUALIZATION_BLOCKED_SEQUENCE_LENGTH")
    imputed = [int(row["target_context_index"]) for row in rows if row["imputed"]]
    reuse = source_reuse(rows)
    return {
        "video_id": rows[0]["video_id"], "split": rows[0]["split"],
        "label": rows[0]["label"], "frames": 32,
        "original_count": sum(bool(row["original_available"]) for row in rows),
        "imputed_count": len(imputed),
        "unique_source_images": len({row["source_image_relpath"] for row in rows}),
        "target_indices": [int(row["target_context_index"]) for row in rows],
        "imputed_targets": imputed, "source_reuse": reuse,
        "canonical_start": int(rows[0]["target_canonical_index"]),
        "canonical_end": int(rows[-1]["target_canonical_index"]),
        "time_start": float(rows[0]["target_timestamp_sec"]),
        "time_end": float(rows[-1]["target_timestamp_sec"]),
        "canonical_policy_hash": rows[0]["canonical_policy_hash"],
        "sequence_missing_policy_hash": rows[0]["sequence_missing_policy_hash"],
    }


def annotation_lines(row: Mapping[str, Any], shared_targets: Sequence[int]) -> list[str]:
    status = "IMPUTED" if row["imputed"] else "ORIGINAL"
    lines = [f"T{row['target_context_index']:02d}  C{row['target_canonical_index']:03d}  "
             f"{row['target_timestamp_sec']:.2f}s", status]
    if row["imputed"]:
        lines.append(f"T{row['target_context_index']:02d} <- S{row['source_context_index']:02d}  "
                     f"TC{row['target_canonical_index']} <- SC{row['source_canonical_index']}")
    elif len(shared_targets) > 1:
        targets = ",".join(f"T{index:02d}" for index in shared_targets)
        lines.append(f"SOURCE DUP x{len(shared_targets)}: {targets}")
    return lines


def _annotated_tile(image: np.ndarray, row: Mapping[str, Any], tile_size: int,
                    shared_targets: Sequence[int]) -> np.ndarray:
    if tile_size < 96:
        raise ValueError("tile_size는 96 이상이어야 합니다")
    display = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    border = 6 if row["imputed"] else 2
    color = (0, 140, 255) if row["imputed"] else (210, 210, 210)
    cv2.rectangle(display, (0, 0), (tile_size - 1, tile_size - 1), color, border)
    band_height = 62
    tile = np.zeros((tile_size + band_height, tile_size, 3), dtype=np.uint8)
    tile[:tile_size] = display
    lines = annotation_lines(row, shared_targets)
    font_scale = max(0.34, min(0.48, tile_size / 400.0))
    for index, line in enumerate(lines[:3]):
        cv2.putText(tile, line, (5, tile_size + 17 + index * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return tile


def render_contact_sheet(rows: Sequence[Mapping[str, Any]], images: Sequence[np.ndarray],
                         tile_size: int = 180) -> np.ndarray:
    if len(rows) != 32 or len(images) != 32:
        raise VisualizationBlocked("VISUALIZATION_BLOCKED_SEQUENCE_LENGTH")
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_image_relpath"])].append(int(row["target_context_index"]))
    tiles = [_annotated_tile(image, row, tile_size,
                             groups[str(row["source_image_relpath"])])
             for row, image in zip(rows, images)]
    return np.vstack([np.hstack(tiles[start:start + 8]) for start in range(0, 32, 8)])


def render_imputed_sheet(rows: Sequence[Mapping[str, Any]], images: Sequence[np.ndarray],
                         tile_size: int = 180) -> np.ndarray:
    pairs = [(row, image) for row, image in zip(rows, images) if row["imputed"]]
    if not pairs:
        canvas = np.zeros((140, tile_size * 2, 3), dtype=np.uint8)
        cv2.putText(canvas, "NO IMPUTED TARGETS", (15, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    output_rows = []
    for row, image in pairs:
        left_row = {**row, "imputed": True}
        left = _annotated_tile(image, left_row, tile_size,
                               [int(row["target_context_index"])])
        cv2.putText(left, "TARGET SLOT (DISPLAYED SOURCE)", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1, cv2.LINE_AA)
        right_row = {**row, "target_context_index": int(row["source_context_index"]),
                     "target_canonical_index": int(row["source_canonical_index"]),
                     "target_timestamp_sec": float(row["source_timestamp_sec"]),
                     "imputed": False}
        right = _annotated_tile(image, right_row, tile_size,
                                [int(row["source_context_index"])])
        cv2.putText(right, "REPLACEMENT SOURCE IMAGE", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
        output_rows.append(np.hstack([left, right]))
    return np.vstack(output_rows)


def output_filename(split: str, video_id: str, mode: str) -> str:
    suffix = "contact" if mode == "contact" else "imputed"
    return f"{split}_{video_id}_context32_{suffix}.jpg"


def save_visualization(output_dir: Path, sheet: np.ndarray,
                       summary: Mapping[str, Any], mode: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / output_filename(summary["split"], summary["video_id"], mode)
    summary_path = output_dir / f"{summary['split']}_{summary['video_id']}_context32_summary.json"
    if image_path.exists():
        raise FileExistsError(f"기존 visualization output이 있습니다: {image_path}")
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != dict(summary):
            raise FileExistsError(f"기존 visualization summary가 현재 입력과 다릅니다: {summary_path}")
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("contact sheet JPEG encoding 실패")
    encoded.tofile(image_path)
    if not summary_path.exists():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return image_path, summary_path


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"VIDEO: {summary['video_id']}")
    print(f"SPLIT: {summary['split']}")
    print(f"LABEL: {summary['label']}")
    print(f"CONTEXT FRAMES: {summary['frames']}")
    print(f"ORIGINAL: {summary['original_count']}")
    print(f"IMPUTED: {summary['imputed_count']}")
    print(f"UNIQUE SOURCE IMAGES: {summary['unique_source_images']}")
    print(f"CANONICAL: {summary['canonical_start']} -> {summary['canonical_end']}")
    print(f"TIME: {summary['time_start']:.2f}s -> {summary['time_end']:.2f}s")
    print(f"IMPUTED TARGETS: {summary['imputed_targets']}")
    for reuse in summary["source_reuse"]:
        targets = ",".join(f"T{index:02d}" for index in reuse["target_indices"])
        print(f"SOURCE REUSE: {targets} -> {reuse['source_image_relpath']}")


def play_sequence(rows: Sequence[Mapping[str, Any]], images: Sequence[np.ndarray],
                  fps: float = 3.2) -> None:
    """표시 속도일 뿐 원본 영상 FPS를 의미하지 않는 inspection playback이다."""
    if fps <= 0:
        raise ValueError("fps는 양수여야 합니다")
    delay = max(1, round(1000 / fps))
    index = 0
    paused = False
    window = "Context Sequence Visual Inspection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while 0 <= index < len(rows):
        row, image = rows[index], images[index]
        canvas = np.zeros((330, 520, 3), dtype=np.uint8)
        canvas[:224, :224] = image
        lines = [f"Video: {row['video_id']}  Split: {row['split']}  Label: {row['label']}",
                 f"Target: {index} / 31  Canonical: {row['target_canonical_index']}",
                 f"Timestamp: {row['target_timestamp_sec']:.2f}s",
                 f"Status: {'IMPUTED' if row['imputed'] else 'ORIGINAL'}",
                 f"Source Context: {row['source_context_index']}  Source Canonical: {row['source_canonical_index']}",
                 "SPACE pause/resume | LEFT/RIGHT step | Q/ESC quit"]
        for line_index, line in enumerate(lines):
            cv2.putText(canvas, line, (8, 245 + line_index * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKeyEx(0 if paused else delay)
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == 32:
            paused = not paused
        elif key in (81, 2424832, 65361):
            index = max(0, index - 1)
            paused = True
        elif key in (83, 2555904, 65363):
            index = min(len(rows) - 1, index + 1)
            paused = True
        elif not paused:
            index += 1
    cv2.destroyWindow(window)
