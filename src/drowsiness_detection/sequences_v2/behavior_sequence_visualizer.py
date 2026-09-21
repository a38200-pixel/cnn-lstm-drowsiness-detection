"""STEP 5-B Behavior 100-slot sequence를 읽기 전용으로 시각 검수한다."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drowsiness_detection.preprocessing_v2.preprocessing_provenance import file_sha256
from drowsiness_detection.sequences_v2.behavior_sequence_builder import (
    FEATURE_NAMES, FEATURE_SCHEMA_HASH, POLICY_HASH, SEQUENCE_POLICY_HASH,
)


EXPECTED_KEYS = {
    "canonical_index", "target_timestamp_sec", "behavior_features",
    "behavior_valid_mask", "detector_valid_mask", "landmark_valid_mask",
    "pose_valid_mask", "feature_valid_mask",
}
MASK_NAMES = (
    ("detector_valid_mask", "Detector valid"),
    ("landmark_valid_mask", "Landmark valid"),
    ("pose_valid_mask", "Pose valid"),
    ("behavior_valid_mask", "Behavior valid"),
)


class BehaviorVisualizationBlocked(RuntimeError):
    """Behavior visualization 입력이 동결 계약을 위반할 때 사용한다."""


@dataclass(frozen=True)
class BehaviorSequence:
    """검증을 마친 하나의 읽기 전용 Behavior sequence view다."""

    video_id: str
    split: str
    label: str
    bundle_path: Path
    features: np.ndarray
    canonical_index: np.ndarray
    stored_timestamp_sec: np.ndarray
    feature_valid_mask: np.ndarray
    detector_valid_mask: np.ndarray
    landmark_valid_mask: np.ndarray
    pose_valid_mask: np.ndarray
    behavior_valid_mask: np.ndarray
    canonical_policy_hash: str
    sequence_missing_policy_hash: str
    behavior_feature_schema_hash: str
    source_metadata_fingerprint: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized not in {"true", "false"}:
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_MASK_MISMATCH")
    return normalized == "true"


def timeline_seconds() -> np.ndarray:
    """검수용 10 Hz slot timeline 0.0~9.9초를 반환한다."""
    return np.arange(100, dtype=np.float64) / 10.0


def resolve_behavior_bundle(project_root: Path, video_id: str,
                            split: str | None = None) -> Path:
    """Test를 탐색하지 않고 train/val에서만 bundle을 결정한다."""
    if split == "test":
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_BLOCKED_TEST_ACCESS")
    if split is not None and split not in {"train", "val"}:
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_BLOCKED_TEST_ACCESS")
    if not video_id or Path(video_id).name != video_id or any(char in video_id for char in "/\\"):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_VIDEO_NOT_FOUND")
    root = project_root / "data/interim/sequences_v2/behavior/per_video"
    candidates = ([root / split / video_id] if split else
                  [root / allowed / video_id for allowed in ("train", "val")])
    found = [path for path in candidates if path.is_dir()]
    if len(found) > 1:
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_AMBIGUOUS_VIDEO")
    if not found:
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_VIDEO_NOT_FOUND")
    return found[0]


def _validate_hashes(bundle: Path, marker: Mapping[str, Any]) -> None:
    for filename in ("sequence.csv", "behavior.npz", "summary.json"):
        path = bundle / filename
        if not path.is_file() or marker.get("file_sha256", {}).get(filename) != file_sha256(path):
            raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_PROVENANCE_MISMATCH")


def _validate_shapes(arrays: Mapping[str, np.ndarray]) -> None:
    if (arrays["behavior_features"].shape != (100, 6)
            or arrays["behavior_features"].dtype != np.float32
            or arrays["canonical_index"].shape != (100,)
            or arrays["target_timestamp_sec"].shape != (100,)
            or arrays["feature_valid_mask"].shape != (100, 6)
            or arrays["feature_valid_mask"].dtype != np.bool_):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")
    for key, _ in MASK_NAMES:
        if arrays[key].shape != (100,) or arrays[key].dtype != np.bool_:
            raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")


def load_behavior_sequence(project_root: Path, video_id: str,
                           split: str | None = None) -> BehaviorSequence:
    """실제 STEP 5-B bundle을 수정하지 않고 읽어 schema·mask·hash를 검증한다."""
    bundle = resolve_behavior_bundle(project_root, video_id, split)
    required = ("sequence.csv", "behavior.npz", "summary.json", "COMPLETE.json")
    if any(not (bundle / filename).is_file() for filename in required):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_ARTIFACT_MISSING")
    summary = _read_json(bundle / "summary.json")
    marker = _read_json(bundle / "COMPLETE.json")
    _validate_hashes(bundle, marker)
    if summary.get("feature_names") != list(FEATURE_NAMES):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_FEATURE_ORDER_MISMATCH")
    if (summary.get("video_id") != video_id or summary.get("split") != bundle.parent.name
            or summary.get("sequence_length") != 100 or summary.get("feature_shape") != [100, 6]):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")
    hashes = {
        "canonical_policy_hash": POLICY_HASH,
        "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
        "behavior_feature_schema_hash": FEATURE_SCHEMA_HASH,
    }
    if any(summary.get(key) != value or marker.get(key) != value
           for key, value in hashes.items()):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_PROVENANCE_MISMATCH")
    with np.load(bundle / "behavior.npz", allow_pickle=False) as loaded:
        if set(loaded.files) != EXPECTED_KEYS:
            raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")
        arrays = {key: loaded[key].copy() for key in loaded.files}
    _validate_shapes(arrays)
    features = arrays["behavior_features"]
    if np.isinf(features).any():
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_NUMERIC_ANOMALY")
    if not np.array_equal(np.isfinite(features), arrays["feature_valid_mask"]):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_MASK_MISMATCH")
    if not np.array_equal(arrays["canonical_index"], np.arange(100, dtype=np.int16)):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")

    rows = _read_csv(bundle / "sequence.csv")
    if len(rows) != 100 or [int(row["canonical_index"]) for row in rows] != list(range(100)):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")
    if any(row["video_id"] != video_id or row["split"] != bundle.parent.name
           or row["label"] != summary.get("label") for row in rows):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_PROVENANCE_MISMATCH")
    csv_features = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows],
                              dtype=np.float32)
    if not np.array_equal(features, csv_features, equal_nan=True):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_PROVENANCE_MISMATCH")
    csv_timestamps = np.asarray([float(row["target_timestamp_sec"]) for row in rows],
                                dtype=np.float64)
    if not np.array_equal(arrays["target_timestamp_sec"], csv_timestamps):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_PROVENANCE_MISMATCH")
    detector = np.asarray([_as_bool(row["yunet_success"]) for row in rows])
    landmark = np.asarray([_as_bool(row["landmark_success"]) for row in rows])
    pose_success = np.asarray([_as_bool(row["pose_success"]) for row in rows])
    behavior_csv = np.asarray([_as_bool(row["behavior_valid"]) for row in rows])
    expected_behavior = detector & landmark & arrays["feature_valid_mask"][:, :2].all(axis=1)
    expected_pose = pose_success & arrays["feature_valid_mask"][:, 2:].all(axis=1)
    if (not np.array_equal(detector, arrays["detector_valid_mask"])
            or not np.array_equal(landmark, arrays["landmark_valid_mask"])
            or not np.array_equal(behavior_csv, arrays["behavior_valid_mask"])
            or not np.array_equal(expected_behavior, arrays["behavior_valid_mask"])
            or not np.array_equal(expected_pose, arrays["pose_valid_mask"])):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_MASK_MISMATCH")
    if any(row["canonical_policy_hash"] != POLICY_HASH
           or row["sequence_missing_policy_hash"] != SEQUENCE_POLICY_HASH for row in rows):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_PROVENANCE_MISMATCH")
    valid_count = int(arrays["behavior_valid_mask"].sum())
    if (summary.get("behavior_valid_count") != valid_count
            or summary.get("behavior_missing_count") != 100 - valid_count):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_MASK_MISMATCH")
    return BehaviorSequence(
        video_id=video_id, split=bundle.parent.name, label=str(summary["label"]),
        bundle_path=bundle, features=features, canonical_index=arrays["canonical_index"],
        stored_timestamp_sec=arrays["target_timestamp_sec"],
        feature_valid_mask=arrays["feature_valid_mask"],
        detector_valid_mask=arrays["detector_valid_mask"],
        landmark_valid_mask=arrays["landmark_valid_mask"],
        pose_valid_mask=arrays["pose_valid_mask"],
        behavior_valid_mask=arrays["behavior_valid_mask"],
        canonical_policy_hash=summary["canonical_policy_hash"],
        sequence_missing_policy_hash=summary["sequence_missing_policy_hash"],
        behavior_feature_schema_hash=summary["behavior_feature_schema_hash"],
        source_metadata_fingerprint=summary["source_metadata_fingerprint"],
    )


def false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """False가 연속된 구간을 양 끝을 포함하는 slot index로 반환한다."""
    if mask.shape != (100,) or mask.dtype != np.bool_:
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_SHAPE_MISMATCH")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, valid in enumerate(mask):
        if not valid and start is None:
            start = index
        elif valid and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def feature_numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """NaN을 채우지 않고 finite 값만 기술 통계로 요약한다."""
    values = np.asarray(values)
    if np.isinf(values).any():
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_NUMERIC_ANOMALY")
    finite = values[np.isfinite(values)]
    result: dict[str, float | int | None] = {
        "finite_count": int(finite.size), "nan_count": int(np.isnan(values).sum()),
        "min": None, "max": None, "mean": None, "median": None,
    }
    if finite.size:
        result.update({"min": float(finite.min()), "max": float(finite.max()),
                       "mean": float(finite.mean()), "median": float(np.median(finite))})
    return result


def sequence_summary(sequence: BehaviorSequence) -> dict[str, Any]:
    """Console과 JSON에 공통으로 사용할 read-only 진단 요약을 만든다."""
    missing_slots = np.flatnonzero(~sequence.behavior_valid_mask).astype(int).tolist()
    behavior_runs = false_runs(sequence.behavior_valid_mask)
    detector_runs = false_runs(sequence.detector_valid_mask)
    numeric = {name: feature_numeric_summary(sequence.features[:, index])
               for index, name in enumerate(FEATURE_NAMES)}
    masks = {label: {"valid": int(getattr(sequence, key).sum()),
                     "invalid": int((~getattr(sequence, key)).sum())}
             for key, label in MASK_NAMES}
    return {
        "video_id": sequence.video_id, "split": sequence.split, "label": sequence.label,
        "sequence_shape": list(sequence.features.shape), "dtype": str(sequence.features.dtype),
        "timeline_hz": 10, "timeline_start_sec": 0.0, "timeline_end_sec": 9.9,
        "valid_count": int(sequence.behavior_valid_mask.sum()),
        "missing_count": len(missing_slots),
        "longest_missing_run": max((end - start + 1 for start, end in behavior_runs), default=0),
        "longest_detector_missing_run": max(
            (end - start + 1 for start, end in detector_runs), default=0),
        "feature_order": list(FEATURE_NAMES), "feature_numeric_summary": numeric,
        "mask_counts": masks, "missing_slots": missing_slots,
        "canonical_policy_hash": sequence.canonical_policy_hash,
        "sequence_missing_policy_hash": sequence.sequence_missing_policy_hash,
        "behavior_feature_schema_hash": sequence.behavior_feature_schema_hash,
        "source_metadata_fingerprint": sequence.source_metadata_fingerprint,
        "interpretation": "visual inspection only; no thresholds or behavior events",
    }


def _shade_missing(ax: Any, mask: np.ndarray) -> None:
    for run_index, (start, end) in enumerate(false_runs(mask)):
        ax.axvspan(max(0.0, start / 10.0 - 0.05), min(9.9, end / 10.0 + 0.05),
                   facecolor="0.85", edgecolor="0.35", hatch="//", alpha=0.45,
                   label="Missing / invalid slot" if run_index == 0 else None)


def _mask_matrix(sequence: BehaviorSequence) -> np.ndarray:
    return np.vstack([getattr(sequence, key) for key, _ in MASK_NAMES]).astype(np.uint8)


def render_overview(sequence: BehaviorSequence, dpi: int = 150) -> Any:
    """EAR, MAR, pose 연속값과 실제 mask를 5개 공유 시간축 panel로 표시한다."""
    if dpi <= 0:
        raise ValueError("dpi는 양수여야 합니다")
    import matplotlib.pyplot as plt

    time = timeline_seconds()
    figure, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, dpi=dpi,
                               gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.9]})
    plot_specs = (
        (axes[0], ((0, "EAR"),), "EAR"),
        (axes[1], ((1, "MAR"),), "MAR"),
        (axes[2], ((2, "pitch_raw"), (3, "pitch_centered_candidate")), "Pitch (degrees)"),
        (axes[3], ((4, "yaw"), (5, "roll")), "Yaw / Roll (degrees)"),
    )
    for ax, series, ylabel in plot_specs:
        for column, label in series:
            ax.plot(time, sequence.features[:, column], linewidth=1.25, label=label)
        _shade_missing(ax, sequence.behavior_valid_mask)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if len(series) > 1 or not sequence.behavior_valid_mask.all():
            ax.legend(loc="best", fontsize=8)
    matrix = _mask_matrix(sequence)
    axes[4].imshow(matrix, aspect="auto", interpolation="nearest", cmap="gray",
                   vmin=0, vmax=1, extent=(-0.05, 9.95, 3.5, -0.5))
    axes[4].set_yticks(range(4), [label for _, label in MASK_NAMES])
    axes[4].set_ylabel("Masks")
    axes[4].set_title("Mask value: white=valid (1), black=invalid (0)", fontsize=9)
    axes[4].set_xlim(0.0, 9.9)
    axes[4].set_xticks(np.arange(0, 10, 1))
    axes[4].set_xlabel("Time (sec)")
    valid = int(sequence.behavior_valid_mask.sum())
    figure.suptitle(
        "Behavior Sequence Inspection\n"
        f"video={sequence.video_id} | split={sequence.split} | label={sequence.label} | "
        f"valid={valid}/100 | missing={100 - valid}\n"
        "Label is metadata only; no threshold/event or head-pose direction is interpreted.",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def render_masks(sequence: BehaviorSequence, dpi: int = 150) -> Any:
    """4개 STEP 5-B mask만 100-slot heatmap으로 표시한다."""
    if dpi <= 0:
        raise ValueError("dpi는 양수여야 합니다")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(14, 4), dpi=dpi)
    ax.imshow(_mask_matrix(sequence), aspect="auto", interpolation="nearest", cmap="gray",
              vmin=0, vmax=1, extent=(-0.5, 99.5, 3.5, -0.5))
    ax.set_yticks(range(4), [label for _, label in MASK_NAMES])
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_xlabel("Slot index (10 Hz; slot / 10 = seconds)")
    ax.set_title(
        f"Behavior Masks | video={sequence.video_id} | split={sequence.split} | "
        "white=valid (1), black=invalid (0)"
    )
    figure.tight_layout()
    return figure


def output_filename(split: str, video_id: str, mode: str) -> str:
    if mode not in {"overview", "masks"}:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    return f"{split}_{video_id}_behavior100_{mode}.png"


def ensure_visualization_output(project_root: Path, output_dir: Path) -> Path:
    """출력 위치를 local visualization root 아래로 제한한다."""
    root = (project_root / "outputs/visualizations").resolve()
    candidate = (output_dir if output_dir.is_absolute() else project_root / output_dir).resolve()
    if candidate != root and root not in candidate.parents:
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_OUTPUT_OUTSIDE_LOCAL_ROOT")
    return candidate


def save_visualization(project_root: Path, output_dir: Path, figure: Any,
                       summary: Mapping[str, Any], mode: str, dpi: int = 150) -> tuple[Path, Path]:
    output_dir = ensure_visualization_output(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / output_filename(summary["split"], summary["video_id"], mode)
    summary_path = output_dir / f"{summary['split']}_{summary['video_id']}_behavior100_summary.json"
    if image_path.exists():
        raise FileExistsError(f"기존 visualization output이 있습니다: {image_path}")
    if summary_path.exists():
        existing = _read_json(summary_path)
        if existing != dict(summary):
            raise FileExistsError(f"기존 visualization summary가 현재 입력과 다릅니다: {summary_path}")
    figure.savefig(image_path, dpi=dpi, bbox_inches="tight")
    if not summary_path.exists():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return image_path, summary_path


def suggest_samples(project_root: Path) -> list[dict[str, str]]:
    """Train/val global metadata에서 네 가지 대표 유형을 결정적으로 추천한다."""
    path = project_root / "data/interim/sequences_v2/behavior/metadata/behavior_sequence_videos.csv"
    rows = _read_csv(path)
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise BehaviorVisualizationBlocked("BEHAVIOR_VISUALIZATION_BLOCKED_TEST_ACCESS")
    categories = (
        ("FULL_VALID", lambda row: int(row["behavior_valid_count"]) == 100),
        ("VALID_99", lambda row: int(row["behavior_valid_count"]) == 99),
        ("LOWEST_VALID", lambda row: int(row["behavior_valid_count"]) == 95),
        ("LONGEST_MISSING_RUN", lambda row: int(row["longest_yunet_missing_run"]) == 5),
    )
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for category, predicate in categories:
        candidate = next((row for row in rows if predicate(row) and row["video_id"] not in used), None)
        if candidate:
            used.add(candidate["video_id"])
            selected.append({"category": category, "video_id": candidate["video_id"],
                             "split": candidate["split"], "label": candidate["label"],
                             "valid_count": candidate["behavior_valid_count"],
                             "longest_missing_run": candidate["longest_yunet_missing_run"]})
    return selected


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"VIDEO: {summary['video_id']}")
    print(f"SPLIT: {summary['split']}")
    print(f"LABEL: {summary['label']}")
    print(f"SEQUENCE SHAPE: {summary['sequence_shape']}")
    print(f"DTYPE: {summary['dtype']}")
    print(f"VALID SLOTS: {summary['valid_count']}")
    print(f"MISSING SLOTS: {summary['missing_count']}")
    print(f"LONGEST MISSING RUN: {summary['longest_missing_run']}")
    print("FEATURES: " + ", ".join(name.upper() for name in summary["feature_order"]))
    for name, values in summary["feature_numeric_summary"].items():
        print(f"{name.upper()} FINITE: {values['finite_count']} / NAN: {values['nan_count']}")
    print("MASK COUNTS:")
    for name, values in summary["mask_counts"].items():
        print(f"  {name}: valid={values['valid']} invalid={values['invalid']}")
