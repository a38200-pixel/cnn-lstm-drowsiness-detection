"""STEP 4-C1 train/val 결측 정책 후보의 영향만 읽기 전용으로 시뮬레이션한다."""

from __future__ import annotations

import csv
import json
import math
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


POLICY_HASH = "29f74d12e4a522352868297c1661224c5d444f2829f1cae6866c8b2d1968e721"
CONTEXT_CANDIDATES = (
    ("C0_COMPLETE_ONLY", 0, 0),
    ("C1_ISOLATED_ONLY", 2, 1),
    ("C2_SHORT_GAP", 4, 2),
    ("C3_MODERATE_GAP", 8, 4),
    ("C4_COVERAGE_ONLY", 8, None),
)
BEHAVIOR_CANDIDATES = (
    ("B0_MASK_ONLY", 1, None),
    ("B1_COVERAGE_90", 90, 10),
    ("B2_COVERAGE_95", 95, 5),
    ("B3_COVERAGE_80", 80, 20),
)
GROUPS = (
    ("overall", "all"), ("split", "train"), ("split", "val"),
    ("label", "drowsy"), ("label", "not_drowsy"),
    ("split_label", "train:drowsy"), ("split_label", "train:not_drowsy"),
    ("split_label", "val:drowsy"), ("split_label", "val:not_drowsy"),
)
CONTEXT_IMPACT_COLUMNS = (
    "candidate", "eligible_videos", "excluded_videos", "retention_rate",
    "train_retention", "val_retention", "drowsy_retention", "not_drowsy_retention",
    "train_drowsy_retention", "train_not_drowsy_retention", "val_drowsy_retention", "val_not_drowsy_retention",
    "min_stratum_retention", "videos_requiring_no_fill", "videos_requiring_fill", "total_substituted_slots",
    "mean_substitutions_per_eligible_video", "median_substitutions", "p90_substitutions", "max_substitutions",
    "total_original_valid_context_slots", "substituted_slot_fraction", "max_allowed_missing_fraction",
    "max_missing_count_retained", "max_missing_run_retained", "max_missing_time_span_sec",
    "median_missing_time_span_sec", "p90_missing_time_span_sec", "p95_missing_time_span_sec",
    "label_ratio_shift_pp", "split_ratio_shift_pp", "max_absolute_ratio_shift_pp",
    "retention_ge_95pct", "min_stratum_retention_ge_90pct", "substitution_fraction_le_5pct",
    "max_missing_fraction_le_25pct", "has_consecutive_run_guard", "substitution_count_histogram",
)
BEHAVIOR_IMPACT_COLUMNS = (
    "candidate", "eligible_videos", "excluded_videos", "retention_rate",
    "train_retention", "val_retention", "drowsy_retention", "not_drowsy_retention",
    "train_drowsy_retention", "train_not_drowsy_retention", "val_drowsy_retention", "val_not_drowsy_retention",
    "min_stratum_retention", "mean_valid_frames", "median_valid_frames", "p10_valid_frames",
    "max_missing_count_retained", "max_missing_run_retained",
    "label_ratio_shift_pp", "split_ratio_shift_pp", "max_absolute_ratio_shift_pp",
    "handling_simulation",
)
EDGE_COLUMNS = (
    "video_id", "split", "label", "context_missing_count", "longest_context_missing_run",
    *(f"{name[:2]}_eligible" for name, _, _ in CONTEXT_CANDIDATES),
    *(f"reason_excluded_{name[:2]}" for name, _, _ in CONTEXT_CANDIDATES),
)
STRATUM_COLUMNS = ("branch", "candidate", "group_type", "group_name", "original_videos",
                   "eligible_videos", "excluded_videos", "retention_rate")
CROSS_COLUMNS = ("context_candidate", "behavior_candidate", "context_eligible", "behavior_eligible", "videos")


@dataclass(frozen=True)
class VideoState:
    """정책 시뮬레이션에 필요한 기존 metadata만 보유한다."""

    video_id: str
    split: str
    label: str
    context_available: tuple[bool, ...]
    context_indices: tuple[int, ...]
    context_timestamps: tuple[float, ...]
    behavior_valid_count: int
    behavior_missing_count: int
    longest_behavior_missing_run: int

    @property
    def context_missing_count(self) -> int:
        return 32 - sum(self.context_available)

    @property
    def context_runs(self) -> tuple[tuple[int, int], ...]:
        return missing_runs(self.context_available)

    @property
    def longest_context_missing_run(self) -> int:
        return max((end - start + 1 for start, end in self.context_runs), default=0)

    @property
    def context_run_spans_sec(self) -> tuple[float, ...]:
        """각 연속 결측의 첫·끝 selected frame 실제 timestamp 차이다."""
        return tuple(self.context_timestamps[end] - self.context_timestamps[start]
                     for start, end in self.context_runs)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _yes(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def missing_runs(available: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    """Context sequence 순서에서 unavailable 연속 구간을 반환한다."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, valid in enumerate(available):
        if not valid and start is None:
            start = index
        if valid and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(available) - 1))
    return tuple(runs)


def nearest_valid_slot(available: Sequence[bool], target: int) -> int | None:
    """동거리면 이전 slot을 우선하며 영상 밖이나 새 이미지는 쓰지 않는다."""
    if target < 0 or target >= len(available) or available[target]:
        raise ValueError("유효하지 않은 결측 Context slot")
    valid = [index for index, exists in enumerate(available) if exists]
    return min(valid, key=lambda index: (abs(index - target), index)) if valid else None


def context_eligible(state: VideoState, candidate: tuple[str, int, int | None]) -> bool:
    """Count와 32-slot run guard를 분리해 적용한다."""
    _, max_count, max_run = candidate
    return (any(state.context_available) and state.context_missing_count <= max_count
            and (max_run is None or state.longest_context_missing_run <= max_run))


def behavior_eligible(state: VideoState, candidate: tuple[str, int, int | None]) -> bool:
    """유효 frame coverage와 YuNet 연속 결측만 비교하고 보간하지 않는다."""
    _, min_valid, max_run = candidate
    return (state.behavior_valid_count >= min_valid
            and (max_run is None or state.longest_behavior_missing_run <= max_run))


def load_inputs(canonical_root: Path, audit_root: Path) -> list[VideoState]:
    """Test를 차단하고 STEP 4-A/4-B와 영상별 metadata를 교차 검증한다."""
    run = json.loads((canonical_root / "metadata/run_metadata.json").read_text(encoding="utf-8"))
    if run.get("policy_hash") != POLICY_HASH:
        raise ValueError("STEP 3-C canonical policy hash 불일치")
    video_rows = _read_csv(canonical_root / "metadata/canonical_videos.csv")
    quality_rows = _read_csv(audit_root / "per_video_quality.csv")
    missing_run_rows = _read_csv(audit_root / "missing_runs.csv")
    split_label_rows = _read_csv(audit_root / "split_label_quality_summary.csv")
    context_cumulative = _read_csv(audit_root / "context_missing_cumulative.csv")
    behavior_cumulative = _read_csv(audit_root / "behavior_missing_cumulative.csv")
    audit_summary = json.loads((audit_root / "quality_missing_summary.json").read_text(encoding="utf-8"))
    review_summary = json.loads((audit_root / "step4b_visual_review/step4b_manual_review_summary.json").read_text(encoding="utf-8"))
    if len(video_rows) != 1763 or len(quality_rows) != 1763:
        raise ValueError("train/val 1763개 영상이 필요합니다")
    if any(row["split"] not in {"train", "val"} for row in video_rows + quality_rows + missing_run_rows):
        raise ValueError("STEP_4C1_BLOCKED_TEST_LEAKAGE: STEP 3/4 영상 metadata")
    if any(row["group_name"] == "test" for row in split_label_rows):
        raise ValueError("STEP_4C1_BLOCKED_TEST_LEAKAGE: STEP 4-A strata")
    if any(row["policy_hash"] != POLICY_HASH for row in video_rows + quality_rows):
        raise ValueError("영상별 canonical policy hash 불일치")
    if audit_summary.get("policy_hash") != POLICY_HASH or audit_summary.get("test_rows") != 0:
        raise ValueError("STEP 4-A policy hash/test row 불일치")
    if not review_summary.get("manual_review_complete") or review_summary.get("reviewed_candidates") != 36:
        raise ValueError("STEP 4-B 사용자 수동 검토 완료 요약이 필요합니다")
    if review_summary.get("test_used") or review_summary.get("missing_policy_selected"):
        raise ValueError("STEP 4-B 보호 상태 불일치")
    video_map = {row["video_id"]: row for row in video_rows}
    quality_map = {row["video_id"]: row for row in quality_rows}
    if len(video_map) != 1763 or len(quality_map) != 1763 or set(video_map) != set(quality_map):
        raise ValueError("canonical video/quality ID 중복 또는 불일치")
    for row in video_rows:
        quality = quality_map[row["video_id"]]
        if row["split"] != quality["split"] or row["label"] != quality["label"]:
            raise ValueError("video/quality split·label 불일치")
    rows_by_video: dict[str, list[dict[str, str]]] = {video_id: [] for video_id in video_map}
    frame_count = 0
    with (canonical_root / "metadata/canonical_frames.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "val"}:
                raise ValueError("STEP_4C1_BLOCKED_TEST_LEAKAGE: canonical_frames")
            if row["video_id"] not in video_map:
                raise ValueError("canonical_frames의 video_id가 canonical_videos에 없습니다")
            rows_by_video[row["video_id"]].append(row)
            frame_count += 1
    if frame_count != 176300:
        raise ValueError(f"canonical frame 수 불일치: {frame_count}")
    states: list[VideoState] = []
    for video_id in sorted(video_map):
        quality = quality_map[video_id]
        frames = sorted(rows_by_video[video_id], key=lambda row: int(row["canonical_index"]))
        if len(frames) != 100 or [int(row["canonical_index"]) for row in frames] != list(range(100)):
            raise ValueError(f"영상당 canonical 100 slot 오류: {video_id}")
        if any(row["split"] != quality["split"] or row["label"] != quality["label"] for row in frames):
            raise ValueError(f"frame split·label 불일치: {video_id}")
        selected = sorted((row for row in frames if _yes(row["context_selected"])),
                          key=lambda row: int(row["context_slot"]))
        if len(selected) != 32 or [int(row["context_slot"]) for row in selected] != list(range(32)):
            raise ValueError(f"Context 32 slot/순서 오류: {video_id}")
        available = tuple(_yes(row["context_crop_available"]) for row in selected)
        indices = tuple(int(row["canonical_index"]) for row in selected)
        timestamps = tuple(float(row["actual_timestamp_sec"]) for row in selected)
        if any(not math.isfinite(value) for value in timestamps) or any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Context timestamp 순서 오류: {video_id}")
        detector_valid = [(_yes(row["yunet_success"]) and _yes(row["landmark_success"])
                          and row["ear"] not in {"", "nan", "NaN"} and row["mar"] not in {"", "nan", "NaN"})
                          for row in frames]
        missing_yunet = [not _yes(row["yunet_success"]) for row in frames]
        longest_yunet = max((end - start + 1 for start, end in missing_runs([not value for value in missing_yunet])), default=0)
        state = VideoState(video_id, quality["split"], quality["label"], available, indices, timestamps,
                           int(quality["behavior_valid_count"]), int(quality["behavior_missing_count"]),
                           int(quality["longest_yunet_missing_run"]))
        if (state.context_missing_count != int(quality["context_missing_count"])
                or state.longest_context_missing_run != int(quality["longest_context_missing_run"])
                or sum(detector_valid) != state.behavior_valid_count
                or 100 - state.behavior_valid_count != state.behavior_missing_count
                or longest_yunet != state.longest_behavior_missing_run):
            raise ValueError(f"STEP 4-A 영상별 품질 집계 불일치: {video_id}")
        states.append(state)
    run_counts = Counter(row["video_id"] for row in missing_run_rows)
    if any(run_counts[video_id] != int(quality_map[video_id]["number_of_yunet_missing_runs"]) for video_id in video_map):
        raise ValueError("STEP 4-A missing_runs 집계 불일치")
    if sum(state.context_missing_count for state in states) != 1744 or sum(state.behavior_missing_count for state in states) != 5429:
        raise ValueError("STEP 4-A 총 결측 수 불일치")
    if sum(state.split == "train" for state in states) != 1452 or sum(state.split == "val" for state in states) != 311:
        raise ValueError("train/val 영상 수 불일치")
    if not context_cumulative or not behavior_cumulative:
        raise ValueError("STEP 4-A cumulative 입력 누락")
    return states


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def _group(state: VideoState, group_type: str, name: str) -> bool:
    if group_type == "overall":
        return True
    if group_type == "split":
        return state.split == name
    if group_type == "label":
        return state.label == name
    return f"{state.split}:{state.label}" == name


def _stratum_rows(states: Sequence[VideoState], branch: str, candidate: str,
                  eligibility: Mapping[str, bool]) -> list[dict[str, Any]]:
    rows = []
    for group_type, name in GROUPS:
        group = [state for state in states if _group(state, group_type, name)]
        count = sum(eligibility[state.video_id] for state in group)
        rows.append({"branch": branch, "candidate": candidate, "group_type": group_type,
                     "group_name": name, "original_videos": len(group), "eligible_videos": count,
                     "excluded_videos": len(group) - count, "retention_rate": count / len(group) if group else 0.0})
    return rows


def _rates(strata: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {row["group_name"]: row["retention_rate"] for row in strata}


def _shift(states: Sequence[VideoState], retained: Sequence[VideoState]) -> tuple[float, float]:
    drowsy_original = sum(state.label == "drowsy" for state in states) / len(states)
    train_original = sum(state.split == "train" for state in states) / len(states)
    if not retained:
        return 0.0, 0.0
    return (100 * (sum(state.label == "drowsy" for state in retained) / len(retained) - drowsy_original),
            100 * (sum(state.split == "train" for state in retained) / len(retained) - train_original))


def _impact_common(states: Sequence[VideoState], retained: Sequence[VideoState],
                   strata: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates = _rates(strata)
    label_shift, split_shift = _shift(states, retained)
    return {
        "eligible_videos": len(retained), "excluded_videos": len(states) - len(retained),
        "retention_rate": len(retained) / len(states),
        "train_retention": rates["train"], "val_retention": rates["val"],
        "drowsy_retention": rates["drowsy"], "not_drowsy_retention": rates["not_drowsy"],
        "train_drowsy_retention": rates["train:drowsy"],
        "train_not_drowsy_retention": rates["train:not_drowsy"],
        "val_drowsy_retention": rates["val:drowsy"],
        "val_not_drowsy_retention": rates["val:not_drowsy"],
        "min_stratum_retention": min(rates[name] for group_type, name in GROUPS if group_type == "split_label"),
        "label_ratio_shift_pp": label_shift, "split_ratio_shift_pp": split_shift,
        "max_absolute_ratio_shift_pp": max(abs(label_shift), abs(split_shift)),
    }


def _edge_reason(state: VideoState, candidate: tuple[str, int, int | None]) -> str:
    if context_eligible(state, candidate):
        return ""
    if not any(state.context_available):
        return "NO_VALID_CONTEXT_SOURCE"
    _, max_count, max_run = candidate
    reasons = []
    if state.context_missing_count > max_count:
        reasons.append("MISSING_COUNT_EXCEEDS")
    if max_run is not None and state.longest_context_missing_run > max_run:
        reasons.append("RUN_LENGTH_EXCEEDS")
    return ";".join(reasons)


def analyze(states: Sequence[VideoState]) -> dict[str, Any]:
    """모든 후보를 같은 train/val 영상에 적용하되 실제 data drop/fill은 하지 않는다."""
    if len(states) != 1763 or len({state.video_id for state in states}) != len(states):
        raise ValueError("STEP 4-C1은 고유 영상 1763개를 필요로 합니다")
    context_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    context_flags: dict[str, dict[str, bool]] = {}
    behavior_flags: dict[str, dict[str, bool]] = {}
    for candidate in CONTEXT_CANDIDATES:
        name, max_count, max_run = candidate
        flags = {state.video_id: context_eligible(state, candidate) for state in states}
        context_flags[name] = flags
        retained = [state for state in states if flags[state.video_id]]
        strata = _stratum_rows(states, "context", name, flags)
        stratum_rows.extend(strata)
        fill_counts = [state.context_missing_count for state in retained]
        substitutions = sum(fill_counts)
        for state in retained:
            for slot, valid in enumerate(state.context_available):
                if not valid and nearest_valid_slot(state.context_available, slot) is None:
                    raise ValueError(f"유효한 Context 대체 source가 없습니다: {state.video_id}")
        spans = [span for state in retained for span in state.context_run_spans_sec]
        burden = substitutions / (32 * len(retained)) if retained else 0.0
        longest = max((state.longest_context_missing_run for state in retained), default=0)
        maximum = max(fill_counts, default=0)
        common = _impact_common(states, retained, strata)
        row = {
            "candidate": name, **common,
            "videos_requiring_no_fill": sum(count == 0 for count in fill_counts),
            "videos_requiring_fill": sum(count > 0 for count in fill_counts),
            "total_substituted_slots": substitutions,
            "mean_substitutions_per_eligible_video": statistics.mean(fill_counts) if fill_counts else 0.0,
            "median_substitutions": statistics.median(fill_counts) if fill_counts else 0.0,
            "p90_substitutions": _percentile(fill_counts, 90), "max_substitutions": maximum,
            "total_original_valid_context_slots": 32 * len(retained) - substitutions,
            "substituted_slot_fraction": burden, "max_allowed_missing_fraction": max_count / 32,
            "max_missing_count_retained": maximum, "max_missing_run_retained": longest,
            "max_missing_time_span_sec": max(spans, default=0.0),
            "median_missing_time_span_sec": statistics.median(spans) if spans else 0.0,
            "p90_missing_time_span_sec": _percentile(spans, 90),
            "p95_missing_time_span_sec": _percentile(spans, 95),
            "retention_ge_95pct": common["retention_rate"] >= 0.95,
            "min_stratum_retention_ge_90pct": common["min_stratum_retention"] >= 0.90,
            "substitution_fraction_le_5pct": burden <= 0.05,
            "max_missing_fraction_le_25pct": maximum / 32 <= 0.25,
            "has_consecutive_run_guard": max_run is not None,
            "substitution_count_histogram": json.dumps(dict(sorted(Counter(fill_counts).items())), separators=(",", ":")),
        }
        context_rows.append(row)
    for candidate in BEHAVIOR_CANDIDATES:
        name, _, _ = candidate
        flags = {state.video_id: behavior_eligible(state, candidate) for state in states}
        behavior_flags[name] = flags
        retained = [state for state in states if flags[state.video_id]]
        strata = _stratum_rows(states, "behavior", name, flags)
        stratum_rows.extend(strata)
        valid_counts = [state.behavior_valid_count for state in retained]
        behavior_rows.append({
            "candidate": name, **_impact_common(states, retained, strata),
            "mean_valid_frames": statistics.mean(valid_counts) if valid_counts else 0.0,
            "median_valid_frames": statistics.median(valid_counts) if valid_counts else 0.0,
            "p10_valid_frames": _percentile(valid_counts, 10),
            "max_missing_count_retained": max((state.behavior_missing_count for state in retained), default=0),
            "max_missing_run_retained": max((state.longest_behavior_missing_run for state in retained), default=0),
            "handling_simulation": "NaN_AND_VALID_MASK_ONLY_NO_INTERPOLATION",
        })
    edge_rows = []
    for state in states:
        row: dict[str, Any] = {"video_id": state.video_id, "split": state.split, "label": state.label,
                               "context_missing_count": state.context_missing_count,
                               "longest_context_missing_run": state.longest_context_missing_run}
        for candidate in CONTEXT_CANDIDATES:
            short_name = candidate[0][:2]
            row[f"{short_name}_eligible"] = context_flags[candidate[0]][state.video_id]
            row[f"reason_excluded_{short_name}"] = _edge_reason(state, candidate)
        edge_rows.append(row)
    n246 = next((row for row in edge_rows if row["video_id"] == "n_246"), None)
    if n246 is None or n246["context_missing_count"] != 32 or any(n246[f"{name[:2]}_eligible"] for name, _, _ in CONTEXT_CANDIDATES):
        raise ValueError("n_246은 Context 32/32 결측이며 C0~C4 모두 불가해야 합니다")
    cross_rows = []
    for context_name in [candidate[0] for candidate in CONTEXT_CANDIDATES[:4]]:
        for behavior_name in [candidate[0] for candidate in BEHAVIOR_CANDIDATES[:3]]:
            counts = Counter((context_flags[context_name][state.video_id],
                              behavior_flags[behavior_name][state.video_id]) for state in states)
            for context_value in (True, False):
                for behavior_value in (True, False):
                    cross_rows.append({"context_candidate": context_name, "behavior_candidate": behavior_name,
                                       "context_eligible": context_value, "behavior_eligible": behavior_value,
                                       "videos": counts[(context_value, behavior_value)]})
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STEP_4C1_POLICY_CANDIDATE_IMPACT_ANALYSIS_COMPLETE",
        "input_videos": 1763, "train_videos": 1452, "val_videos": 311, "test_rows": 0,
        "canonical_policy_hash": POLICY_HASH,
        "context_candidates": context_rows, "behavior_candidates": behavior_rows,
        "manual_review_used_as_qualitative_evidence": True,
        "manual_review_used_for_population_rate_estimation": False,
        "canonical_artifacts_modified": False, "raw_video_opened": False,
        "detector_rerun": False, "landmark_rerun": False,
        "policy_selected": False, "step4c2_required": True,
        "n_246_all_context_candidates_ineligible": True,
    }
    return {"context_rows": context_rows, "behavior_rows": behavior_rows,
            "edge_rows": edge_rows, "stratum_rows": stratum_rows,
            "cross_rows": cross_rows, "summary": summary}


def _plot_bars(path: Path, names: Sequence[str], values: Sequence[float], title: str,
               y_label: str, color: tuple[int, int, int]) -> None:
    """추가 그래프 의존성 없이 후보별 비교 PNG만 그린다."""
    width, height = 1150, 620
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (45, 50), font, 0.95, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (45, 87), font, 0.56, (70, 70, 70), 1, cv2.LINE_AA)
    top, bottom = 125, 500
    x0, x1 = 65, width - 40
    ceiling = max(1.0, max(values, default=0.0) * 1.08)
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - round((bottom - top) * fraction)
        cv2.line(canvas, (x0, y), (x1, y), (218, 218, 218), 1)
        cv2.putText(canvas, f"{ceiling * fraction:.1f}", (5, y + 4), font, 0.42, (80, 80, 80), 1)
    step = (x1 - x0) / len(names)
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        center = round(x0 + step * (index + 0.5))
        bar_width = round(step * 0.58)
        y = bottom - round((bottom - top) * value / ceiling)
        cv2.rectangle(canvas, (center - bar_width // 2, y), (center + bar_width // 2, bottom), color, -1)
        cv2.putText(canvas, f"{value:.2f}", (center - 35, max(115, y - 12)), font, 0.52, (25, 25, 25), 1)
        cv2.putText(canvas, name.split("_")[0], (center - 19, 530), font, 0.64, (35, 35, 35), 1)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"plot PNG 저장 실패: {path}")


def _report(result: Mapping[str, Any]) -> str:
    context = result["context_rows"]
    behavior = result["behavior_rows"]
    context_lines = [
        f"{row['candidate']}: eligible {row['eligible_videos']}/1763 ({row['retention_rate']:.2%}), "
        f"min stratum {row['min_stratum_retention']:.2%}, fill videos {row['videos_requiring_fill']}, "
        f"substituted slots {row['total_substituted_slots']} ({row['substituted_slot_fraction']:.2%}), "
        f"longest retained gap {row['max_missing_run_retained']} slots, "
        f"max/p95 timestamp span {row['max_missing_time_span_sec']:.3f}/{row['p95_missing_time_span_sec']:.3f}s"
        for row in context
    ]
    behavior_lines = [
        f"{row['candidate']}: eligible {row['eligible_videos']}/1763 ({row['retention_rate']:.2%}), "
        f"min stratum {row['min_stratum_retention']:.2%}, max retained missing/run "
        f"{row['max_missing_count_retained']}/{row['max_missing_run_retained']}"
        for row in behavior
    ]
    return "\n".join([
        "STEP 4-C1 Missing Policy Candidate Impact Analysis", "",
        "[Scope]", "1763개 train/val canonical 영상에 후보 자격만 시뮬레이션했다. 실제 frame 보완·영상 제외·학습은 없다. Test 0행.",
        "", "[Inputs]", "STEP 4-A per_video_quality·missing_runs·cumulative·split_label 및 canonical metadata. STEP 4-B 수동 결과는 정성 근거로만 사용.",
        "", "[Why Count And Run Length Are Separate]", "동일 결측 수라도 고립 결측과 연속 공백은 시간 정보를 다르게 훼손한다. 36개 목적 선정 시각 검토에서 두 유형 모두 설명 가능한 miss·false negative가 관찰됐다.",
        "", "[Context Candidate Definitions]", "C0: count=0/no fill; C1: count<=2, run<=1; C2: count<=4, run<=2; C3: count<=8, run<=4; C4: count<=8, run guard 없음(진단용). C1~C4는 같은 영상의 가장 가까운 기존 valid slot 복제를 가정하며 동거리는 이전 slot 우선.",
        "", "[Context Candidate Impact]", *context_lines,
        "", "[Split / Label Retention]", "각 후보의 train/val·drowsy/not-drowsy·네 split×label retention은 policy_candidate_stratum_impact.csv에 있다. 모든 집단에 동일 기준을 적용했다.",
        "", "[Distribution Shift]", "context_policy_candidate_impact.csv와 behavior_policy_candidate_impact.csv의 label_ratio_shift_pp·split_ratio_shift_pp는 원본 대비 retained 구성 차이(percentage point)다. 원인을 뜻하지 않는다.",
        "", "[Context Imputation Burden]", "total_substituted_slots는 실제 생성량이 아닌 후보 적용 시 필요한 복제 slot 수다. 원래 valid crop 수와 대체 비율·영상별 대체 수 히스토그램을 함께 기록했다. C4는 연속 결측 보호가 없는 비교용이다.",
        "", "[Context Temporal Gap Impact]", "시간 길이는 연속 결측 Context slot의 첫·끝 실제 canonical timestamp 차이로 정의했다. 단일 missing slot의 span은 0초이며 안전성을 보장하지 않는다. max/median/p90/p95를 계산했다.",
        "", "[Behavior Candidate Definitions]", "B0: valid>=1; B1: valid>=90 및 YuNet run<=10; B2: valid>=95 및 run<=5; B3: valid>=80 및 run<=20(완화 비교용). 모두 NaN+valid mask 유지, interpolation 없음.",
        "", "[Behavior Candidate Impact]", *behavior_lines,
        "", "[Why Behavior Imputation Is Avoided]", "눈 감김·하품·head movement의 실제 시간 변화가 인공 보간으로 생성/삭제될 수 있다. 이는 STEP 4-B 정성 근거와 설계상 위험 설명이지 모델 성능의 정량 근거는 아니다.",
        "", "[Branch Eligibility]", "Context/Behavior 적격은 별개이며 global drop 기준으로 합치지 않았다. 대표 후보 조합의 4분할 교차 집계만 branch_eligibility_cross_summary.csv에 제공한다.",
        "", "[STEP 4-B Qualitative Evidence]", "선정 36개에서 expected miss 13, likely false negative 12, mixed 5, normal reference 6. 목적 선정 사례이므로 비율을 전체 데이터로 일반화하지 않는다. threshold 산출에는 사용하지 않았다.",
        "", "[Limitations]", "모델 정확도 비교가 아니며 nearest-valid 복제가 정확도를 개선한다는 증거도 없다. Behavior event threshold도 미확정이다. 영상별 eligibility는 시뮬레이션일 뿐 실제 데이터는 불변이다.",
        "", "[Trade-Off Summary]", "C0는 대체가 없지만 제외가 가장 많다. C1/C2/C3는 허용 count·run이 커지며 retention과 복제 부담이 함께 변한다. C4는 C3와 count 상한이 같으나 run 제한이 없다. B0/B1/B2/B3는 coverage·연속 결측 제약의 서로 다른 보존량을 보여준다. 한 후보를 우승자로 선정하지 않았다.",
        "", "[Decision Pending]", "STEP 4-C2에서는 전체·최저 strata retention, 구성 비율 변화, 대체 부담, 장기 gap 통제, mask·원본 provenance 보존을 함께 판단해야 한다. CONTEXT POLICY: NOT SELECTED. BEHAVIOR POLICY: NOT SELECTED. STEP 4-C2: PENDING.", "",
    ])


def write_analysis(result: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """기존 artifact를 덮어쓰지 않고 새 결과 디렉터리를 발행한다."""
    if output_dir.exists():
        raise FileExistsError(f"기존 STEP 4-C1 결과를 덮어쓰지 않습니다: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step4c1_staging_", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        _write_csv(stage / "context_policy_candidate_impact.csv", CONTEXT_IMPACT_COLUMNS, result["context_rows"])
        _write_csv(stage / "behavior_policy_candidate_impact.csv", BEHAVIOR_IMPACT_COLUMNS, result["behavior_rows"])
        _write_csv(stage / "context_policy_edge_cases.csv", EDGE_COLUMNS, result["edge_rows"])
        _write_csv(stage / "branch_eligibility_cross_summary.csv", CROSS_COLUMNS, result["cross_rows"])
        _write_csv(stage / "policy_candidate_stratum_impact.csv", STRATUM_COLUMNS, result["stratum_rows"])
        (stage / "step4c_policy_analysis_summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "step4c_policy_analysis_report.txt").write_text(_report(result), encoding="utf-8")
        plots = stage / "plots"
        plots.mkdir()
        context_rows = result["context_rows"]
        behavior_rows = result["behavior_rows"]
        _plot_bars(plots / "context_policy_retention.png", [row["candidate"] for row in context_rows],
                   [100 * row["retention_rate"] for row in context_rows], "Context candidate overall retention", "Percent of 1763 train/val videos", (170, 105, 45))
        _plot_bars(plots / "context_policy_stratum_retention.png", [row["candidate"] for row in context_rows],
                   [100 * row["min_stratum_retention"] for row in context_rows], "Context minimum split-label retention", "Percent in least-retained stratum", (125, 135, 68))
        _plot_bars(plots / "context_policy_imputation_burden.png", [row["candidate"] for row in context_rows],
                   [100 * row["substituted_slot_fraction"] for row in context_rows], "Context simulated substitution burden", "Percent of retained 32-slot sequences", (70, 125, 175))
        _plot_bars(plots / "behavior_policy_retention.png", [row["candidate"] for row in behavior_rows],
                   [100 * row["retention_rate"] for row in behavior_rows], "Behavior candidate overall retention", "Percent of 1763 train/val videos", (150, 92, 130))
        stage.rename(output_dir)
    return result["summary"]


def dry_run(states: Sequence[VideoState], canonical_root: Path, audit_root: Path, output_dir: Path) -> dict[str, Any]:
    """후보 영향 계산·원본 영상 접근·출력 없이 입력 수와 예정 파일만 보인다."""
    return {
        "mode": "DRY_RUN", "canonical_root": str(canonical_root), "audit_root": str(audit_root),
        "output_dir": str(output_dir), "input_videos": len(states),
        "split_counts": dict(Counter(state.split for state in states)), "test_rows": 0,
        "context_candidates": [{"name": name, "max_missing_count": count, "max_run": run} for name, count, run in CONTEXT_CANDIDATES],
        "behavior_candidates": [{"name": name, "min_valid_frames": valid, "max_run": run} for name, valid, run in BEHAVIOR_CANDIDATES],
        "planned_outputs": ["context_policy_candidate_impact.csv", "behavior_policy_candidate_impact.csv",
                            "context_policy_edge_cases.csv", "branch_eligibility_cross_summary.csv",
                            "policy_candidate_stratum_impact.csv", "step4c_policy_analysis_summary.json",
                            "step4c_policy_analysis_report.txt", "plots/*.png"],
        "actual_analysis": False, "raw_video_opened": False, "output_written": False,
    }
