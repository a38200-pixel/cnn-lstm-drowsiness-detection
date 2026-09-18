"""train/validation에서 라벨 균형을 고려한 audit frame을 선택한다."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AuditSample:
    """재현 가능한 frame audit 대상 하나."""

    sample_id: str
    video_id: str
    split: str
    label: str
    video_path: Path
    timestamp_sec: float


def load_split_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    """train 또는 validation metadata를 읽고 test 유입을 차단한다."""

    if expected_split not in {"train", "val"}:
        raise ValueError("STEP 2에서는 train과 val split만 사용할 수 있습니다")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "video_path", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"필수 metadata 컬럼이 없습니다: {path}")
        rows = [dict(row) for row in reader]
    for row in rows:
        declared = row.get("split", expected_split)
        if declared and declared != expected_split:
            raise ValueError(f"{path}에 예상하지 않은 split이 있습니다: {declared}")
    return rows


def _resolve_video_path(value: str, project_root: Path) -> Path:
    """metadata의 영상 경로를 프로젝트 루트 기준으로 해석한다."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _valid_timestamps(row: dict[str, str], timestamps: Sequence[float]) -> list[float]:
    """metadata duration 안에 있는 timestamp만 선택한다."""

    try:
        duration = float(row.get("duration_seconds", ""))
    except ValueError:
        duration = 0.0
    if duration <= 0:
        return [float(value) for value in timestamps]
    return [float(value) for value in timestamps if 0.0 <= float(value) < duration]


def build_audit_candidates(
    train_rows: Sequence[dict[str, str]],
    val_rows: Sequence[dict[str, str]],
    project_root: Path,
    timestamps: Sequence[float],
    random_seed: int,
) -> list[AuditSample]:
    """
    split/label strata를 round-robin으로 섞어 여러 영상과 시점을 고르게 선택한다.

    라벨은 이 함수의 sampling 균형과 report 표시에만 사용되며 이후 detector,
    landmark, geometry 함수에는 전달되지 않는다.
    """

    random_generator = random.Random(random_seed)
    strata: dict[tuple[str, str], list[AuditSample]] = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        by_label: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_label.setdefault(row["label"], []).append(row)
        for label, label_rows in sorted(by_label.items()):
            shuffled = list(label_rows)
            random_generator.shuffle(shuffled)
            samples: list[AuditSample] = []
            for video_order, row in enumerate(shuffled):
                valid_timestamps = _valid_timestamps(row, timestamps)
                if not valid_timestamps:
                    continue
                # 영상마다 시작 timestamp를 회전해 특정 시점에 표본이 몰리지 않게 한다.
                rotated = valid_timestamps[video_order % len(valid_timestamps) :] + valid_timestamps[: video_order % len(valid_timestamps)]
                for timestamp in rotated:
                    token = str(timestamp).replace(".", "p")
                    samples.append(
                        AuditSample(
                            sample_id=f"{split}_{row['video_id']}_t{token}",
                            video_id=row["video_id"],
                            split=split,
                            label=label,
                            video_path=_resolve_video_path(row["video_path"], project_root),
                            timestamp_sec=timestamp,
                        )
                    )
            strata[(split, label)] = samples

    ordered: list[AuditSample] = []
    keys = sorted(strata)
    index = 0
    while True:
        added = False
        for key in keys:
            if index < len(strata[key]):
                ordered.append(strata[key][index])
                added = True
        if not added:
            break
        index += 1
    return ordered


def take_unique_samples(
    candidates: Iterable[AuditSample], maximum: int
) -> list[AuditSample]:
    """동일 sample ID를 제거하고 최대 개수까지 반환한다."""

    selected: list[AuditSample] = []
    seen: set[str] = set()
    for sample in candidates:
        if sample.sample_id in seen:
            continue
        seen.add(sample.sample_id)
        selected.append(sample)
        if len(selected) >= maximum:
            break
    return selected

