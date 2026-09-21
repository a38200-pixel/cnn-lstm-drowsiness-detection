"""동결된 STEP 5-D Context CNN feature를 읽는 PyTorch Dataset/DataLoader."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


BACKBONES = ("resnet18", "vgg16")
SPLITS = ("train", "val")
LABEL_TO_INDEX = {"not_drowsy": 0, "drowsy": 1}
EXPECTED_SHAPE = (32, 512)
INDEX_FILENAME = "full_feature_videos.csv"
REQUIRED_INDEX_COLUMNS = {
    "feature_video_index",
    "video_id",
    "split",
    "label",
    "imputed_count",
    "context_sequence_path",
}


class ContextFeatureDatasetError(ValueError):
    """Artifact 계약 또는 leakage 보호 규칙 위반."""


@dataclass(frozen=True)
class ContextFeatureRecord:
    """Index에서 읽은 단일 Context feature sample의 metadata."""

    feature_video_index: int
    video_id: str
    split: str
    label: str
    imputed_count: int


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.is_file():
        raise ContextFeatureDatasetError(f"필수 index가 없습니다: {path}")
    # STEP 5-D CSV는 Windows Excel 호환 UTF-8 BOM을 포함할 수 있다.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return rows, columns


def _safe_video_id(value: str) -> str:
    video_id = value.strip()
    if not video_id or Path(video_id).name != video_id or "/" in video_id or "\\" in video_id:
        raise ContextFeatureDatasetError(f"안전하지 않은 video_id입니다: {value!r}")
    return video_id


def _parse_index(index_path: Path) -> list[ContextFeatureRecord]:
    rows, columns = _read_csv(index_path)
    missing = REQUIRED_INDEX_COLUMNS.difference(columns)
    if missing:
        raise ContextFeatureDatasetError(f"index column이 누락됐습니다: {sorted(missing)}")
    if not rows:
        raise ContextFeatureDatasetError("feature index가 비어 있습니다")

    records: list[ContextFeatureRecord] = []
    seen_video_ids: set[str] = set()
    for position, row in enumerate(rows):
        split = row["split"].strip()
        if split == "test":
            raise ContextFeatureDatasetError("test split artifact 접근은 차단됩니다")
        if split not in SPLITS:
            raise ContextFeatureDatasetError(f"지원하지 않는 split입니다: {split!r}")
        if "test" in Path(row["context_sequence_path"]).parts:
            raise ContextFeatureDatasetError("test path가 feature index에 포함됐습니다")

        video_id = _safe_video_id(row["video_id"])
        if video_id in seen_video_ids:
            raise ContextFeatureDatasetError(f"중복 video_id입니다: {video_id}")
        seen_video_ids.add(video_id)

        label = row["label"].strip()
        if label not in LABEL_TO_INDEX:
            raise ContextFeatureDatasetError(f"지원하지 않는 label입니다: {label!r}")
        try:
            feature_video_index = int(row["feature_video_index"])
            imputed_count = int(row["imputed_count"])
        except ValueError as error:
            raise ContextFeatureDatasetError("index의 정수 field가 잘못됐습니다") from error
        if feature_video_index != position:
            raise ContextFeatureDatasetError(
                "feature_video_index가 결정적 연속 순서와 일치하지 않습니다")
        if not 0 <= imputed_count <= EXPECTED_SHAPE[0]:
            raise ContextFeatureDatasetError(f"imputed_count가 잘못됐습니다: {imputed_count}")
        records.append(ContextFeatureRecord(
            feature_video_index=feature_video_index,
            video_id=video_id,
            split=split,
            label=label,
            imputed_count=imputed_count,
        ))
    return records


class ContextFeatureDataset(Dataset[dict[str, Any]]):
    """추가 변환 없이 per-video `[32,512]` frozen feature를 반환한다."""

    def __init__(
        self,
        project_root: Path | str,
        backbone: str,
        split: str,
        *,
        artifact_root: Path | str | None = None,
    ) -> None:
        if backbone not in BACKBONES:
            raise ContextFeatureDatasetError(
                f"backbone은 {BACKBONES} 중 하나여야 합니다: {backbone!r}")
        if split == "test":
            raise ContextFeatureDatasetError("test split Dataset 생성은 차단됩니다")
        if split not in SPLITS:
            raise ContextFeatureDatasetError(
                f"split은 {SPLITS} 중 하나여야 합니다: {split!r}")

        self.project_root = Path(project_root).resolve()
        base = (Path(artifact_root).resolve() if artifact_root is not None else
                self.project_root / "data/interim/features_v2/context_full")
        self.backbone = backbone
        self.split = split
        self.backbone_root = base / backbone
        all_records = _parse_index(self.backbone_root / INDEX_FILENAME)
        self.records = tuple(record for record in all_records if record.split == split)
        if not self.records:
            raise ContextFeatureDatasetError(f"{backbone}/{split} sample이 없습니다")

    def __len__(self) -> int:
        return len(self.records)

    def _bundle(self, record: ContextFeatureRecord) -> Path:
        return self.backbone_root / "per_video" / record.split / record.video_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        bundle = self._bundle(record)
        feature_path = bundle / "features.npy"
        mask_path = bundle / "masks.npz"
        if not feature_path.is_file() or not mask_path.is_file():
            raise ContextFeatureDatasetError(f"bundle artifact가 누락됐습니다: {bundle}")

        features = np.load(feature_path, allow_pickle=False)
        if features.shape != EXPECTED_SHAPE:
            raise ContextFeatureDatasetError(
                f"feature shape가 {EXPECTED_SHAPE}가 아닙니다: {features.shape}")
        if features.dtype != np.float32:
            raise ContextFeatureDatasetError(
                f"feature dtype이 float32가 아닙니다: {features.dtype}")
        if not np.isfinite(features).all():
            raise ContextFeatureDatasetError(f"NaN/Inf feature가 있습니다: {record.video_id}")

        with np.load(mask_path, allow_pickle=False) as masks:
            if "context_imputed_mask" not in masks.files:
                raise ContextFeatureDatasetError(
                    f"context_imputed_mask가 없습니다: {record.video_id}")
            imputed_mask = np.asarray(masks["context_imputed_mask"])
        if imputed_mask.shape != (EXPECTED_SHAPE[0],) or imputed_mask.dtype != np.bool_:
            raise ContextFeatureDatasetError(
                f"imputed mask 계약이 잘못됐습니다: {record.video_id}")
        if int(np.count_nonzero(imputed_mask)) != record.imputed_count:
            raise ContextFeatureDatasetError(
                f"imputed_count가 mask와 일치하지 않습니다: {record.video_id}")

        # np.load가 만든 메모리 배열만 tensor와 공유하며 원본 .npy는 수정하지 않는다.
        return {
            "features": torch.from_numpy(features),
            "label": torch.tensor(LABEL_TO_INDEX[record.label], dtype=torch.long),
            "video_id": record.video_id,
            "imputed_mask": torch.from_numpy(imputed_mask.copy()),
        }

    @property
    def ordered_signature(self) -> tuple[tuple[str, str, int], ...]:
        """Backbone 간 universe/순서/label 비교용 결정적 signature."""

        return tuple(
            (record.video_id, record.split, LABEL_TO_INDEX[record.label])
            for record in self.records
        )

    @property
    def label_counts(self) -> Mapping[str, int]:
        return dict(Counter(record.label for record in self.records))


def build_context_feature_dataloader(
    dataset: ContextFeatureDataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, Any]]:
    """Train은 shuffle, validation은 고정 순서로 DataLoader를 만든다."""

    if batch_size <= 0:
        raise ContextFeatureDatasetError("batch_size는 양수여야 합니다")
    if num_workers < 0:
        raise ContextFeatureDatasetError("num_workers는 0 이상이어야 합니다")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.split == "train",
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )


def same_universe_and_order(datasets: Sequence[ContextFeatureDataset]) -> bool:
    """두 개 이상의 Dataset이 video/split/label 순서를 공유하는지 확인한다."""

    if len(datasets) < 2:
        raise ContextFeatureDatasetError("비교할 Dataset이 두 개 이상 필요합니다")
    reference = datasets[0].ordered_signature
    return all(dataset.ordered_signature == reference for dataset in datasets[1:])


def context_only_video_ids(
    datasets: Sequence[ContextFeatureDataset],
    eligibility_manifest: Path | str,
) -> tuple[str, ...]:
    """Dataset을 필터링하지 않고 frozen manifest로 Context-only 포함만 진단한다."""

    rows, columns = _read_csv(Path(eligibility_manifest))
    required = {"video_id", "split", "context_eligible", "behavior_eligible"}
    missing = required.difference(columns)
    if missing:
        raise ContextFeatureDatasetError(
            f"eligibility column이 누락됐습니다: {sorted(missing)}")
    included = {
        (record.video_id, record.split)
        for dataset in datasets
        for record in dataset.records
    }
    result = []
    for row in rows:
        key = (row["video_id"], row["split"])
        if (row["context_eligible"].casefold() == "true"
                and row["behavior_eligible"].casefold() == "false"
                and key in included):
            result.append(row["video_id"])
    return tuple(result)
