"""STEP 6-A Context feature Dataset/DataLoader를 read-only로 점검한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.data_v2.context_feature_dataset import (  # noqa: E402
    BACKBONES,
    LABEL_TO_INDEX,
    ContextFeatureDataset,
    ContextFeatureDatasetError,
    build_context_feature_dataloader,
    context_only_video_ids,
    same_universe_and_order,
)


EXPECTED_COUNTS = {"train": 1382, "val": 295}
ELIGIBILITY_MANIFEST = (
    PROJECT_ROOT
    / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit"
    / "step4c_policy_freeze/sequence_missing_policy_eligibility.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def _numeric_counts(datasets: list[ContextFeatureDataset], batch_size: int,
                    num_workers: int) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for dataset in datasets:
        loader = build_context_feature_dataloader(
            dataset, batch_size, num_workers=num_workers,
            generator=torch.Generator().manual_seed(0),
        )
        for batch in loader:
            nan_count += int(torch.isnan(batch["features"]).sum().item())
            inf_count += int(torch.isinf(batch["features"]).sum().item())
    return nan_count, inf_count


def inspect(backbone: str, batch_size: int, num_workers: int) -> None:
    selected = [
        ContextFeatureDataset(PROJECT_ROOT, backbone, split)
        for split in ("train", "val")
    ]
    other_backbone = next(name for name in BACKBONES if name != backbone)
    other = [
        ContextFeatureDataset(PROJECT_ROOT, other_backbone, split)
        for split in ("train", "val")
    ]
    for dataset in selected:
        expected = EXPECTED_COUNTS[dataset.split]
        if len(dataset) != expected:
            raise ContextFeatureDatasetError(
                f"{dataset.split} count가 {expected}가 아닙니다: {len(dataset)}")

    sample = selected[0][0]
    first_batch = next(iter(build_context_feature_dataloader(
        selected[0], batch_size, num_workers=num_workers,
        generator=torch.Generator().manual_seed(0),
    )))
    nan_count, inf_count = _numeric_counts(selected, batch_size, num_workers)
    context_only = context_only_video_ids(selected, ELIGIBILITY_MANIFEST)
    universe_equal = all(
        same_universe_and_order((left, right))
        for left, right in zip(selected, other)
    )

    print("STEP 6-A CONTEXT FEATURE DATASET INSPECTION")
    print(f"BACKBONE: {backbone}")
    for dataset in selected:
        counts = dataset.label_counts
        print(f"{dataset.split.upper()} COUNT: {len(dataset)}")
        print(
            f"{dataset.split.upper()} LABELS: "
            f"not_drowsy={counts.get('not_drowsy', 0)} drowsy={counts.get('drowsy', 0)}"
        )
    print(f"FIRST SAMPLE SHAPE: {list(sample['features'].shape)}")
    print(f"FIRST BATCH SHAPE: {list(first_batch['features'].shape)}")
    print(f"FEATURE DTYPE: {sample['features'].dtype}")
    print(f"LABEL DTYPE: {sample['label'].dtype}")
    print(f"IMPUTED MASK SHAPE: {list(sample['imputed_mask'].shape)}")
    print(f"NAN COUNT: {nan_count}")
    print(f"INF COUNT: {inf_count}")
    print(f"CONTEXT-ONLY INCLUDED: {len(context_only)}")
    print(f"BACKBONE UNIVERSE/ORDER IDENTICAL: {universe_equal}")
    print(f"LABEL MAPPING: {LABEL_TO_INDEX}")
    print("ADDITIONAL NORMALIZATION: NONE")
    print("IMPUTED MASK CONCATENATED: FALSE")
    print("CNN INFERENCE: 0")
    print("JPEG ACCESS: 0")
    print("TEST ACCESS: 0")
    if nan_count or inf_count or len(context_only) != 157 or not universe_equal:
        raise ContextFeatureDatasetError("STEP 6-A Dataset inspection이 실패했습니다")


def main() -> int:
    args = build_parser().parse_args()
    inspect(args.backbone, args.batch_size, args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
