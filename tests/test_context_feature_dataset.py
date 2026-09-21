"""STEP 6-A Context frozen feature Dataset/DataLoader 테스트."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from drowsiness_detection.data_v2.context_feature_dataset import (
    LABEL_TO_INDEX,
    ContextFeatureDataset,
    ContextFeatureDatasetError,
    build_context_feature_dataloader,
    context_only_video_ids,
    same_universe_and_order,
)


INDEX_COLUMNS = (
    "feature_video_index", "video_id", "split", "label", "imputed_count",
    "context_sequence_path",
)


def _write_fixture(root: Path, backbone: str = "resnet18", *,
                   bad_shape: bool = False, bad_dtype: bool = False,
                   nonfinite: bool = False, mask_count: int = 1) -> Path:
    backbone_root = root / backbone
    rows = [
        (0, "n_1", "train", "not_drowsy", mask_count),
        (1, "d_1", "train", "drowsy", 0),
        (2, "d_2", "val", "drowsy", 0),
    ]
    backbone_root.mkdir(parents=True)
    with (backbone_root / "full_feature_videos.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for index, video_id, split, label, imputed_count in rows:
            writer.writerow({
                "feature_video_index": index,
                "video_id": video_id,
                "split": split,
                "label": label,
                "imputed_count": imputed_count,
                "context_sequence_path": f"data/interim/sequences_v2/context/per_video/{split}/{video_id}/sequence.csv",
            })
            bundle = backbone_root / "per_video" / split / video_id
            bundle.mkdir(parents=True)
            shape = (31, 512) if bad_shape and index == 0 else (32, 512)
            dtype = np.float64 if bad_dtype and index == 0 else np.float32
            features = np.full(shape, index + 0.25, dtype=dtype)
            if nonfinite and index == 0:
                features[0, 0] = np.nan
            np.save(bundle / "features.npy", features, allow_pickle=False)
            mask = np.zeros(32, dtype=np.bool_)
            if index == 0:
                mask[:mask_count] = True
            np.savez_compressed(bundle / "masks.npz", context_imputed_mask=mask)
    return root


def _dataset(tmp_path: Path, split: str = "train", **kwargs: object) -> ContextFeatureDataset:
    artifact_root = _write_fixture(tmp_path / "features", **kwargs)
    return ContextFeatureDataset(tmp_path, "resnet18", split, artifact_root=artifact_root)


def test_sample_contract_and_label_mapping(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    assert set(sample) == {"features", "label", "video_id", "imputed_mask"}
    assert sample["features"].shape == (32, 512)
    assert sample["features"].dtype == torch.float32
    assert sample["label"].shape == ()
    assert sample["label"].dtype == torch.long
    assert sample["label"].item() == LABEL_TO_INDEX["not_drowsy"] == 0
    assert dataset[1]["label"].item() == LABEL_TO_INDEX["drowsy"] == 1
    assert sample["video_id"] == "n_1"
    assert sample["imputed_mask"].shape == (32,)
    assert sample["imputed_mask"].dtype == torch.bool


def test_feature_values_are_returned_without_normalization(tmp_path: Path) -> None:
    sample = _dataset(tmp_path)[0]
    assert torch.all(sample["features"] == torch.tensor(0.25, dtype=torch.float32))
    assert sample["features"].shape[-1] == 512


def test_dataset_order_is_index_order(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    assert [dataset[index]["video_id"] for index in range(len(dataset))] == ["n_1", "d_1"]


def test_utf8_bom_index_is_supported(tmp_path: Path) -> None:
    artifact_root = _write_fixture(tmp_path / "features")
    index = artifact_root / "resnet18/full_feature_videos.csv"
    content = index.read_text(encoding="utf-8")
    index.write_text(content, encoding="utf-8-sig")
    assert len(ContextFeatureDataset(
        tmp_path, "resnet18", "train", artifact_root=artifact_root)) == 2


def test_train_loader_shuffles_and_val_loader_does_not(tmp_path: Path) -> None:
    artifact_root = _write_fixture(tmp_path / "features")
    train = ContextFeatureDataset(tmp_path, "resnet18", "train", artifact_root=artifact_root)
    val = ContextFeatureDataset(tmp_path, "resnet18", "val", artifact_root=artifact_root)
    assert isinstance(build_context_feature_dataloader(train, 2).sampler, RandomSampler)
    assert isinstance(build_context_feature_dataloader(val, 2).sampler, SequentialSampler)
    assert build_context_feature_dataloader(train, 2).num_workers == 0


def test_batch_contract(tmp_path: Path) -> None:
    batch = next(iter(build_context_feature_dataloader(_dataset(tmp_path), 2)))
    assert batch["features"].shape == (2, 32, 512)
    assert batch["features"].dtype == torch.float32
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.long
    assert batch["imputed_mask"].shape == (2, 32)


@pytest.mark.parametrize("split", ["test", "dev", ""])
def test_test_and_unknown_splits_are_blocked(tmp_path: Path, split: str) -> None:
    artifact_root = _write_fixture(tmp_path / "features")
    with pytest.raises(ContextFeatureDatasetError):
        ContextFeatureDataset(tmp_path, "resnet18", split, artifact_root=artifact_root)


def test_unknown_backbone_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(ContextFeatureDatasetError):
        ContextFeatureDataset(tmp_path, "mobilenet", "train", artifact_root=tmp_path)


def test_test_row_in_index_is_blocked_before_bundle_access(tmp_path: Path) -> None:
    artifact_root = _write_fixture(tmp_path / "features")
    index = artifact_root / "resnet18/full_feature_videos.csv"
    text = index.read_text(encoding="utf-8").replace(
        "2,d_2,val,drowsy,0,", "2,d_2,test,drowsy,0,")
    index.write_text(text, encoding="utf-8")
    with pytest.raises(ContextFeatureDatasetError, match="test split"):
        ContextFeatureDataset(tmp_path, "resnet18", "train", artifact_root=artifact_root)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bad_shape": True}, "shape"),
        ({"bad_dtype": True}, "dtype"),
        ({"nonfinite": True}, "NaN/Inf"),
        ({"mask_count": 2}, "imputed_count"),
    ],
)
def test_invalid_bundle_contract_is_rejected(
        tmp_path: Path, kwargs: dict[str, object], message: str) -> None:
    dataset = _dataset(tmp_path, **kwargs)
    if kwargs == {"mask_count": 2}:
        # Index count만 변경해 mask와 불일치시키기 위한 별도 조작이다.
        dataset.records = (dataset.records[0].__class__(0, "n_1", "train", "not_drowsy", 1),
                           dataset.records[1])
    with pytest.raises(ContextFeatureDatasetError, match=message):
        dataset[0]


def test_backbone_universe_and_order_comparison(tmp_path: Path) -> None:
    artifact_root = _write_fixture(tmp_path / "features", "resnet18")
    _write_fixture(artifact_root, "vgg16")
    pairs = []
    for split in ("train", "val"):
        pairs.append((
            ContextFeatureDataset(tmp_path, "resnet18", split, artifact_root=artifact_root),
            ContextFeatureDataset(tmp_path, "vgg16", split, artifact_root=artifact_root),
        ))
    assert all(same_universe_and_order(pair) for pair in pairs)


def test_context_only_is_diagnostic_not_filter(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = tmp_path / "eligibility.csv"
    manifest.write_text(
        "video_id,split,context_eligible,behavior_eligible\n"
        "n_1,train,True,False\n"
        "d_1,train,True,True\n",
        encoding="utf-8",
    )
    assert len(dataset) == 2
    assert context_only_video_ids((dataset,), manifest) == ("n_1",)


def test_module_has_no_cnn_or_jpeg_dependency() -> None:
    source = Path(
        "src/drowsiness_detection/data_v2/context_feature_dataset.py"
    ).read_text(encoding="utf-8")
    assert "torchvision" not in source
    assert "cnn_feature_extractor" not in source
    assert ".jpg" not in source.casefold()
