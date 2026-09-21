"""STEP 6-C Context LSTM training pipeline 테스트."""

from __future__ import annotations

import builtins
import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from drowsiness_detection.models_v2 import build_context_lstm
from drowsiness_detection.training_v2 import context_lstm_trainer as trainer
from scripts.train_context_lstm import build_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/context_lstm_baseline.yaml"


class TinyContextDataset(Dataset):
    def __init__(self, count: int) -> None:
        generator = torch.Generator().manual_seed(7)
        self.features = torch.randn(count, 32, 512, generator=generator)
        self.labels = torch.tensor([index % 2 for index in range(count)], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "video_id": f"video_{index}",
            "imputed_mask": torch.zeros(32, dtype=torch.bool),
        }


@pytest.fixture(scope="module")
def config_mapping() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def training_config(config_mapping: dict) -> trainer.ContextTrainingConfig:
    return trainer.training_config_from_mapping(config_mapping)


def test_training_config_matches_step6c_policy(
        training_config: trainer.ContextTrainingConfig) -> None:
    assert training_config.train_batch_size == 16
    assert training_config.val_batch_size == 32
    assert training_config.max_epochs == 60
    assert training_config.seeds == (42, 123, 2026)
    assert training_config.learning_rate == 0.0005
    assert training_config.weight_decay == 0.0001
    assert training_config.gradient_clip_max_norm == 1.0
    assert training_config.scheduler_factor == 0.5
    assert training_config.scheduler_patience == 4
    assert training_config.scheduler_min_lr == 0.000001
    assert training_config.early_stopping_patience == 10
    assert training_config.early_stopping_min_delta == 0.0001
    assert training_config.mlflow_experiment_name == "context_lstm_baseline_v1"


def test_batch8_override_and_tag_resolve_without_changing_baseline() -> None:
    assert trainer.resolve_train_batch_size(16, 8, "batch8") == 8
    assert trainer.resolve_run_names("resnet18", 42, "batch8") == (
        "seed_42_batch8", "resnet18_seed42_batch8")
    assert trainer.resolve_run_names("vgg16", 42, "batch8") == (
        "seed_42_batch8", "vgg16_seed42_batch8")
    assert trainer.resolve_train_batch_size(16, None, None) == 16
    assert trainer.resolve_run_names("resnet18", 42, None) == (
        "seed_42", "resnet18_seed42")


def test_changed_batch_size_requires_run_tag() -> None:
    with pytest.raises(trainer.TrainingPipelineError, match="run-tag"):
        trainer.resolve_train_batch_size(16, 8, None)


def test_classifier_dropout_override_requires_tag_and_keeps_batch16() -> None:
    assert trainer.resolve_classifier_dropout(0.0, 0.2, "dropout02") == 0.2
    assert trainer.resolve_train_batch_size(16, None, "dropout02") == 16
    assert trainer.resolve_run_names("resnet18", 42, "dropout02") == (
        "seed_42_dropout02", "resnet18_seed42_dropout02")
    assert trainer.resolve_run_names("vgg16", 42, "dropout02") == (
        "seed_42_dropout02", "vgg16_seed42_dropout02")
    with pytest.raises(trainer.TrainingPipelineError, match="run-tag"):
        trainer.resolve_classifier_dropout(0.0, 0.2, None)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_invalid_classifier_dropout_is_rejected(dropout: float) -> None:
    with pytest.raises(trainer.TrainingPipelineError, match="classifier dropout"):
        trainer.resolve_classifier_dropout(0.0, dropout, "invalid")


@pytest.mark.parametrize("run_tag", ["", "../batch8", "batch 8", "batch/8"])
def test_unsafe_run_tag_is_rejected(run_tag: str) -> None:
    with pytest.raises(trainer.TrainingPipelineError, match="run tag"):
        trainer.resolve_run_names("resnet18", 42, run_tag)


def test_cli_accepts_batch8_tuning_options() -> None:
    args = build_parser().parse_args([
        "--backbone", "vgg16", "--seed", "42", "--device", "cuda",
        "--mlflow", "--train-batch-size", "8", "--run-tag", "batch8",
    ])
    assert args.backbone == "vgg16"
    assert args.seed == 42
    assert args.device == "cuda"
    assert args.mlflow is True
    assert args.train_batch_size == 8
    assert args.run_tag == "batch8"


def test_cli_accepts_classifier_dropout_tuning_options() -> None:
    args = build_parser().parse_args([
        "--backbone", "resnet18", "--seed", "42", "--device", "cuda",
        "--mlflow", "--classifier-dropout", "0.2", "--run-tag", "dropout02",
    ])
    assert args.train_batch_size is None
    assert args.classifier_dropout == 0.2
    assert args.run_tag == "dropout02"


def test_classification_metrics() -> None:
    result = trainer.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["val_accuracy"] == 0.75
    assert result["val_drowsy_precision"] == pytest.approx(2 / 3)
    assert result["val_drowsy_recall"] == 1.0
    assert result["val_drowsy_f1"] == pytest.approx(0.8)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_train_epoch_updates_parameters_and_clips_gradients(
        config_mapping: dict, training_config: trainer.ContextTrainingConfig) -> None:
    torch.manual_seed(3)
    model = build_context_lstm(config_mapping)
    loader = DataLoader(TinyContextDataset(4), batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=0.0)
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate,
                      weight_decay=training_config.weight_decay)
    before = model.classifier[0].weight.detach().clone()
    result = trainer.train_one_epoch(
        model, loader, criterion, optimizer, torch.device("cpu"), 1.0)
    assert result["gradient_clip_steps"] == 2
    assert torch.isfinite(torch.tensor(result["train_loss"]))
    assert not torch.equal(before, model.classifier[0].weight.detach())


def test_evaluate_returns_required_finite_metrics(config_mapping: dict) -> None:
    model = build_context_lstm(config_mapping)
    loader = DataLoader(TinyContextDataset(4), batch_size=4, shuffle=False)
    result = trainer.evaluate(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    expected = {
        "val_loss", "val_accuracy", "val_macro_f1", "val_drowsy_precision",
        "val_drowsy_recall", "val_drowsy_f1", "confusion_matrix",
    }
    assert set(result) == expected
    assert all(torch.isfinite(torch.tensor(result[key])) for key in expected - {"confusion_matrix"})


def test_fit_saves_all_local_artifacts(
        tmp_path: Path, config_mapping: dict,
        training_config: trainer.ContextTrainingConfig) -> None:
    torch.manual_seed(5)
    model = build_context_lstm(config_mapping)
    train_loader = DataLoader(TinyContextDataset(4), batch_size=2, shuffle=True)
    val_loader = DataLoader(TinyContextDataset(4), batch_size=4, shuffle=False)
    logger = trainer.OptionalMLflowLogger(False, "unused", "unused")
    output = tmp_path / "training"
    result = trainer.fit_context_lstm(
        model, train_loader, val_loader, training_config, torch.device("cpu"),
        output, 1, {"backbone": "resnet18", "seed": 42}, config_mapping, logger)

    expected_files = {
        "best_model.pt", "history.csv", "final_metrics.json",
        "training_curves.png", "confusion_matrix.png", "config_snapshot.yaml",
    }
    assert expected_files == {path.name for path in output.iterdir()}
    assert all((output / name).stat().st_size > 0 for name in expected_files)
    with (output / "history.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert tuple(rows[0]) == trainer.HISTORY_COLUMNS
    saved = json.loads((output / "final_metrics.json").read_text(encoding="utf-8"))
    assert saved["gradient_clip_steps"] == 2
    assert saved["scheduler_steps"] == 1
    assert saved["test_access_count"] == 0
    assert saved["mlflow_enabled"] is False
    assert saved["mlflow_run_id"] is None
    assert result == saved
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["test_used"] is False
    assert checkpoint["epoch"] == 1


def test_disabled_mlflow_logger_does_not_import_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow" or name.startswith("mlflow."):
            raise AssertionError("disabled logger imported MLflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    logger = trainer.OptionalMLflowLogger(False, "unused", "unused")
    logger.start("disabled", {})
    logger.log_epoch(1, {"val_loss": 1.0})
    logger.finish(Path("unused"))
    assert logger.run_id is None


def test_nonfinite_loss_is_blocked() -> None:
    with pytest.raises(trainer.TrainingPipelineError, match="NaN/Inf"):
        trainer._require_finite(float("nan"), "loss")


def test_source_has_no_amp_or_test_dataset_access() -> None:
    source = (
        PROJECT_ROOT
        / "src/drowsiness_detection/training_v2/context_lstm_trainer.py"
    ).read_text(encoding="utf-8")
    assert "autocast" not in source
    assert "GradScaler" not in source
    assert 'ContextFeatureDataset(project_root, backbone, "test")' not in source
