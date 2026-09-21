"""STEP 6-C Context LSTM train/validation 학습 파이프라인."""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from drowsiness_detection.data_v2 import (
    ContextFeatureDataset,
    build_context_feature_dataloader,
)
from drowsiness_detection.models_v2 import build_context_lstm


HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_drowsy_precision",
    "val_drowsy_recall",
    "val_drowsy_f1",
    "learning_rate",
)


class TrainingPipelineError(RuntimeError):
    """학습 정책 또는 numeric integrity 위반."""


def resolve_train_batch_size(
    configured_batch_size: int,
    override: int | None,
    run_tag: str | None,
) -> int:
    """Run별 batch override를 검증하고 baseline 덮어쓰기를 차단한다."""

    if override is None:
        return configured_batch_size
    if override <= 0:
        raise TrainingPipelineError("train batch size override는 양수여야 합니다")
    if override != configured_batch_size and run_tag is None:
        raise TrainingPipelineError(
            "baseline과 다른 train batch size에는 --run-tag가 필요합니다")
    return override


def resolve_classifier_dropout(
    configured_dropout: float,
    override: float | None,
    run_tag: str | None,
) -> float:
    """Classifier dropout override를 검증하고 baseline 덮어쓰기를 차단한다."""

    if override is None:
        return configured_dropout
    if not 0.0 <= override < 1.0:
        raise TrainingPipelineError("classifier dropout은 0 이상 1 미만이어야 합니다")
    if override != configured_dropout and run_tag is None:
        raise TrainingPipelineError(
            "baseline과 다른 classifier dropout에는 --run-tag가 필요합니다")
    return override


def resolve_weight_decay(
    configured_weight_decay: float,
    override: float | None,
    run_tag: str | None,
) -> float:
    """Run별 weight decay override를 검증하고 baseline 덮어쓰기를 차단한다."""

    if override is None:
        return configured_weight_decay
    if override < 0.0:
        raise TrainingPipelineError("weight decay는 0 이상이어야 합니다")
    if override != configured_weight_decay and run_tag is None:
        raise TrainingPipelineError(
            "baseline과 다른 weight decay에는 --run-tag가 필요합니다")
    return override


def resolve_learning_rate(
    configured_learning_rate: float,
    override: float | None,
    run_tag: str | None,
) -> float:
    """Run별 initial learning rate override를 검증하고 baseline 덮어쓰기를 차단한다."""

    if override is None:
        return configured_learning_rate
    if override <= 0.0:
        raise TrainingPipelineError("learning rate는 양수여야 합니다")
    if override != configured_learning_rate and run_tag is None:
        raise TrainingPipelineError(
            "baseline과 다른 learning rate에는 --run-tag가 필요합니다")
    return override


def resolve_input_layer_norm(
    configured_input_layer_norm: bool,
    override: bool | None,
    run_tag: str | None,
) -> bool:
    """Run별 input LayerNorm override를 검증하고 baseline 덮어쓰기를 차단한다."""

    if not isinstance(configured_input_layer_norm, bool):
        raise TrainingPipelineError("input_layer_norm config는 bool이어야 합니다")
    if override is None:
        return configured_input_layer_norm
    if not isinstance(override, bool):
        raise TrainingPipelineError("input layer norm override는 bool이어야 합니다")
    if override != configured_input_layer_norm and run_tag is None:
        raise TrainingPipelineError(
            "baseline과 다른 input LayerNorm에는 --run-tag가 필요합니다")
    return override


def resolve_run_names(backbone: str, seed: int, run_tag: str | None) -> tuple[str, str]:
    """안전한 local output directory와 MLflow run 이름을 만든다."""

    if run_tag is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_tag):
            raise TrainingPipelineError(
                "run tag는 영문자/숫자로 시작하고 영문자, 숫자, _, -만 사용할 수 있습니다")
        suffix = f"_{run_tag}"
    else:
        suffix = ""
    return f"seed_{seed}{suffix}", f"{backbone}_seed{seed}{suffix}"


class ContextTrainingConfig:
    """STEP 6-C에서 사용하는 학습 설정."""

    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        max_epochs: int,
        seeds: Sequence[int],
        learning_rate: float,
        weight_decay: float,
        gradient_clip_max_norm: float,
        scheduler_factor: float,
        scheduler_patience: int,
        scheduler_min_lr: float,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        mlflow_enabled: bool,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: str,
        output_root: str,
    ) -> None:
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.seeds = tuple(seeds)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_min_lr = scheduler_min_lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.mlflow_enabled = mlflow_enabled
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.output_root = output_root

    def validate(self) -> None:
        positive = {
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_min_lr": self.scheduler_min_lr,
            "early_stopping_patience": self.early_stopping_patience,
        }
        if any(value <= 0 for value in positive.values()):
            raise TrainingPipelineError(f"양수여야 하는 training config가 있습니다: {positive}")
        if self.num_workers < 0 or self.weight_decay < 0:
            raise TrainingPipelineError("num_workers와 weight_decay는 0 이상이어야 합니다")
        if self.scheduler_patience < 0 or self.early_stopping_min_delta < 0:
            raise TrainingPipelineError("patience/min_delta가 잘못됐습니다")
        if not 0 < self.scheduler_factor < 1:
            raise TrainingPipelineError("scheduler factor는 0과 1 사이여야 합니다")
        if not self.seeds:
            raise TrainingPipelineError("seed가 하나 이상 필요합니다")


def training_config_from_mapping(config: Mapping[str, Any]) -> ContextTrainingConfig:
    """YAML mapping을 검증된 training config로 변환한다."""

    try:
        data = config["data"]
        training = config["training"]
        loss = training["loss"]
        optimizer = training["optimizer"]
        clipping = training["gradient_clip"]
        scheduler = training["scheduler"]
        early = training["early_stopping"]
        checkpoint = training["checkpoint"]
        mlflow_config = config["mlflow"]
        if data["additional_normalization"] != "none":
            raise TrainingPipelineError("feature normalization은 none이어야 합니다")
        if data["imputed_mask_concatenated"] is not False:
            raise TrainingPipelineError("imputed mask를 feature에 concat할 수 없습니다")
        if loss != {"name": "cross_entropy", "class_weight": None,
                    "label_smoothing": 0.0}:
            raise TrainingPipelineError("CrossEntropyLoss 정책이 다릅니다")
        if optimizer["name"] != "adamw":
            raise TrainingPipelineError("optimizer는 AdamW여야 합니다")
        if scheduler["name"] != "reduce_lr_on_plateau" or scheduler["mode"] != "min":
            raise TrainingPipelineError("scheduler 정책이 다릅니다")
        if early["monitor"] != "val_loss":
            raise TrainingPipelineError("early stopping monitor는 val_loss여야 합니다")
        if checkpoint["criterion"] != "minimum_val_loss":
            raise TrainingPipelineError("checkpoint 기준은 minimum_val_loss여야 합니다")
        if training["precision"] != "float32" or training["amp"] is not False:
            raise TrainingPipelineError("precision은 float32, AMP는 off여야 합니다")
        if config["test_policy"]["sealed"] is not True:
            raise TrainingPipelineError("test split은 sealed 상태여야 합니다")
        instance = ContextTrainingConfig(
            train_batch_size=int(data["train_batch_size"]),
            val_batch_size=int(data["val_batch_size"]),
            num_workers=int(data["num_workers"]),
            max_epochs=int(training["max_epochs"]),
            seeds=[int(seed) for seed in training["seeds"]],
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_max_norm=float(clipping["max_norm"]),
            scheduler_factor=float(scheduler["factor"]),
            scheduler_patience=int(scheduler["patience"]),
            scheduler_min_lr=float(scheduler["min_lr"]),
            early_stopping_patience=int(early["patience"]),
            early_stopping_min_delta=float(early["min_delta"]),
            mlflow_enabled=bool(mlflow_config["enabled"]),
            mlflow_tracking_uri=str(mlflow_config["tracking_uri"]),
            mlflow_experiment_name=str(mlflow_config["experiment_name"]),
            output_root=str(config["output"]["root"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, TrainingPipelineError):
            raise
        raise TrainingPipelineError("training config field가 잘못됐습니다") from error
    instance.validate()
    return instance


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise TrainingPipelineError(f"{name}에 NaN/Inf가 발생했습니다")


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise TrainingPipelineError("metric 입력 길이가 잘못됐습니다")
    confusion = np.zeros((2, 2), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        if label not in (0, 1) or prediction not in (0, 1):
            raise TrainingPipelineError("binary label/prediction만 지원합니다")
        confusion[label, prediction] += 1

    class_f1 = []
    precision = recall = drowsy_f1 = 0.0
    for class_index in (0, 1):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        class_precision = true_positive / (true_positive + false_positive) \
            if true_positive + false_positive else 0.0
        class_recall = true_positive / (true_positive + false_negative) \
            if true_positive + false_negative else 0.0
        f1 = 2 * class_precision * class_recall / (class_precision + class_recall) \
            if class_precision + class_recall else 0.0
        class_f1.append(f1)
        if class_index == 1:
            precision, recall, drowsy_f1 = class_precision, class_recall, f1
    accuracy = float(np.trace(confusion) / confusion.sum())
    return {
        "val_accuracy": accuracy,
        "val_macro_f1": float(sum(class_f1) / 2),
        "val_drowsy_precision": precision,
        "val_drowsy_recall": recall,
        "val_drowsy_f1": drowsy_f1,
        "confusion_matrix": confusion.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    gradient_clip_max_norm: float,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    sample_count = 0
    clipping_steps = 0
    for batch in loader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        _require_finite(float(loss.detach().item()), "train_loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_max_norm)
        _require_finite(float(gradient_norm.detach().item()), "gradient_norm")
        clipping_steps += 1
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * count
        sample_count += count
    if sample_count == 0:
        raise TrainingPipelineError("train loader가 비어 있습니다")
    mean_loss = total_loss / sample_count
    _require_finite(mean_loss, "train_loss")
    return {"train_loss": mean_loss, "gradient_clip_steps": clipping_steps}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            logits = model(features)
            loss = criterion(logits, labels)
            _require_finite(float(loss.item()), "val_loss")
            count = int(labels.shape[0])
            total_loss += float(loss.item()) * count
            sample_count += count
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise TrainingPipelineError("validation loader가 비어 있습니다")
    val_loss = total_loss / sample_count
    _require_finite(val_loss, "val_loss")
    return {"val_loss": val_loss, **classification_metrics(labels_all, predictions_all)}


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in HISTORY_COLUMNS} for row in history)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_training_curves(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o",
                 label="accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], marker="o",
                 label="macro F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set_xticks((0, 1), labels=("not_drowsy", "drowsy"))
    axis.set_yticks((0, 1), labels=("not_drowsy", "drowsy"))
    axis.set(xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=150)
    plt.close(figure)


class OptionalMLflowLogger:
    """비활성화 시 MLflow를 import하지 않는 선택적 logging layer."""

    def __init__(self, enabled: bool, tracking_uri: str, experiment_name: str) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.module: Any | None = None
        self.run_id: str | None = None

    def start(self, run_name: str, parameters: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(dict(parameters))
        self.module = mlflow
        self.run_id = run.info.run_id

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if self.module is not None:
            self.module.log_metrics(dict(metrics), step=epoch)

    def log_summary(self, metrics: Mapping[str, float]) -> None:
        if self.module is not None:
            self.module.log_metrics(dict(metrics))

    def finish(self, output_dir: Path, status: str = "FINISHED") -> None:
        if self.module is not None:
            if status == "FINISHED":
                self.module.log_artifacts(str(output_dir))
            self.module.end_run(status=status)


def fit_context_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: ContextTrainingConfig,
    device: torch.device,
    output_dir: Path,
    max_epochs: int,
    metadata: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    mlflow_logger: OptionalMLflowLogger,
) -> dict[str, Any]:
    training_started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(config_snapshot)
    snapshot["resolved_run"] = dict(metadata) | {
        "device": str(device), "max_epochs": max_epochs,
        "mlflow_enabled": mlflow_logger.enabled,
    }
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")

    criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=0.0)
    optimizer = AdamW(
        model.parameters(), lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience, min_lr=training_config.scheduler_min_lr)
    model.to(device)

    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    early_reference = math.inf
    epochs_without_improvement = 0
    gradient_clip_steps = 0
    scheduler_steps = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            training_config.gradient_clip_max_norm)
        validation = evaluate(model, val_loader, criterion, device)
        gradient_clip_steps += int(train_result["gradient_clip_steps"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_result["train_loss"]),
            "val_loss": float(validation["val_loss"]),
            "val_accuracy": float(validation["val_accuracy"]),
            "val_macro_f1": float(validation["val_macro_f1"]),
            "val_drowsy_precision": float(validation["val_drowsy_precision"]),
            "val_drowsy_recall": float(validation["val_drowsy_recall"]),
            "val_drowsy_f1": float(validation["val_drowsy_f1"]),
            "learning_rate": learning_rate,
        }
        history.append(row)
        _write_history(output_dir / "history.csv", history)

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            best_metrics = {**row, "confusion_matrix": validation["confusion_matrix"]}
            _atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "metadata": dict(metadata),
                "test_used": False,
            }, output_dir / "best_model.pt")

        scheduler.step(row["val_loss"])
        scheduler_steps += 1
        mlflow_logger.log_epoch(epoch, {
            key: float(value) for key, value in row.items() if key != "epoch"
        })
        print(
            f"epoch={epoch}/{max_epochs} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_accuracy={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} lr={learning_rate:.8f}",
            flush=True,
        )

        if row["val_loss"] < early_reference - training_config.early_stopping_min_delta:
            early_reference = row["val_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                stopped_early = True
                break

    _save_training_curves(output_dir / "training_curves.png", history)
    _save_confusion_matrix(output_dir / "confusion_matrix.png", best_metrics["confusion_matrix"])
    training_time_sec = time.perf_counter() - training_started_at
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    scheduler_lr_reduced = final_learning_rate < training_config.learning_rate
    final_metrics = {
        **dict(metadata),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "stopped_early": stopped_early,
        "gradient_clip_steps": gradient_clip_steps,
        "scheduler_steps": scheduler_steps,
        "final_learning_rate": final_learning_rate,
        "scheduler_lr_reduced": scheduler_lr_reduced,
        "training_time_sec": training_time_sec,
        "amp_enabled": False,
        "feature_normalization": "none",
        "imputed_mask_input": False,
        "test_access_count": 0,
        "mlflow_enabled": mlflow_logger.enabled,
        "mlflow_run_id": mlflow_logger.run_id,
    }
    _write_json(output_dir / "final_metrics.json", final_metrics)
    mlflow_logger.log_summary({
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_metrics["val_loss"]),
        "best_val_accuracy": float(best_metrics["val_accuracy"]),
        "best_val_macro_f1": float(best_metrics["val_macro_f1"]),
        "best_val_drowsy_recall": float(best_metrics["val_drowsy_recall"]),
        "epochs_completed": float(len(history)),
        "final_learning_rate": final_learning_rate,
        "scheduler_lr_reduced": float(scheduler_lr_reduced),
        "training_time_sec": training_time_sec,
        "stopped_early": float(stopped_early),
    })
    return final_metrics


def train_context_lstm(
    project_root: Path,
    backbone: str,
    seed: int,
    config: Mapping[str, Any],
    *,
    epochs_override: int | None = None,
    mlflow_enabled_override: bool | None = None,
    device_name: str = "auto",
    output_root_override: Path | None = None,
    train_batch_size_override: int | None = None,
    classifier_dropout_override: float | None = None,
    weight_decay_override: float | None = None,
    learning_rate_override: float | None = None,
    input_layer_norm_override: bool | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    baseline_training_config = training_config_from_mapping(config)
    if seed not in baseline_training_config.seeds:
        raise TrainingPipelineError(f"허용되지 않은 seed입니다: {seed}")
    max_epochs = (epochs_override if epochs_override is not None
                  else baseline_training_config.max_epochs)
    if max_epochs <= 0:
        raise TrainingPipelineError("epochs는 양수여야 합니다")
    resolved_train_batch_size = resolve_train_batch_size(
        baseline_training_config.train_batch_size, train_batch_size_override, run_tag)
    configured_classifier_dropout = float(
        config["model_baseline_future"]["classifier_dropout"])
    resolved_classifier_dropout = resolve_classifier_dropout(
        configured_classifier_dropout, classifier_dropout_override, run_tag)
    resolved_weight_decay = resolve_weight_decay(
        baseline_training_config.weight_decay, weight_decay_override, run_tag)
    resolved_learning_rate = resolve_learning_rate(
        baseline_training_config.learning_rate, learning_rate_override, run_tag)
    resolved_input_layer_norm = resolve_input_layer_norm(
        config["model_baseline_future"]["input_layer_norm"],
        input_layer_norm_override,
        run_tag,
    )
    output_name, mlflow_run_name = resolve_run_names(backbone, seed, run_tag)
    resolved_config = copy.deepcopy(config)
    resolved_config["model_baseline_future"]["classifier_dropout"] = (
        resolved_classifier_dropout)
    resolved_config["training"]["optimizer"]["weight_decay"] = resolved_weight_decay
    resolved_config["training"]["optimizer"]["learning_rate"] = resolved_learning_rate
    resolved_config["model_baseline_future"]["input_layer_norm"] = (
        resolved_input_layer_norm)
    training_config = training_config_from_mapping(resolved_config)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name in ("cpu", "cuda"):
        if device_name == "cuda" and not torch.cuda.is_available():
            raise TrainingPipelineError("CUDA를 사용할 수 없습니다")
        device = torch.device(device_name)
    else:
        raise TrainingPipelineError(f"지원하지 않는 device입니다: {device_name}")

    set_reproducible_seed(seed)
    train_dataset = ContextFeatureDataset(project_root, backbone, "train")
    val_dataset = ContextFeatureDataset(project_root, backbone, "val")
    generator = torch.Generator().manual_seed(seed)
    train_loader = build_context_feature_dataloader(
        train_dataset, resolved_train_batch_size,
        num_workers=training_config.num_workers, generator=generator)
    val_loader = build_context_feature_dataloader(
        val_dataset, training_config.val_batch_size,
        num_workers=training_config.num_workers)
    model = build_context_lstm(resolved_config)

    base_output = (output_root_override if output_root_override is not None else
                   project_root / training_config.output_root)
    output_dir = base_output / backbone / output_name
    enabled = (training_config.mlflow_enabled if mlflow_enabled_override is None
               else mlflow_enabled_override)
    logger = OptionalMLflowLogger(
        enabled, training_config.mlflow_tracking_uri,
        training_config.mlflow_experiment_name)
    metadata = {
        "backbone": backbone,
        "seed": seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_batch_size": resolved_train_batch_size,
        "val_batch_size": training_config.val_batch_size,
        "classifier_dropout": resolved_classifier_dropout,
        "lstm_dropout": float(config["model_baseline_future"]["lstm_dropout"]),
        "weight_decay": resolved_weight_decay,
        "learning_rate": resolved_learning_rate,
        "input_layer_norm": resolved_input_layer_norm,
        "run_tag": run_tag,
    }
    model_config = resolved_config["model_baseline_future"]
    logger.start(mlflow_run_name, {
        **metadata,
        "max_epochs": max_epochs,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "hidden_size": model_config["hidden_size"],
        "num_layers": model_config["num_layers"],
        "bidirectional": model_config["bidirectional"],
        "feature_normalization": config["data"]["additional_normalization"],
    })
    try:
        result = fit_context_lstm(
            model, train_loader, val_loader, training_config, device,
            output_dir, max_epochs, metadata, resolved_config, logger)
    except Exception:
        logger.finish(output_dir, status="FAILED")
        raise
    logger.finish(output_dir)
    return result
