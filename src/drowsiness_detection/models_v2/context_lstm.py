"""Frozen Context CNN feature용 단방향 LSTM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class ContextLSTMConfigError(ValueError):
    """Context LSTM baseline config 계약 위반."""


@dataclass(frozen=True)
class ContextLSTMConfig:
    """STEP 6-B에서 동결한 Context LSTM 구조 설정."""

    sequence_length: int
    input_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    lstm_dropout: float
    classifier_hidden: int
    classifier_dropout: float
    num_classes: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ContextLSTMConfig":
        """`context_lstm_baseline.yaml`을 읽은 mapping에서 설정을 만든다."""

        try:
            data = config["data"]
            model = config["model_baseline_future"]
            if model["type"] != "lstm":
                raise ContextLSTMConfigError("model type은 lstm이어야 합니다")
            instance = cls(
                sequence_length=int(data["sequence_length"]),
                input_size=int(model["input_size"]),
                hidden_size=int(model["hidden_size"]),
                num_layers=int(model["num_layers"]),
                bidirectional=bool(model["bidirectional"]),
                lstm_dropout=float(model["lstm_dropout"]),
                classifier_hidden=int(model["classifier_hidden"]),
                classifier_dropout=float(model["classifier_dropout"]),
                num_classes=int(model["num_classes"]),
            )
            feature_dim = int(data["feature_dim"])
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ContextLSTMConfigError):
                raise
            raise ContextLSTMConfigError("Context LSTM config field가 잘못됐습니다") from error
        if feature_dim != instance.input_size:
            raise ContextLSTMConfigError("data feature_dim과 LSTM input_size가 다릅니다")
        instance.validate()
        return instance

    def validate(self) -> None:
        """이번 baseline에서 허용하는 구조만 수용한다."""

        expected = {
            "sequence_length": 32,
            "input_size": 512,
            "hidden_size": 128,
            "num_layers": 1,
            "bidirectional": False,
            "lstm_dropout": 0.0,
            "classifier_hidden": 64,
            "classifier_dropout": 0.0,
            "num_classes": 2,
        }
        actual = {
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "lstm_dropout": self.lstm_dropout,
            "classifier_hidden": self.classifier_hidden,
            "classifier_dropout": self.classifier_dropout,
            "num_classes": self.num_classes,
        }
        mismatches = {
            key: (actual[key], value)
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise ContextLSTMConfigError(
                f"STEP 6-B baseline 구조와 다른 config입니다: {mismatches}")


class ContextLSTMBaseline(nn.Module):
    """`[B,32,512]`를 받아 `[B,2]` raw logits를 반환한다."""

    def __init__(self, config: ContextLSTMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden, config.num_classes),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ContextLSTMBaseline":
        return cls(ContextLSTMConfig.from_mapping(config))

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError("features는 torch.Tensor여야 합니다")
        if features.ndim != 3:
            raise ValueError(
                f"features rank는 3이어야 합니다: shape={tuple(features.shape)}")
        if features.shape[1] != self.config.sequence_length:
            raise ValueError(
                f"sequence length는 {self.config.sequence_length}여야 합니다: "
                f"shape={tuple(features.shape)}")
        if features.shape[2] != self.config.input_size:
            raise ValueError(
                f"feature dimension은 {self.config.input_size}여야 합니다: "
                f"shape={tuple(features.shape)}")
        if features.dtype != torch.float32:
            raise TypeError(f"features dtype은 float32여야 합니다: {features.dtype}")

        _, (hidden, _) = self.lstm(features)
        final_hidden = hidden[-1]
        return self.classifier(final_hidden)

    def parameter_counts(self) -> dict[str, int]:
        """전체 및 학습 가능 parameter 수를 반환한다."""

        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }
