"""STEP 6-B Context LSTM baseline model 테스트."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from drowsiness_detection.models_v2.context_lstm import (
    ContextLSTMConfig,
    ContextLSTMConfigError,
    ContextLSTMBaseline,
    build_context_lstm,
    context_lstm_config_from_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/context_lstm_baseline.yaml"
EXPECTED_PARAMETERS = 337_090


@pytest.fixture(scope="module")
def config_mapping() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def model(config_mapping: dict) -> ContextLSTMBaseline:
    torch.manual_seed(0)
    return build_context_lstm(config_mapping)


def test_config_contract(config_mapping: dict) -> None:
    config = context_lstm_config_from_mapping(config_mapping)
    assert config.sequence_length == 32
    assert config.input_size == 512
    assert config.hidden_size == 128
    assert config.num_layers == 1
    assert config.bidirectional is False
    assert config.lstm_dropout == 0.0
    assert config.classifier_hidden == 64
    assert config.classifier_dropout == 0.0
    assert config.num_classes == 2


def test_model_architecture(model: ContextLSTMBaseline) -> None:
    assert model.lstm.input_size == 512
    assert model.lstm.hidden_size == 128
    assert model.lstm.num_layers == 1
    assert model.lstm.batch_first is True
    assert model.lstm.bidirectional is False
    assert model.lstm.dropout == 0.0
    assert isinstance(model.classifier[0], nn.Linear)
    assert (model.classifier[0].in_features, model.classifier[0].out_features) == (128, 64)
    assert isinstance(model.classifier[1], nn.ReLU)
    assert isinstance(model.classifier[2], nn.Dropout)
    assert model.classifier[2].p == 0.0
    assert isinstance(model.classifier[3], nn.Linear)
    assert (model.classifier[3].in_features, model.classifier[3].out_features) == (64, 2)


def test_forward_returns_finite_raw_logits(model: ContextLSTMBaseline) -> None:
    logits = model(torch.randn(4, 32, 512, dtype=torch.float32))
    assert logits.shape == (4, 2)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert not any(isinstance(module, (nn.Softmax, nn.Sigmoid)) for module in model.modules())


@pytest.mark.parametrize("shape", [(32, 512), (2, 3, 32, 512)])
def test_invalid_input_rank_is_blocked(
        model: ContextLSTMBaseline, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="rank"):
        model(torch.zeros(shape, dtype=torch.float32))


def test_invalid_sequence_length_is_blocked(model: ContextLSTMBaseline) -> None:
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.zeros(2, 31, 512, dtype=torch.float32))


def test_invalid_feature_dimension_is_blocked(model: ContextLSTMBaseline) -> None:
    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.zeros(2, 32, 511, dtype=torch.float32))


def test_non_float32_input_is_blocked(model: ContextLSTMBaseline) -> None:
    with pytest.raises(TypeError, match="float32"):
        model(torch.zeros(2, 32, 512, dtype=torch.float64))


def test_backward_propagates_finite_gradients(model: ContextLSTMBaseline) -> None:
    logits = model(torch.randn(2, 32, 512, dtype=torch.float32))
    logits.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_parameter_count(model: ContextLSTMBaseline) -> None:
    assert model.parameter_counts() == {
        "total": EXPECTED_PARAMETERS,
        "trainable": EXPECTED_PARAMETERS,
    }


def test_same_model_for_both_backbones(config_mapping: dict) -> None:
    resnet_model = build_context_lstm(config_mapping)
    vgg_model = build_context_lstm(config_mapping)
    assert type(resnet_model) is type(vgg_model)
    assert resnet_model.parameter_counts() == vgg_model.parameter_counts()
    assert all("backbone" not in name for name, _ in resnet_model.named_modules())


def test_forward_accepts_features_only() -> None:
    signature = inspect.signature(ContextLSTMBaseline.forward)
    assert tuple(signature.parameters) == ("self", "features")


def test_nonbaseline_config_is_rejected(config_mapping: dict) -> None:
    changed = {
        **config_mapping,
        "model_baseline_future": {
            **config_mapping["model_baseline_future"],
            "bidirectional": True,
        },
    }
    with pytest.raises(ContextLSTMConfigError, match="baseline"):
        build_context_lstm(changed)


def test_module_contains_no_training_components() -> None:
    source = (
        PROJECT_ROOT / "src/drowsiness_detection/models_v2/context_lstm.py"
    ).read_text(encoding="utf-8")
    forbidden = ("Optimizer", "CrossEntropyLoss", "scheduler", ".step()", "softmax(", "sigmoid(")
    assert all(token not in source for token in forbidden)
