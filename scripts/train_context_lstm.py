"""STEP 6-C frozen Context feature LSTM train/validation 실행기."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.data_v2 import BACKBONES  # noqa: E402
from drowsiness_detection.training_v2 import train_context_lstm  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/context_lstm_baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mlflow_group = parser.add_mutually_exclusive_group()
    mlflow_group.add_argument("--mlflow", dest="mlflow", action="store_true")
    mlflow_group.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    parser.set_defaults(mlflow=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    result = train_context_lstm(
        PROJECT_ROOT,
        args.backbone,
        args.seed,
        config,
        epochs_override=args.epochs,
        mlflow_enabled_override=args.mlflow,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
