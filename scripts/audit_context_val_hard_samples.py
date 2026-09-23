"""STEP 6-E4 baseline validation hard-sample 감사를 실행한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_val_hard_sample_audit import (  # noqa: E402
    run_context_val_hard_sample_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline checkpoint의 validation per-sample CE를 감사합니다.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="추론 device (기본값: CUDA 사용 가능 시 CUDA, 아니면 CPU)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_context_val_hard_sample_audit(
        PROJECT_ROOT,
        PROJECT_ROOT / "outputs/audits_v2/context_val_hard_samples",
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
