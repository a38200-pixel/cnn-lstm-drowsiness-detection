"""STEP 5-D2R pilot/full feature mismatch를 read-only로 조사한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.cnn_feature_regression_audit import (  # noqa: E402
    run_audit, write_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-inference", action="store_true",
                        help="Mismatch source와 관련 D1/D2 batch만 재추론합니다.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_audit(PROJECT_ROOT, run_inference=args.run_inference, device=args.device)
    write_audit_report(
        PROJECT_ROOT / "outputs/features_v2/context_cnn/full_regression_audit", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
