"""STEP 5-D3 full Context CNN feature artifact를 read-only로 전수 감사한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_cnn_feature_audit import (  # noqa: E402
    run_final_audit, write_audit_report,
)


def main() -> int:
    result = run_final_audit(PROJECT_ROOT)
    write_audit_report(
        PROJECT_ROOT / "outputs/features_v2/context_cnn/final_integrity_audit", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "STEP_5D3_FINAL_INTEGRITY_AUDIT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
