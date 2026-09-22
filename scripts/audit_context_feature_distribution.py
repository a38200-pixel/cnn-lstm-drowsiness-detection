"""STEP 6-E2 frozen Context CNN feature의 train/validation 정적 분포를 감사한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_feature_distribution_audit import (  # noqa: E402
    run_context_feature_distribution_audit,
)


def main() -> int:
    result = run_context_feature_distribution_audit(
        PROJECT_ROOT / "data/interim/features_v2/context_full",
        PROJECT_ROOT / "outputs/audits_v2/context_feature_distribution",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
