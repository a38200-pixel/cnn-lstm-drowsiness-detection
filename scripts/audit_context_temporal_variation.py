"""STEP 6-E3 frozen Context feature의 temporal variation/redundancy를 감사한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_temporal_variation_audit import (  # noqa: E402
    run_context_temporal_variation_audit,
)


def main() -> int:
    result = run_context_temporal_variation_audit(
        PROJECT_ROOT / "data/interim/features_v2/context_full",
        PROJECT_ROOT / "outputs/audits_v2/context_temporal_variation",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
