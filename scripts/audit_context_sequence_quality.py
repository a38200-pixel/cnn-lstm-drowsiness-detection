"""STEP 6-E1 train/validation 32-slot Context sequence 품질을 감사한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_sequence_quality_audit import (  # noqa: E402
    run_context_sequence_quality_audit,
)


def main() -> int:
    result = run_context_sequence_quality_audit(
        PROJECT_ROOT / "data/interim/sequences_v2/context",
        PROJECT_ROOT / "outputs/audits_v2/context_sequence_quality",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
