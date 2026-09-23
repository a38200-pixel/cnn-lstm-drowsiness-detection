"""완료된 STEP 6-E5 manual review를 count/rate로 요약한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_shared_hard_visual_audit import (  # noqa: E402
    summarize_context_shared_hard_visual_audit,
)


def main() -> int:
    result = summarize_context_shared_hard_visual_audit(
        PROJECT_ROOT / "outputs/audits_v2/context_shared_hard_visual")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
