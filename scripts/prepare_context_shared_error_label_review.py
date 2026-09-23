"""STEP 6-E6 shared-error 원본 영상 label consistency review를 준비한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_shared_error_label_review import (  # noqa: E402
    prepare_context_shared_error_label_review,
)


def main() -> int:
    result = prepare_context_shared_error_label_review(
        PROJECT_ROOT,
        PROJECT_ROOT / "outputs/audits_v2/context_shared_error_label_review",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
