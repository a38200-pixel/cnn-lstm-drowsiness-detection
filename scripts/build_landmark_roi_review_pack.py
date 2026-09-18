"""기존 STEP 2-C 결과에서 compact visual review pack을 생성한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drowsiness_detection.preprocessing_v2.landmark_roi_review_pack import build_review_pack


def main() -> int:
    """기존 artifact 경로를 고정해 review pack을 생성한다."""

    result = build_review_pack(
        PROJECT_ROOT,
        PROJECT_ROOT / "outputs" / "preprocessing_v2" / "landmark_roi_margin_audit",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
