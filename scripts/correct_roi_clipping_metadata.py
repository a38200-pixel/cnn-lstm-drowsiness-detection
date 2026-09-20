"""STEP 2-C 원본을 보존하며 clipping metadata correction을 생성한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.clipping_metadata_correction import write_correction


def main() -> int:
    """기존 결과를 읽고 별도 correction 디렉터리에 기록한다."""

    result_dir = PROJECT_ROOT / "outputs" / "preprocessing_v2" / "landmark_roi_margin_audit"
    print(json.dumps(write_correction(PROJECT_ROOT, result_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
