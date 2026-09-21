"""사용자 확정 STEP 4-B 판정을 기존 미판정 CSV에 안전하게 적용한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.step4b_manual_review import apply_manual_review


def main() -> int:
    """첨부 원문을 재판단하지 않고 manual 열만 채운다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="사용자가 확정한 review_id별 판정 원문")
    parser.add_argument("--review-root", type=Path, default=PROJECT_ROOT /
                        "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4b_visual_review")
    args = parser.parse_args()
    result = apply_manual_review(args.review_root / "step4b_manual_review.csv", args.source, args.review_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
