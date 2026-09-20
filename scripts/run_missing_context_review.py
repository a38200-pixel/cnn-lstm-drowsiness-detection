"""STEP 3-B Context 결측 27건의 읽기 전용 후처리 진입점이다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.preprocessing_v2.missing_context_review import collect_records, generate_review


def main() -> int:
    """Dry-run에서는 CSV만 확인하며 실제 모드에서만 원본 frame을 읽는다."""

    parser = argparse.ArgumentParser(description="STEP 3-B missing Context crop review pack")
    parser.add_argument("--pilot-root", type=Path, default=PROJECT_ROOT / "data/interim/preprocessing_v2/canonical_pilot")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/pilot/missing_context_review")
    parser.add_argument("--dry-run", action="store_true", help="영상 decode와 파일 생성 없이 대상 row만 확인")
    args = parser.parse_args()
    if args.dry_run:
        records, selected, hashes = collect_records(args.pilot_root)
        result = {"mode": "DRY_RUN", "total_context_selected": selected,
                  "total_context_missing": len(records), "policy_hashes": sorted(hashes),
                  "video_opened": False, "output_written": False}
    else:
        summary = generate_review(args.pilot_root, args.output_dir, expected_count=27)
        result = {"mode": "GENERATED", "total_context_selected": summary["total_context_selected"],
                  "total_context_missing": summary["total_context_missing"],
                  "sample_count": summary["visual_artifacts"]["sample_count"],
                  "sheet_count": summary["visual_artifacts"]["sheet_count"],
                  "output_dir": str(args.output_dir), "manual_interpretation": "WAITING"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
