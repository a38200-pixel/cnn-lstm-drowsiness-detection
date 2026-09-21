"""동결 C3 정책으로 train/val Context 32-slot 참조 시퀀스를 만든다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.context_sequence_builder import (  # noqa: E402
    EXPECTED, POLICY_HASH, SEQUENCE_POLICY_HASH, load_plan, materialize,
    plan_statistics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="계획만 검증하고 출력하지 않음")
    parser.add_argument("--resume", action="store_true", help="동일 COMPLETE bundle을 검증 후 건너뜀")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run과 --resume을 함께 사용할 수 없습니다")
    canonical = PROJECT_ROOT / "data/interim/preprocessing_v2/canonical"
    freeze = PROJECT_ROOT / "outputs/preprocessing_v2/canonical_preprocessing/quality_missing_audit/step4c_policy_freeze"
    root = PROJECT_ROOT / "data/interim/sequences_v2/context"
    reports = PROJECT_ROOT / "outputs/sequences_v2/context_sequence"
    plans, excluded = load_plan(canonical, freeze, PROJECT_ROOT / "configs/sequence_missing_policy.yaml",
                                PROJECT_ROOT, check_images=not args.dry_run)
    counts = plan_statistics(plans, excluded, EXPECTED)
    if args.dry_run:
        result = {"status": "STEP_5A_DRY_RUN_PASS", **counts, "test_rows": 0,
                  "canonical_policy_hash": POLICY_HASH,
                  "sequence_missing_policy_hash": SEQUENCE_POLICY_HASH,
                  "planned_sequence_root": root.relative_to(PROJECT_ROOT).as_posix(),
                  "planned_report_root": reports.relative_to(PROJECT_ROOT).as_posix(),
                  "jpeg_written": 0, "raw_video_opened": False}
    else:
        result = materialize(plans, excluded, root, reports, PROJECT_ROOT, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
