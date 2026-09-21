"""STEP 5-D1 Context CNN feature preflight와 16-video controlled pilot을 실행한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.cnn_feature_extractor import (  # noqa: E402
    DependencyBlocked, dry_run, full_dry_run, run_full, run_pilot, select_pilot,
    write_blocked_report, write_pilot_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resnet-batch-size", type=int, default=64)
    parser.add_argument("--vgg-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run과 --resume은 함께 사용할 수 없습니다")
    if args.resnet_batch_size < 1 or args.vgg_batch_size < 1:
        parser.error("batch size는 양수여야 합니다")
    config = PROJECT_ROOT / "configs/context_cnn_features.yaml"
    if args.mode == "full":
        planned = full_dry_run(
            PROJECT_ROOT, config, resnet_batch_size=args.resnet_batch_size,
            vgg_batch_size=args.vgg_batch_size)
        if args.dry_run:
            print(json.dumps(planned, ensure_ascii=False, indent=2))
            return 0
        result = run_full(
            PROJECT_ROOT, config, PROJECT_ROOT / "data/interim/features_v2/context_full",
            PROJECT_ROOT / "outputs/features_v2/context_cnn/full",
            device=args.device, resnet_batch_size=args.resnet_batch_size,
            vgg_batch_size=args.vgg_batch_size, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    planned = dry_run(PROJECT_ROOT, config)
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return 0
    manifest = PROJECT_ROOT / "data/metadata/experiment2_step5d_cnn_feature_pilot_manifest.csv"
    write_pilot_manifest(manifest, select_pilot(PROJECT_ROOT / "data/interim/sequences_v2/context"))
    try:
        result = run_pilot(
            PROJECT_ROOT, config, PROJECT_ROOT / "data/interim/features_v2/context_pilot",
            PROJECT_ROOT / "outputs/features_v2/context_cnn/pilot", manifest,
            device=args.device, resnet_batch_size=args.resnet_batch_size,
            vgg_batch_size=args.vgg_batch_size, resume=args.resume)
    except DependencyBlocked as error:
        write_blocked_report(PROJECT_ROOT, planned)
        print(json.dumps({**planned, "status": "STEP_5D1_IMPLEMENTED_BLOCKED_BY_ENVIRONMENT",
                          "blocker": str(error), "pilot_manifest_written": True,
                          "feature_artifacts_created": False}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
