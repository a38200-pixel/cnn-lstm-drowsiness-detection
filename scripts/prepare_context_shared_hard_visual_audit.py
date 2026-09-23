"""STEP 6-E5 shared hard/control validation contact sheet를 준비한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.evaluation_v2.context_shared_hard_visual_audit import (  # noqa: E402
    prepare_context_shared_hard_visual_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6-E5 validation visual audit pack 준비")
    parser.add_argument("--tile-size", type=int, default=180)
    args = parser.parse_args()
    result = prepare_context_shared_hard_visual_audit(
        PROJECT_ROOT,
        PROJECT_ROOT / "outputs/audits_v2/context_shared_hard_visual",
        tile_size=args.tile_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
