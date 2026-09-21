"""STEP 5-B Behavior 100-slot sequence를 read-only 시간축 그래프로 검수한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.behavior_sequence_visualizer import (  # noqa: E402
    load_behavior_sequence, print_summary, render_masks, render_overview,
    save_visualization, sequence_summary, suggest_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--video-id")
    target.add_argument("--suggest-samples", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--mode", choices=("overview", "masks"), default="overview")
    parser.add_argument("--save", action="store_true")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true")
    display.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/visualizations/behavior_sequences"))
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.suggest_samples:
        if args.split or args.save or not args.show or args.mode != "overview":
            parser.error("--suggest-samples는 다른 visualization option과 함께 사용할 수 없습니다")
        for sample in suggest_samples(PROJECT_ROOT):
            print(" | ".join(f"{key}={value}" for key, value in sample.items()))
        return 0
    if args.dpi <= 0:
        parser.error("--dpi는 양수여야 합니다")
    if not args.show and not args.save:
        parser.error("--no-show 사용 시 --save가 필요합니다")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg", force=True)
    sequence = load_behavior_sequence(PROJECT_ROOT, args.video_id, args.split)
    summary = sequence_summary(sequence)
    print_summary(summary)
    figure = (render_overview(sequence, args.dpi) if args.mode == "overview"
              else render_masks(sequence, args.dpi))
    if args.save:
        image_path, summary_path = save_visualization(
            PROJECT_ROOT, args.output_dir, figure, summary, args.mode, args.dpi)
        print(f"VISUALIZATION: {image_path}")
        print(f"SUMMARY: {summary_path}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
