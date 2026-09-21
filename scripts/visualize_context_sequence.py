"""STEP 5-A Context RGB 32-frame sequence를 read-only로 시각 검수한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_detection.sequences_v2.context_sequence_visualizer import (  # noqa: E402
    load_context_sequence, play_sequence, print_summary, render_contact_sheet,
    render_imputed_sheet, save_visualization, sequence_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--mode", choices=("contact", "play", "imputed"), default="contact")
    parser.add_argument("--tile-size", type=int, default=180)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/visualizations/context_sequences")
    parser.add_argument("--fps", type=float, default=3.2,
                        help="원본 FPS가 아닌 visual inspection playback speed")
    args = parser.parse_args()
    rows, images, _ = load_context_sequence(PROJECT_ROOT, args.video_id, args.split)
    summary = sequence_summary(rows)
    print_summary(summary)
    if args.mode == "play":
        if args.save:
            parser.error("play mode는 --save를 지원하지 않습니다")
        play_sequence(rows, images, args.fps)
        return 0
    sheet = (render_contact_sheet(rows, images, args.tile_size)
             if args.mode == "contact" else render_imputed_sheet(rows, images, args.tile_size))
    if args.save:
        image_path, summary_path = save_visualization(args.output_dir, sheet, summary, args.mode)
        print(f"VISUALIZATION: {image_path}")
        print(f"SUMMARY: {summary_path}")
    else:
        window = f"Context Sequence: {summary['split']}/{summary['video_id']} ({args.mode})"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, sheet)
        cv2.waitKey(0)
        cv2.destroyWindow(window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
