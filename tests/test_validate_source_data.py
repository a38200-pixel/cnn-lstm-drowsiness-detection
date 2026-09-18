"""실제 영상을 decode하지 않는 2차 실험 원본 데이터 검증 테스트."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from drowsiness_detection.preprocessing_v2.validate_source_data import (  # noqa: E402
    REPORT_FILENAMES,
    build_config,
    read_csv_data,
    run_source_validation,
)


class SourceValidationTest(unittest.TestCase):
    """경로 해석, CSV 파싱, 보고서 생성을 검증한다."""

    def test_small_fixture_generates_all_reports(self) -> None:
        """실제 데이터를 조사하지 않고 작은 독립 fixture를 검증한다."""

        # 제한된 CI 또는 sandbox 환경에서 사용자 전역 임시 폴더에 접근하지 않도록
        # fixture를 저장소 내부에 생성한다.
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "data/raw/SUST Driver Drowsiness Dataset"
            metadata_dir = root / "data/metadata"
            dlib_path = root / "assets/dlib/predictor.dat"
            yunet_path = root / "assets/yunet/model.onnx"
            output_dir = root / "outputs/source_validation"
            for directory in (
                raw_dir / "drowsy",
                raw_dir / "not_drowsy",
                metadata_dir,
                dlib_path.parent,
                yunet_path.parent,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            dlib_path.touch()
            yunet_path.touch()

            rows = {
                "train": ("d_1", "drowsy"),
                "val": ("n_1", "not_drowsy"),
                "test": ("d_2", "drowsy"),
            }
            master_lines = ["video_id,video_path,label"]
            for split, (video_id, label) in rows.items():
                relative_video = f"{label}/{video_id}.mp4"
                (raw_dir / relative_video).touch()
                project_video = (
                    "data/raw/SUST Driver Drowsiness Dataset/" + relative_video
                )
                master_lines.append(f"{video_id},{project_video},{label}")
                (metadata_dir / f"{split}.csv").write_text(
                    "video_id,video_path,label,split\n"
                    f"{video_id},{project_video},{label},{split}\n",
                    encoding="utf-8",
                )
            (metadata_dir / "sust_ddd_video_metadata.csv").write_text(
                "\n".join(master_lines) + "\n", encoding="utf-8"
            )
            (metadata_dir / "split_summary.csv").write_text(
                "split,total_videos\ntrain,1\nval,1\ntest,1\n",
                encoding="utf-8",
            )

            parsed = read_csv_data(metadata_dir / "train.csv")
            self.assertEqual(parsed.schema["selected_video_id_column"], "video_id")
            self.assertEqual(parsed.schema["selected_label_column"], "label")
            self.assertEqual(parsed.schema["selected_source_path_column"], "video_path")

            config = build_config(
                project_root=root,
                raw_dir="data/raw/SUST Driver Drowsiness Dataset",
                metadata_dir="data/metadata",
                output_dir="outputs/source_validation",
                dlib_predictor="assets/dlib/predictor.dat",
                yunet_model="assets/yunet/model.onnx",
                skip_decode=True,
                skip_asset_load=True,
            )
            summary = run_source_validation(config)

            self.assertEqual(summary["decision"], "READY_WITH_WARNINGS")
            self.assertEqual(summary["raw_video_matching"]["raw_video_count"], 3)
            self.assertEqual(
                summary["split_integrity"]["pairwise_overlap"]["train_vs_val"][
                    "count"
                ],
                0,
            )
            for filename in REPORT_FILENAMES:
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
