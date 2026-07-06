from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fkg_mm.config import DEFAULT_K_IMG, DEFAULT_K_TAB
from fkg_mm.fis_frb import run_fis_frb
from fkg_mm.fusion import fuse_feature_selection
from fkg_mm.fkgs import run_fkgs_experiments
from fkg_mm.preprocessing.data import load_records, prepare_records
from fkg_mm.preprocessing.image_features import extract_image_features
from fkg_mm.preprocessing.tabular_features import extract_tabular_features

DEFAULT_LABELS_CSV = Path("data/labels_brset.csv")  # Input nhãn/metadata gốc.
DEFAULT_IMAGE_DIR = Path("data/fundus_photos")  # Thư mục ảnh fundus đầu vào.
DEFAULT_RECORDS_CSV = Path("outputs/intermediate/records.csv")  # Bảng records sau bước prepare_records.
DEFAULT_SUMMARY_JSON = Path("outputs/reports/dataset_summary.json")  # Báo cáo thống kê dataset.
DEFAULT_TABLE_FEATURES_CSV = Path("outputs/features/table_fts.csv")  # Feature tabular đã select K.
DEFAULT_TABLE_SOURCE_CSV = Path("outputs/intermediate/data_process_table_source_columns.csv")  # Bản sao cột tabular gốc để debug.
DEFAULT_SOURCE_STYLE_TABLE_CSV = Path("outputs/intermediate/data_process_tabular.csv")  # Bảng tabular mapping theo style source.
DEFAULT_IMAGE_FEATURES_CSV = Path("outputs/features/image_features.csv")  # Feature ảnh raw trước normalize.
DEFAULT_IMAGE_FEATURES_NORM_CSV = Path("outputs/features/image_fts_norm.csv")  # Feature ảnh đã normalize [0,1].
DEFAULT_FUSED_FEATURES_CSV = Path("outputs/features/data_process.csv")  # Feature fusion cuối cho FIS/FRB.
DEFAULT_FIS_OUTPUT_DIR = Path("outputs/fis")  # Root output cho bước sinh FIS/FRB.
DEFAULT_TRAIN_RULE_CSV = Path("outputs/fis/output/feature_selection/FRB/TrainDataRule.csv")  # FRB train đầu vào FKGS.
DEFAULT_TEST_RULE_CSV = Path("outputs/fis/output/feature_selection/FRB/TestDataRule.csv")  # FRB test đầu vào FKGS.
DEFAULT_FKGS_OUTPUT_DIR = Path("outputs/fkgs/feature_selection")  # Root output kết quả FKGS.


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run_preprocessing(args: argparse.Namespace) -> None:
    records_csv = project_path(args.records_csv)
    summary_json = project_path(args.summary_json)
    records = prepare_records(
        labels_csv=project_path(args.labels_csv),
        image_dir=project_path(args.image_dir),
        records_csv=records_csv,
        summary_json=summary_json,
    )
    print(f"Wrote {len(records)} records to {records_csv}")

    loaded = load_records(records_csv, limit=args.limit)
    extract_tabular_features(
        loaded,
        output_csv=project_path(args.table_features_csv),
        processed_table_csv=project_path(args.table_source_csv),
        source_style_table_csv=project_path(args.source_style_table_csv),
    )

    if args.skip_images:
        print("Skipped image feature extraction.")
        return

    extract_image_features(
        loaded,
        raw_output_csv=project_path(args.image_features_csv),
        normalized_output_csv=project_path(args.image_features_norm_csv),
    )


def run_feature_selection_fusion(args: argparse.Namespace) -> None:
    fused = fuse_feature_selection(
        image_features_csv=project_path(args.image_features_csv),
        table_features_csv=project_path(args.table_features_csv),
        output_csv=project_path(args.output_csv),
        k_img=args.k_img,
        k_tab=args.k_tab,
        balance=not args.no_balance,
    )
    print(f"Wrote {len(fused)} fused feature rows to {project_path(args.output_csv)}")


def run_fis_frb_generation(args: argparse.Namespace) -> None:
    summary = run_fis_frb(
        input_csv=project_path(args.input_csv),
        output_dir=project_path(args.output_dir),
        file_name=args.file_name,
        random_state=args.random_state,
    )
    print(f"Wrote train FRB to {summary['train_rule_csv']}")
    print(f"Wrote test FRB to {summary['test_rule_csv']}")
    print(f"Generated {summary['rule_rows']} rules in {summary['runtime_seconds']:.2f}s")


def run_fkgs(args: argparse.Namespace) -> None:
    summary = run_fkgs_experiments(
        train_rule_csv=project_path(args.train_rule_csv),
        test_rule_csv=project_path(args.test_rule_csv),
        output_dir=project_path(args.output_dir),
        ran_values=args.ran,
        error_thresholds=args.error_threshold,
        turns=args.turns,
        random_state=args.random_state,
    )
    print(f"Wrote FKGS summary to {project_path(args.output_dir) / 'fkgs_summary.csv'}")
    print(summary)


def run_all(args: argparse.Namespace) -> None:
    print("[1/4] Running preprocess...")
    run_preprocessing(
        argparse.Namespace(
            labels_csv=args.labels_csv,
            image_dir=args.image_dir,
            records_csv=args.records_csv,
            summary_json=args.summary_json,
            table_features_csv=args.table_features_csv,
            table_source_csv=args.table_source_csv,
            source_style_table_csv=args.source_style_table_csv,
            image_features_csv=args.image_features_csv,
            image_features_norm_csv=args.image_features_norm_csv,
            limit=args.limit,
            skip_images=args.skip_images,
        )
    )
    print("[2/4] Running fuse-feature-selection...")
    run_feature_selection_fusion(
        argparse.Namespace(
            image_features_csv=args.image_features_norm_csv,
            table_features_csv=args.table_features_csv,
            output_csv=args.fused_output_csv,
            k_img=args.k_img,
            k_tab=args.k_tab,
            no_balance=args.no_balance,
        )
    )
    print("[3/4] Running generate-fis-frb...")
    run_fis_frb_generation(
        argparse.Namespace(
            input_csv=args.fused_output_csv,
            output_dir=args.fis_output_dir,
            file_name=args.file_name,
            random_state=args.random_state,
        )
    )
    print("[4/4] Running run-fkgs...")
    run_fkgs(
        argparse.Namespace(
            train_rule_csv=args.train_rule_csv,
            test_rule_csv=args.test_rule_csv,
            output_dir=args.fkgs_output_dir,
            ran=args.ran,
            error_threshold=args.error_threshold,
            turns=args.turns,
            random_state=args.random_state,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FKG-MM pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    prep = subparsers.add_parser("preprocess", help="Run data/image/table preprocessing.")
    prep.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    prep.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    prep.add_argument("--records-csv", type=Path, default=DEFAULT_RECORDS_CSV)
    prep.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    prep.add_argument("--table-features-csv", type=Path, default=DEFAULT_TABLE_FEATURES_CSV)
    prep.add_argument("--table-source-csv", type=Path, default=DEFAULT_TABLE_SOURCE_CSV)
    prep.add_argument("--source-style-table-csv", type=Path, default=DEFAULT_SOURCE_STYLE_TABLE_CSV)
    prep.add_argument("--image-features-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_CSV)
    prep.add_argument("--image-features-norm-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_NORM_CSV)
    prep.add_argument("--limit", type=int, default=None, help="Use a subset for smoke tests.")
    prep.add_argument("--skip-images", action="store_true", help="Only run records + tabular preprocessing.")
    prep.set_defaults(func=run_preprocessing)

    fuse = subparsers.add_parser("fuse-feature-selection", help="Fuse image/table features with SelectKBest.")
    fuse.add_argument("--image-features-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_NORM_CSV)
    fuse.add_argument("--table-features-csv", type=Path, default=DEFAULT_TABLE_FEATURES_CSV)
    fuse.add_argument("--output-csv", type=Path, default=DEFAULT_FUSED_FEATURES_CSV)
    fuse.add_argument("--k-img", type=int, default=DEFAULT_K_IMG)
    fuse.add_argument("--k-tab", type=int, default=DEFAULT_K_TAB)
    fuse.add_argument("--no-balance", action="store_true", help="Skip BorderlineSMOTE.")
    fuse.set_defaults(func=run_feature_selection_fusion)

    fis = subparsers.add_parser("generate-fis-frb", help="Generate FIS fuzzy rules and FRB files.")
    fis.add_argument("--input-csv", type=Path, default=DEFAULT_FUSED_FEATURES_CSV)
    fis.add_argument("--output-dir", type=Path, default=DEFAULT_FIS_OUTPUT_DIR)
    fis.add_argument("--file-name", default="feature_selection")
    fis.add_argument("--random-state", type=int, default=None)
    fis.set_defaults(func=run_fis_frb_generation)

    fkgs = subparsers.add_parser("run-fkgs", help="Run FKGS on generated FRB files.")
    fkgs.add_argument("--train-rule-csv", type=Path, default=DEFAULT_TRAIN_RULE_CSV)
    fkgs.add_argument("--test-rule-csv", type=Path, default=DEFAULT_TEST_RULE_CSV)
    fkgs.add_argument("--output-dir", type=Path, default=DEFAULT_FKGS_OUTPUT_DIR)
    fkgs.add_argument("--ran", type=int, nargs="+", default=[15, 20])
    fkgs.add_argument("--error-threshold", type=float, nargs="+", default=[0.2, 0.3])
    fkgs.add_argument("--turns", type=int, default=5)
    fkgs.add_argument("--random-state", type=int, default=None)
    fkgs.set_defaults(func=run_fkgs)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        run_all(argparse.Namespace(
            labels_csv=DEFAULT_LABELS_CSV,
            image_dir=DEFAULT_IMAGE_DIR,
            records_csv=DEFAULT_RECORDS_CSV,
            summary_json=DEFAULT_SUMMARY_JSON,
            table_features_csv=DEFAULT_TABLE_FEATURES_CSV,
            table_source_csv=DEFAULT_TABLE_SOURCE_CSV,
            source_style_table_csv=DEFAULT_SOURCE_STYLE_TABLE_CSV,
            image_features_csv=DEFAULT_IMAGE_FEATURES_CSV,
            image_features_norm_csv=DEFAULT_IMAGE_FEATURES_NORM_CSV,
            limit=None,
            skip_images=False,
            fused_output_csv=DEFAULT_FUSED_FEATURES_CSV,
            k_img=DEFAULT_K_IMG,
            k_tab=DEFAULT_K_TAB,
            no_balance=False,
            fis_output_dir=DEFAULT_FIS_OUTPUT_DIR,
            file_name="feature_selection",
            random_state=None,
            train_rule_csv=DEFAULT_TRAIN_RULE_CSV,
            test_rule_csv=DEFAULT_TEST_RULE_CSV,
            fkgs_output_dir=DEFAULT_FKGS_OUTPUT_DIR,
            ran=[15, 20],
            error_threshold=[0.2, 0.3],
            turns=5,
        ))
        return
    args.func(args)


if __name__ == "__main__":
    main()
