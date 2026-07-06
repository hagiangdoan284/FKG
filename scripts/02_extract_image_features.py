from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RECORDS_CSV = Path("outputs/intermediate/records.csv")
DEFAULT_IMAGE_FEATURES_CSV = Path("outputs/features/image_features.csv")
DEFAULT_IMAGE_FEATURES_NORM_CSV = Path("outputs/features/image_fts_norm.csv")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path

from fkg_mm.preprocessing.data import load_records
from fkg_mm.preprocessing.image_features import extract_image_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract source-compatible image features.")
    parser.add_argument("--records-csv", type=Path, default=DEFAULT_RECORDS_CSV)
    parser.add_argument("--raw-output-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_CSV)
    parser.add_argument(
        "--normalized-output-csv",
        type=Path,
        default=DEFAULT_IMAGE_FEATURES_NORM_CSV,
    )
    parser.add_argument("--limit", type=int, default=None, help="Use a small subset for smoke tests.")
    args = parser.parse_args()

    raw_output_csv = project_path(args.raw_output_csv)
    normalized_output_csv = project_path(args.normalized_output_csv)
    records = load_records(project_path(args.records_csv), limit=args.limit)
    extract_image_features(records, raw_output_csv, normalized_output_csv)
    print(f"Wrote raw image features to {raw_output_csv}")
    print(f"Wrote normalized image features to {normalized_output_csv}")


if __name__ == "__main__":
    main()
