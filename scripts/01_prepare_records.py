from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_LABELS_CSV = Path("data/labels_brset.csv")
DEFAULT_IMAGE_DIR = Path("data/fundus_photos")
DEFAULT_RECORDS_CSV = Path("outputs/intermediate/records.csv")
DEFAULT_SUMMARY_JSON = Path("outputs/reports/dataset_summary.json")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path

from fkg_mm.preprocessing.data import prepare_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BRSET records for FKG-MM preprocessing.")
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--records-csv", type=Path, default=DEFAULT_RECORDS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()

    records_csv = project_path(args.records_csv)
    summary_json = project_path(args.summary_json)
    records = prepare_records(
        project_path(args.labels_csv),
        project_path(args.image_dir),
        records_csv,
        summary_json,
    )
    print(f"Wrote {len(records)} records to {records_csv}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
