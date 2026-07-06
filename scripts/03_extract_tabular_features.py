from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RECORDS_CSV = Path("outputs/intermediate/records.csv")
# Path file sau khi đã lấy 15 features sẽ sử dụng cho bài toán (dùng để debug, ...)
DEFAULT_TABLE_FEATURES_CSV = Path("outputs/features/table_fts.csv")
# Path file features sau khi đã qua các bước LabelEncoder, MinMaxScaler, SelectKBest
DEFAULT_TABLE_SOURCE_CSV = Path("outputs/intermediate/data_process_table_source_columns.csv")
DEFAULT_SOURCE_STYLE_TABLE_CSV = Path("outputs/intermediate/data_process_tabular.csv")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path

from fkg_mm.config import DEFAULT_K_TAB
from fkg_mm.preprocessing.data import load_records
from fkg_mm.preprocessing.tabular_features import extract_tabular_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract source-compatible tabular features.")
    parser.add_argument("--records-csv", type=Path, default=DEFAULT_RECORDS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TABLE_FEATURES_CSV)
    parser.add_argument("--processed-table-csv", type=Path, default=DEFAULT_TABLE_SOURCE_CSV)
    parser.add_argument("--source-style-table-csv", type=Path, default=DEFAULT_SOURCE_STYLE_TABLE_CSV)
    parser.add_argument("--k-tab", type=int, default=DEFAULT_K_TAB)
    parser.add_argument("--limit", type=int, default=None, help="Use a small subset for smoke tests.")
    args = parser.parse_args()

    output_csv = project_path(args.output_csv)
    processed_table_csv = project_path(args.processed_table_csv)
    source_style_table_csv = project_path(args.source_style_table_csv)
    records = load_records(project_path(args.records_csv), limit=args.limit)
    features = extract_tabular_features(
        records,
        output_csv,
        processed_table_csv,
        source_style_table_csv=source_style_table_csv,
        k_tab=args.k_tab,
    )
    print(f"Wrote {len(features)} tabular feature rows to {output_csv}")
    print(f"Wrote source-column table to {processed_table_csv}")
    print(f"Wrote source-style tabular heatmap table to {source_style_table_csv}")


if __name__ == "__main__":
    main()
