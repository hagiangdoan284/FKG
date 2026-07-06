from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_FUSED_FEATURES_CSV = Path("outputs/features/data_process.csv")
DEFAULT_FIS_OUTPUT_DIR = Path("outputs/fis")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


from fkg_mm.fis_frb import run_fis_frb


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FIS fuzzy rules and FRB files.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FUSED_FEATURES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIS_OUTPUT_DIR)
    parser.add_argument("--file-name", default="feature_selection")
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    summary = run_fis_frb(
        input_csv=project_path(args.input_csv),
        output_dir=project_path(args.output_dir),
        file_name=args.file_name,
        random_state=args.random_state,
    )
    print(f"Wrote train FRB to {summary['train_rule_csv']}")
    print(f"Wrote test FRB to {summary['test_rule_csv']}")
    print(f"Generated {summary['rule_rows']} rules in {summary['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
