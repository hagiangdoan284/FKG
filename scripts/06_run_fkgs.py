from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_TRAIN_RULE_CSV = Path("outputs/fis/output/feature_selection/FRB/TrainDataRule.csv")
DEFAULT_TEST_RULE_CSV = Path("outputs/fis/output/feature_selection/FRB/TestDataRule.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/fkgs/feature_selection")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


from fkg_mm.fkgs import run_fkgs_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FKGS on generated FRB files.")
    parser.add_argument("--train-rule-csv", type=Path, default=DEFAULT_TRAIN_RULE_CSV)
    parser.add_argument("--test-rule-csv", type=Path, default=DEFAULT_TEST_RULE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ran", type=int, nargs="+", default=[15, 20])
    parser.add_argument("--error-threshold", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
