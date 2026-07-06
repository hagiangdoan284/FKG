from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_IMAGE_FEATURES_NORM_CSV = Path("outputs/features/image_fts_norm.csv")
DEFAULT_TABLE_FEATURES_CSV = Path("outputs/features/table_fts.csv")
DEFAULT_FUSED_FEATURES_CSV = Path("outputs/features/data_process.csv")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


from fkg_mm.config import DEFAULT_K_IMG, DEFAULT_K_TAB
from fkg_mm.fusion import fuse_feature_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse image and tabular features with feature selection.")
    parser.add_argument("--image-features-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_NORM_CSV)
    parser.add_argument("--table-features-csv", type=Path, default=DEFAULT_TABLE_FEATURES_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FUSED_FEATURES_CSV)
    parser.add_argument("--k-img", type=int, default=DEFAULT_K_IMG)
    parser.add_argument("--k-tab", type=int, default=DEFAULT_K_TAB)
    parser.add_argument("--no-balance", action="store_true", help="Skip BorderlineSMOTE.")
    args = parser.parse_args()

    output_csv = project_path(args.output_csv)
    fused = fuse_feature_selection(
        image_features_csv=project_path(args.image_features_csv),
        table_features_csv=project_path(args.table_features_csv),
        output_csv=output_csv,
        k_img=args.k_img,
        k_tab=args.k_tab,
        balance=not args.no_balance,
    )
    print(f"Wrote {len(fused)} fused feature rows to {output_csv}")


if __name__ == "__main__":
    main()
