from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fkg_mm.config import DEFAULT_K_IMG, DEFAULT_K_TAB
from fkg_mm.fis_frb import run_fis_frb
from fkg_mm.fkgs import run_fkgs_experiments

DEFAULT_IMAGE_FEATURES_NORM_CSV = Path("outputs/features/image_fts_norm.csv")
DEFAULT_TABLE_FEATURES_CSV = Path("outputs/features/table_fts.csv")
DEFAULT_RESULTS_DIR = Path("outputs/comparison")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METHOD_CLUSTER_MAP: dict[str, list[int]] = {
    # Theo source gốc diabetic_retinopathy scenarios.
    "feature_selection_fusion": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    "tensor_product_fusion": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    "filter_based_multimodal_feature_selection": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    "hadamard_product_fusion": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    "wrapper_based_multimodal_feature_selection": [5, 5, 5, 5, 2, 2, 2, 2, 2],
}


def run_single_method(
    method_name: str,
    module_file: str,
    run_kwargs: dict,
    image_features_csv: Path,
    table_features_csv: Path,
    results_dir: Path,
    ran_values: list[int],
    error_thresholds: list[float],
    turns: int,
    random_state: int | None,
) -> pd.DataFrame:
    module_path = ROOT / "fusion_methods" / module_file
    module = load_module(module_path, f"fusion_method_{method_name}")

    fused_csv = results_dir / "features" / f"{method_name}.csv"
    fused_df = module.run(
        image_features_csv=image_features_csv,
        table_features_csv=table_features_csv,
        output_csv=fused_csv,
        **run_kwargs,
    )

    n_features = fused_df.shape[1] - 1
    if method_name == "wrapper_based_multimodal_feature_selection":
        target_len = n_features + 1
        cluster = [5, 5, 5, 5] + [2] * max(0, target_len - 4)
    else:
        cluster = METHOD_CLUSTER_MAP[method_name]

    fis_dir = results_dir / "fis"
    fis_summary = run_fis_frb(
        input_csv=fused_csv,
        output_dir=fis_dir,
        file_name=method_name,
        cluster=cluster,
        random_state=random_state,
    )

    fkgs_dir = results_dir / "fkgs" / method_name
    fkgs_summary = run_fkgs_experiments(
        train_rule_csv=fis_summary["train_rule_csv"],
        test_rule_csv=fis_summary["test_rule_csv"],
        output_dir=fkgs_dir,
        ran_values=ran_values,
        error_thresholds=error_thresholds,
        turns=turns,
        random_state=random_state,
    )

    fkgs_summary["method"] = method_name
    fkgs_summary["fused_feature_count"] = n_features
    fkgs_summary["fused_rows"] = len(fused_df)
    return fkgs_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 5 fusion methods from the original FKGS paper pipeline.")
    parser.add_argument("--image-features-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_NORM_CSV)
    parser.add_argument("--table-features-csv", type=Path, default=DEFAULT_TABLE_FEATURES_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ran", type=int, nargs="+", default=[15, 20])
    parser.add_argument("--error-threshold", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers for fusion methods.")
    parser.add_argument("--no-balance", action="store_true", help="Skip BorderlineSMOTE in fusion steps.")
    args = parser.parse_args()

    image_features_csv = project_path(args.image_features_csv)
    table_features_csv = project_path(args.table_features_csv)
    results_dir = project_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        (
            "feature_selection_fusion",
            "feature_selection_fusion.py",
            {"k_img": DEFAULT_K_IMG, "k_tab": DEFAULT_K_TAB, "balance": not args.no_balance},
        ),
        (
            "tensor_product_fusion",
            "tensor_product_fusion.py",
            {"rank": DEFAULT_K_IMG + DEFAULT_K_TAB, "balance": not args.no_balance},
        ),
        (
            "hadamard_product_fusion",
            "hadamard_product_fusion.py",
            {"common_dim": 5, "alpha": 0.01, "balance": not args.no_balance},
        ),
        (
            "filter_based_multimodal_feature_selection",
            "filter_based_multimodal_feature_selection.py",
            {
                "k_img": DEFAULT_K_IMG,
                "k_tab": DEFAULT_K_TAB,
                "corr_threshold": 0.95,
                "balance": not args.no_balance,
            },
        ),
        (
            "wrapper_based_multimodal_feature_selection",
            "wrapper_based_multimodal_feature_selection.py",
            {"max_img": DEFAULT_K_IMG, "max_tab": DEFAULT_K_TAB, "min_img": 2, "min_tab": 2, "balance": not args.no_balance},
        ),
    ]

    worker_count = max(1, min(args.workers, len(methods)))
    print(f"Running {len(methods)} methods in parallel with {worker_count} workers...", flush=True)
    all_summaries_by_method: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {}
        for method_name, module_file, run_kwargs in methods:
            print(f"\n=== Queued {method_name} ===", flush=True)
            future = executor.submit(
                run_single_method,
                method_name=method_name,
                module_file=module_file,
                run_kwargs=run_kwargs,
                image_features_csv=image_features_csv,
                table_features_csv=table_features_csv,
                results_dir=results_dir,
                ran_values=args.ran,
                error_thresholds=args.error_threshold,
                turns=args.turns,
                random_state=args.random_state,
            )
            future_map[future] = method_name

        for future in as_completed(future_map):
            method_name = future_map[future]
            print(f"\n=== Running {method_name} completed ===", flush=True)
            all_summaries_by_method[method_name] = future.result()

    all_summaries = [all_summaries_by_method[name] for name, _, _ in methods]

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_csv = results_dir / "fusion_method_comparison.csv"
    combined.to_csv(combined_csv, index=False)

    best_rows = combined.sort_values("accuracy_mean", ascending=False).groupby("method", as_index=False).first()
    best_csv = results_dir / "fusion_method_best_by_method.csv"
    best_rows.to_csv(best_csv, index=False)

    ranking_csv = results_dir / "fusion_method_ranking.csv"
    best_rows.sort_values("accuracy_mean", ascending=False).to_csv(ranking_csv, index=False)

    print(f"\nWrote full comparison: {combined_csv}")
    print(f"Wrote best config per method: {best_csv}")
    print(f"Wrote ranking by accuracy_mean: {ranking_csv}")


if __name__ == "__main__":
    main()
