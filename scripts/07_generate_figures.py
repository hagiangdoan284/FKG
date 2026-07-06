
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TABLE_SOURCE_CSV = Path("outputs/intermediate/data_process_tabular.csv")
DEFAULT_IMAGE_FEATURES_CSV = Path("outputs/features/image_features.csv")
DEFAULT_FUSED_FEATURES_CSV = Path("outputs/features/data_process_fusion_named.csv")
DEFAULT_FKGS_SUMMARY_CSV = Path("outputs/fkgs/feature_selection/fkgs_summary.csv")
DEFAULT_FKGS_TURN_RESULTS_CSV = Path("outputs/fkgs/feature_selection/fkgs_turn_results.csv")
DEFAULT_FKGS_OUTPUT_DIR = Path("outputs/fkgs/feature_selection")
DEFAULT_FUSION_COMPARISON_CSV = Path("outputs/comparison/fusion_method_best_by_method.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures")
DEFAULT_DPI = 200


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


os.environ.setdefault("MPLCONFIGDIR", str(project_path(Path("outputs/.matplotlib"))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def config_labels(df: pd.DataFrame) -> list[str]:
    return [f"ran={int(row.ran)}, e={row.error_threshold}" for row in df.itertuples()]


def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for column in encoded.columns:
        if pd.api.types.is_numeric_dtype(encoded[column]):
            encoded[column] = encoded[column].fillna(encoded[column].median())
        else:
            encoded[column] = pd.factorize(encoded[column].fillna("NA").astype(str))[0]
    return encoded


def plot_annotated_heatmap(
    corr: pd.DataFrame,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig_size = max(8, len(corr.columns) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    for row_index in range(len(corr.index)):
        for col_index in range(len(corr.columns)):
            value = corr.iloc[row_index, col_index]
            text_color = "white" if abs(value) > 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_tabular_correlation_heatmap(table_source_csv: Path, output_dir: Path, dpi: int) -> None:
    print("Drawing tabular Pearson correlation heatmap...", flush=True)
    df = pd.read_csv(table_source_csv).drop(columns=["image_id", "diabetic_retinopathy"], errors="ignore")
    corr = encode_for_correlation(df).corr(method="pearson")
    plot_annotated_heatmap(
        corr,
        "Pearson Correlation Heatmap - Tabular Features",
        output_dir / "tabular_pearson_correlation_heatmap.png",
        dpi,
    )


def plot_image_correlation_heatmap(image_features_csv: Path, output_dir: Path, dpi: int) -> None:
    print("Drawing image Pearson correlation heatmap...", flush=True)
    df = pd.read_csv(image_features_csv).drop(columns=["image_id", "diabetic_retinopathy"], errors="ignore")
    corr = df.corr(method="pearson", numeric_only=True)
    plot_annotated_heatmap(
        corr,
        "Pearson Correlation Heatmap - Image Features",
        output_dir / "image_pearson_correlation_heatmap.png",
        dpi,
    )


SOURCE_IMAGE_FEATURE_NAMES = {
    "glcm_contrast": "Contrast Feature",
    "glcm_dissimilarity": "Dissimilarity Feature",
    "glcm_homogeneity": "Homogeneity Feature",
    "glcm_energy": "Energy Feature",
    "glcm_correlation": "Correlation Feature",
    "glcm_asm": "ASM Feature",
    "stat_mean": "Mean Feature",
    "stat_variance": "Variance Feature",
    "stat_std": "Standard Deviation Feature",
    "stat_rms": "RMS Feature",
}


def min_max_normalize_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in feature_columns:
        col_min = normalized[column].min()
        col_max = normalized[column].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
            normalized[column] = 0.0
        else:
            normalized[column] = (normalized[column] - col_min) / (col_max - col_min)
    return normalized


def build_named_fusion_features(
    table_source_csv: Path,
    image_features_csv: Path,
    fused_features_csv: Path,
) -> pd.DataFrame:
    image_df = pd.read_csv(image_features_csv).rename(columns=SOURCE_IMAGE_FEATURE_NAMES)
    image_feature_columns = list(SOURCE_IMAGE_FEATURE_NAMES.values())
    image_df = min_max_normalize_feature_columns(image_df, image_feature_columns)

    table_df = pd.read_csv(table_source_csv)
    records_df = pd.read_csv(project_path(Path("outputs/intermediate/records.csv")))[["image_id"]].copy()
    records_df["row_order"] = np.arange(len(records_df))

    table_with_id = pd.concat([records_df, table_df.reset_index(drop=True)], axis=1)
    table_with_id = table_with_id.drop(columns=["diabetic_retinopathy"])
    image_with_label = image_df[["image_id", *image_feature_columns, "diabetic_retinopathy"]]
    fused = table_with_id.merge(image_with_label, how="inner", on="image_id").sort_values("row_order")
    fused = fused.drop(columns=["image_id", "row_order"])
    fused = fused[[column for column in fused.columns if column != "diabetic_retinopathy"] + ["diabetic_retinopathy"]]

    fused_features_csv.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(fused_features_csv, index=False)
    return fused


def plot_random_forest_feature_importance(
    table_source_csv: Path,
    image_features_csv: Path,
    fused_features_csv: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    print("Drawing Random Forest feature importance...", flush=True)
    df = (
        pd.read_csv(fused_features_csv)
        if fused_features_csv.exists()
        else build_named_fusion_features(table_source_csv, image_features_csv, fused_features_csv)
    )
    if any(column.startswith(("img_fs_", "tab_fs_")) for column in df.columns):
        df = build_named_fusion_features(table_source_csv, image_features_csv, fused_features_csv)

    x_values = df.iloc[:, :-1]
    y_values = df.iloc[:, -1]

    model = RandomForestClassifier(random_state=42)
    model.fit(x_values, y_values)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = x_values.columns.to_numpy()

    fig_height = max(6, len(importances) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(range(len(importances)), importances[indices], color="skyblue")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(feature_names[indices])
    ax.invert_yaxis()
    ax.set_title("Feature Importances (using Random Forest)")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "random_forest_feature_importance.png", dpi=dpi)
    fig.savefig(output_dir / "Importance.png", dpi=dpi)
    plt.close(fig)

    pd.DataFrame(
        {
            "feature": feature_names[indices],
            "importance": importances[indices],
        }
    ).to_csv(output_dir / "random_forest_feature_importance.csv", index=False)


def plot_metric_comparison(summary_csv: Path, output_dir: Path, dpi: int) -> None:
    print("Drawing FKGS metric comparison...", flush=True)
    df = pd.read_csv(summary_csv)
    labels = config_labels(df)
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df["accuracy_mean"], width, label="Accuracy")
    ax.bar(x, df["precision_mean"], width, label="Precision")
    ax.bar(x + width, df["recall_mean"], width, label="Recall")
    ax.set_title("FKGS Metric Comparison")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fkgs_metric_comparison.png", dpi=dpi)
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=dpi)
    plt.close(fig)


def plot_time_comparison(summary_csv: Path, output_dir: Path, dpi: int) -> None:
    print("Drawing FKGS runtime comparison...", flush=True)
    df = pd.read_csv(summary_csv)
    labels = config_labels(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, df["total_seconds_mean"], label="Total seconds")
    ax.set_title("FKGS Runtime Comparison")
    ax.set_ylabel("Mean seconds")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "time_comparison.png", dpi=dpi)
    plt.close(fig)


def best_prediction_csv(turn_results_csv: Path, fkgs_output_dir: Path) -> Path:
    turns = pd.read_csv(turn_results_csv)
    best = turns.sort_values("accuracy", ascending=False).iloc[0]
    ran = int(best["ran"])
    error_threshold = str(best["error_threshold"]).replace(".", "_")
    turn = int(best["turn"])
    return fkgs_output_dir / f"ran_{ran}_e_{error_threshold}" / f"turn_{turn}" / "predictions.csv"


def plot_confusion_matrix(turn_results_csv: Path, fkgs_output_dir: Path, output_dir: Path, dpi: int) -> None:
    print("Drawing best FKGS confusion matrix...", flush=True)
    prediction_csv = best_prediction_csv(turn_results_csv, fkgs_output_dir)
    predictions = pd.read_csv(prediction_csv)
    y_true = predictions["y_true"].astype(int)
    y_pred = predictions["y_pred"].astype(int)
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Best FKGS Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_best_model.png", dpi=dpi)
    plt.close(fig)


def plot_fusion_method_comparison(comparison_csv: Path, output_dir: Path, dpi: int) -> None:
    if not comparison_csv.exists():
        print(f"Skipping fusion method comparison (missing file): {comparison_csv}", flush=True)
        return

    print("Drawing fusion method comparison (accuracy only)...", flush=True)
    df = pd.read_csv(comparison_csv)
    required_columns = {"method", "accuracy_mean"}
    if not required_columns.issubset(df.columns):
        print(
            f"Skipping fusion method comparison (required columns not found): {sorted(required_columns)}",
            flush=True,
        )
        return

    ranked = df[["method", "accuracy_mean"]].dropna().sort_values("accuracy_mean", ascending=False)
    if ranked.empty:
        print("Skipping fusion method comparison (no valid rows).", flush=True)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ranked["method"], ranked["accuracy_mean"], color="steelblue")
    ax.set_title("Fusion Method Comparison by Accuracy")
    ax.set_ylabel("Accuracy mean")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, ranked["accuracy_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "fusion_method_accuracy_comparison.png", dpi=dpi)
    plt.close(fig)


def generate_figures(
    table_source_csv: Path,
    image_features_csv: Path,
    fused_features_csv: Path,
    summary_csv: Path,
    turn_results_csv: Path,
    fkgs_output_dir: Path,
    fusion_comparison_csv: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    project_path(Path("outputs/.matplotlib")).mkdir(parents=True, exist_ok=True)
    plot_tabular_correlation_heatmap(table_source_csv, output_dir, dpi)
    plot_image_correlation_heatmap(image_features_csv, output_dir, dpi)
    plot_random_forest_feature_importance(table_source_csv, image_features_csv, fused_features_csv, output_dir, dpi)
    plot_metric_comparison(summary_csv, output_dir, dpi)
    plot_time_comparison(summary_csv, output_dir, dpi)
    plot_confusion_matrix(turn_results_csv, fkgs_output_dir, output_dir, dpi)
    plot_fusion_method_comparison(fusion_comparison_csv, output_dir, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from FKGS outputs.")
    parser.add_argument("--table-source-csv", type=Path, default=DEFAULT_TABLE_SOURCE_CSV)
    parser.add_argument("--image-features-csv", type=Path, default=DEFAULT_IMAGE_FEATURES_CSV)
    parser.add_argument("--fused-features-csv", type=Path, default=DEFAULT_FUSED_FEATURES_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_FKGS_SUMMARY_CSV)
    parser.add_argument("--turn-results-csv", type=Path, default=DEFAULT_FKGS_TURN_RESULTS_CSV)
    parser.add_argument("--fkgs-output-dir", type=Path, default=DEFAULT_FKGS_OUTPUT_DIR)
    parser.add_argument("--fusion-comparison-csv", type=Path, default=DEFAULT_FUSION_COMPARISON_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    generate_figures(
        table_source_csv=project_path(args.table_source_csv),
        image_features_csv=project_path(args.image_features_csv),
        fused_features_csv=project_path(args.fused_features_csv),
        summary_csv=project_path(args.summary_csv),
        turn_results_csv=project_path(args.turn_results_csv),
        fkgs_output_dir=project_path(args.fkgs_output_dir),
        fusion_comparison_csv=project_path(args.fusion_comparison_csv),
        output_dir=project_path(args.output_dir),
        dpi=args.dpi,
    )
    print(f"Wrote figures to {project_path(args.output_dir)}")


if __name__ == "__main__":
    main()
