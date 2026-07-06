from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from fkg_mm.config import IMAGE_ID_COLUMN, TARGET_COLUMN


def _compute_feature_importance(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    # Chấm điểm từng feature độc lập: Mutual Information + RandomForest
    x_values = MinMaxScaler().fit_transform(x_values)
    mi_scores = mutual_info_classif(x_values, y_values, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_values, y_values)
    return (mi_scores + rf.feature_importances_) / 2.0


def _remove_correlated_features(x_values: np.ndarray, indices: np.ndarray, threshold: float) -> list[int]:
    # Loại bớt feature tương quan cao để giảm dư thừa
    selected: list[int] = []
    for i in indices:
        too_correlated = False
        for j in selected:
            corr = np.corrcoef(x_values[:, i], x_values[:, j])[0, 1]
            if np.isfinite(corr) and abs(corr) > threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(int(i))
    return selected


def _filter_multimodal_selection(
    x_img: np.ndarray,
    x_tab: np.ndarray,
    y_values: np.ndarray,
    k_img: int = 7,
    k_tab: int = 9,
    corr_threshold: float = 0.95,
) -> tuple[np.ndarray, list[int], list[int]]:
    # B1: chấm điểm mức quan trọng từng feature cho ảnh và bảng
    img_scores = _compute_feature_importance(x_img, y_values)
    tab_scores = _compute_feature_importance(x_tab, y_values)

    # B2: lấy danh sách ứng viên top (2*k), sau đó lọc tương quan
    sorted_img_indices = np.argsort(img_scores)[::-1]
    sorted_tab_indices = np.argsort(tab_scores)[::-1]

    candidate_img_indices = sorted_img_indices[: min(2 * k_img, x_img.shape[1])]
    candidate_tab_indices = sorted_tab_indices[: min(2 * k_tab, x_tab.shape[1])]

    final_img_indices = _remove_correlated_features(x_img, candidate_img_indices, corr_threshold)[:k_img]
    final_tab_indices = _remove_correlated_features(x_tab, candidate_tab_indices, corr_threshold)[:k_tab]

    # B3: cắt còn đúng k_img/k_tab rồi nối bảng và ảnh
    x_fused = np.concatenate([x_img[:, final_img_indices], x_tab[:, final_tab_indices]], axis=1)
    return x_fused, final_img_indices, final_tab_indices


def run(
    image_features_csv: Path,
    table_features_csv: Path,
    output_csv: Path,
    k_img: int = 7,
    k_tab: int = 9,
    corr_threshold: float = 0.95,
    balance: bool = True,
) -> pd.DataFrame:
    # Đọc và ghép dữ liệu 2 nguồn theo image_id
    image_df = pd.read_csv(image_features_csv)
    table_df = pd.read_csv(table_features_csv)

    merged = image_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
    img_label = f"{TARGET_COLUMN}_img"
    tab_label = f"{TARGET_COLUMN}_tab"
    if not merged[img_label].equals(merged[tab_label]):
        raise ValueError("Image and table labels do not match after merging by image_id.")

    # Tách nhãn + ma trận feature
    y = merged[tab_label].astype(int).to_numpy()
    image_feature_cols = [c for c in image_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]
    table_feature_cols = [c for c in table_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]

    x_img = merged[image_feature_cols].astype(float).to_numpy()
    x_tab = merged[table_feature_cols].astype(float).to_numpy()

    # Chạy thuật toán Filter-based multi-modal feature selection
    x_fused, selected_img, selected_tab = _filter_multimodal_selection(
        x_img,
        x_tab,
        y,
        k_img=k_img,
        k_tab=k_tab,
        corr_threshold=corr_threshold,
    )

    columns = [
        *[f"img_filter_{idx}" for idx in selected_img],
        *[f"tab_filter_{idx}" for idx in selected_tab],
    ]
    fused_df = pd.DataFrame(x_fused, columns=columns)
    fused_df[TARGET_COLUMN] = y

    # Cân bằng lớp để đồng nhất với các pipeline fusion khác
    if balance and len(np.unique(y)) >= 2 and int(pd.Series(y).value_counts().min()) > 5:
        smote = BorderlineSMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
        x_resampled, y_resampled = smote.fit_resample(fused_df[columns], fused_df[TARGET_COLUMN])
        fused_df = pd.concat(
            [pd.DataFrame(x_resampled, columns=columns), pd.Series(y_resampled, name=TARGET_COLUMN)],
            axis=1,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Lưu dữ liệu fused
    fused_df.to_csv(output_csv, index=False)
    return fused_df
