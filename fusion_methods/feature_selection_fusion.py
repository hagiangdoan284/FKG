from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from fkg_mm.config import IMAGE_ID_COLUMN, TARGET_COLUMN


def _select_features(x_values: np.ndarray, y_values: pd.Series, k: int) -> np.ndarray:
    # Chọn top-k feature theo ANOVA F-score
    k = min(k, x_values.shape[1])
    if k <= 0:
        raise ValueError("k must be positive after clipping to feature count.")
    return SelectKBest(score_func=f_classif, k=k).fit_transform(x_values, y_values)


def run(
    image_features_csv: Path,
    table_features_csv: Path,
    output_csv: Path,
    k_img: int = 7,
    k_tab: int = 9,
    balance: bool = True,
) -> pd.DataFrame:
    # Đọc 2 nguồn feature đã trích xuất sẵn: ảnh và bảng
    image_df = pd.read_csv(image_features_csv)
    table_df = pd.read_csv(table_features_csv)

    # Ghép theo image_id để tạo cặp sample đa modal
    merged = image_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
    img_label = f"{TARGET_COLUMN}_img"
    tab_label = f"{TARGET_COLUMN}_tab"

    if not merged[img_label].equals(merged[tab_label]):
        raise ValueError("Image and table labels do not match after merging by image_id.")

    # Tách nhãn và danh sách cột feature
    y = merged[tab_label].astype(int)
    image_feature_cols = [c for c in image_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]
    table_feature_cols = [c for c in table_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]

    # Chuẩn hóa từng loại dữ liệu trước khi chọn feature và fusion
    x_img = StandardScaler().fit_transform(merged[image_feature_cols].astype(float).values)
    x_tab = StandardScaler().fit_transform(merged[table_feature_cols].astype(float).values)

    # Chọn feature tốt nhất từng loại dữ liệu rồi nối lại thành vector fused
    x_img_selected = _select_features(x_img, y, k_img)
    x_tab_selected = _select_features(x_tab, y, k_tab)
    x_fused = np.concatenate([x_img_selected, x_tab_selected], axis=1)

    columns = [
        *[f"img_fs_{i}" for i in range(x_img_selected.shape[1])],
        *[f"tab_fs_{i}" for i in range(x_tab_selected.shape[1])],
    ]
    fused_df = pd.DataFrame(x_fused, columns=columns)
    fused_df[TARGET_COLUMN] = y.reset_index(drop=True)

    # Cân bằng lớp bằng BorderlineSMOTE
    if balance and y.nunique() >= 2 and int(y.value_counts().min()) > 5:
        smote = BorderlineSMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
        x_resampled, y_resampled = smote.fit_resample(fused_df[columns], fused_df[TARGET_COLUMN])
        fused_df = pd.concat(
            [
                pd.DataFrame(x_resampled, columns=columns),
                pd.Series(y_resampled, name=TARGET_COLUMN),
            ],
            axis=1,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Xuất dữ liệu fusion để chạy tiếp FIS/FRB và FKGS
    fused_df.to_csv(output_csv, index=False)
    return fused_df
