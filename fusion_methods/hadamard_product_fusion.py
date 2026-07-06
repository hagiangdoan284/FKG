from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from fkg_mm.config import IMAGE_ID_COLUMN, TARGET_COLUMN


def _hadamard_fusion(x_img: np.ndarray, x_tab: np.ndarray, common_dim: int = 5, alpha: float = 0.01) -> np.ndarray:
    # Chiếu 2 nguồn dữ liệu vào cùng không gian kích thước common_dim bằng Ridge projection
    proj_img = Ridge(alpha=alpha, fit_intercept=False)
    proj_tab = Ridge(alpha=alpha, fit_intercept=False)

    x_random = np.random.RandomState(42).randn(x_img.shape[0], common_dim)
    y_random = np.random.RandomState(43).randn(x_tab.shape[0], common_dim)

    proj_img.fit(x_img, x_random)
    proj_tab.fit(x_tab, y_random)

    # Lấy biểu diễn đã chiếu cho từng nguồn
    x_img_proj = proj_img.predict(x_img)
    x_tab_proj = proj_tab.predict(x_tab)

    # Chuẩn hóa vector để phép nhân Hadamard ổn định hơn
    x_img_proj = x_img_proj / (np.linalg.norm(x_img_proj, axis=1, keepdims=True) + 1e-8)
    x_tab_proj = x_tab_proj / (np.linalg.norm(x_tab_proj, axis=1, keepdims=True) + 1e-8)

    # Hadamard product + nối thêm phi tuyến tanh
    x_hadamard = x_img_proj * x_tab_proj
    return np.concatenate([x_hadamard, np.tanh(x_img_proj), np.tanh(x_tab_proj)], axis=1)


def run(
    image_features_csv: Path,
    table_features_csv: Path,
    output_csv: Path,
    common_dim: int = 5,
    alpha: float = 0.01,
    balance: bool = True,
) -> pd.DataFrame:
    # Đọc và ghép 2 nguồn feature
    image_df = pd.read_csv(image_features_csv)
    table_df = pd.read_csv(table_features_csv)

    merged = image_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
    img_label = f"{TARGET_COLUMN}_img"
    tab_label = f"{TARGET_COLUMN}_tab"
    if not merged[img_label].equals(merged[tab_label]):
        raise ValueError("Image and table labels do not match after merging by image_id.")

    # Tách nhãn + các cột feature
    y = merged[tab_label].astype(int)
    image_feature_cols = [c for c in image_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]
    table_feature_cols = [c for c in table_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]

    # Chuẩn hóa từng nguồn trước bước projection
    x_img = StandardScaler().fit_transform(merged[image_feature_cols].astype(float).values)
    x_tab = StandardScaler().fit_transform(merged[table_feature_cols].astype(float).values)

    # Sinh fused feature theo Hadamard Product Fusion
    x_fused = _hadamard_fusion(x_img, x_tab, common_dim=common_dim, alpha=alpha)
    columns = [f"hadamard_{i}" for i in range(x_fused.shape[1])]
    fused_df = pd.DataFrame(x_fused, columns=columns)
    fused_df[TARGET_COLUMN] = y.reset_index(drop=True)

    # Cân bằng lớp nếu đủ mẫu lớp thiểu số cho BorderlineSMOTE
    if balance and y.nunique() >= 2 and int(y.value_counts().min()) > 5:
        smote = BorderlineSMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
        x_resampled, y_resampled = smote.fit_resample(fused_df[columns], fused_df[TARGET_COLUMN])
        fused_df = pd.concat(
            [pd.DataFrame(x_resampled, columns=columns), pd.Series(y_resampled, name=TARGET_COLUMN)],
            axis=1,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Lưu file fused để dùng cho bước sau
    fused_df.to_csv(output_csv, index=False)
    return fused_df
