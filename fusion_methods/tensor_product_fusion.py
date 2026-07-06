from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from fkg_mm.config import IMAGE_ID_COLUMN, TARGET_COLUMN


def _tensor_fusion(x_img: np.ndarray, x_tab: np.ndarray, rank: int = 16) -> np.ndarray:
    # Tạo tensor outer-product cho từng mẫu để mô hình hóa tương tác mọi cặp feature ảnh-bảng
    flattened_tensors = []
    for img_vec, tab_vec in zip(x_img, x_tab):
        flattened_tensors.append(np.outer(img_vec, tab_vec).ravel())
    x_tensor = np.stack(flattened_tensors)
    rank = min(rank, x_tensor.shape[1], x_tensor.shape[0] - 1)
    if rank <= 0:
        raise ValueError("rank must be positive for TruncatedSVD.")
    # Giảm chiều tensor về rank thành phần chính
    return TruncatedSVD(n_components=rank, random_state=42).fit_transform(x_tensor)


def run(
    image_features_csv: Path,
    table_features_csv: Path,
    output_csv: Path,
    rank: int = 16,
    balance: bool = True,
) -> pd.DataFrame:
    # Đọc feature ảnh và bảng, sau đó ghép theo image_id
    image_df = pd.read_csv(image_features_csv)
    table_df = pd.read_csv(table_features_csv)

    merged = image_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
    img_label = f"{TARGET_COLUMN}_img"
    tab_label = f"{TARGET_COLUMN}_tab"
    if not merged[img_label].equals(merged[tab_label]):
        raise ValueError("Image and table labels do not match after merging by image_id.")

    # Tách nhãn và cột feature theo ảnh và bảng
    y = merged[tab_label].astype(int)
    image_feature_cols = [c for c in image_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]
    table_feature_cols = [c for c in table_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]

    # Chuẩn hóa trước khi tạo tensor để giảm lệch thang đo giữa các cột
    x_img = StandardScaler().fit_transform(merged[image_feature_cols].astype(float).values)
    x_tab = StandardScaler().fit_transform(merged[table_feature_cols].astype(float).values)

    # Sinh fused feature bằng Tensor Product Fusion
    x_fused = _tensor_fusion(x_img, x_tab, rank=rank)
    columns = [f"tensor_{i}" for i in range(x_fused.shape[1])]
    fused_df = pd.DataFrame(x_fused, columns=columns)
    fused_df[TARGET_COLUMN] = y.reset_index(drop=True)

    # Cân bằng lớp theo cùng cấu hình với các phương pháp khác
    if balance and y.nunique() >= 2 and int(y.value_counts().min()) > 5:
        smote = BorderlineSMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
        x_resampled, y_resampled = smote.fit_resample(fused_df[columns], fused_df[TARGET_COLUMN])
        fused_df = pd.concat(
            [pd.DataFrame(x_resampled, columns=columns), pd.Series(y_resampled, name=TARGET_COLUMN)],
            axis=1,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Lưu kết quả fusion
    fused_df.to_csv(output_csv, index=False)
    return fused_df
