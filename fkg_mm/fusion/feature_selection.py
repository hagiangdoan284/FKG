from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from fkg_mm.config import DEFAULT_K_IMG, DEFAULT_K_TAB, IMAGE_ID_COLUMN, TARGET_COLUMN


def select_features(x_values: np.ndarray, y_values: pd.Series, k: int) -> np.ndarray:
    k = min(k, x_values.shape[1])
    if k <= 0:
        raise ValueError("k must be positive after clipping to feature count.")

    return SelectKBest(score_func=f_classif, k=k).fit_transform(x_values, y_values) # hihi SelectKBest kết hợp f_classif chỗ này


def fuse_feature_selection(
    image_features_csv: Path,
    table_features_csv: Path,
    output_csv: Path,
    k_img: int = DEFAULT_K_IMG,
    k_tab: int = DEFAULT_K_TAB,
    balance: bool = True,
) -> pd.DataFrame:
    image_df = pd.read_csv(image_features_csv)
    table_df = pd.read_csv(table_features_csv)

    # Merge 2 DataFrame (image với table) lại với nhau theo image_id
    merged = image_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
    img_label = f"{TARGET_COLUMN}_img"
    tab_label = f"{TARGET_COLUMN}_tab"

    if img_label not in merged.columns or tab_label not in merged.columns:
        raise ValueError("Both image and table feature files must contain the target column.")
    if not merged[img_label].equals(merged[tab_label]):
        raise ValueError("Image and table labels do not match after merging by image_id.")

    label = merged[tab_label].astype(int)

    # Lấy các features của image và table
    image_feature_cols = [
        col for col in image_df.columns if col not in {IMAGE_ID_COLUMN, TARGET_COLUMN}
    ]
    table_feature_cols = [
        col for col in table_df.columns if col not in {IMAGE_ID_COLUMN, TARGET_COLUMN}
    ]

    # Lấy giá trị các features của image với table rồi chuẩn hoá StandardScaler (đưa 2 giá trị về cùng thang đo)
    f_img = merged[image_feature_cols].astype(float).values
    f_tab = merged[table_feature_cols].astype(float).values

    f_img = StandardScaler().fit_transform(f_img)
    f_tab = StandardScaler().fit_transform(f_tab)

    # Chọn lấy K features tốt nhất (image = 7, table = 9) rồi fuse lại -> 16 features
    f_img_selected = select_features(f_img, label, k_img)
    f_tab_selected = select_features(f_tab, label, k_tab)
    fused = np.concatenate([f_img_selected, f_tab_selected], axis=1)

    # Tạo tên features
    fused_columns = [
        *[f"img_fs_{i}" for i in range(f_img_selected.shape[1])],
        *[f"tab_fs_{i}" for i in range(f_tab_selected.shape[1])],
    ]

    # Tạo DataFrame và thêm label
    fused_df = pd.DataFrame(fused, columns=fused_columns)
    fused_df[TARGET_COLUMN] = label.reset_index(drop=True)

    if balance and label.nunique() >= 2:
        class_counts = label.value_counts()
        min_class_count = int(class_counts.min())
        
        # Chỉ chạy SMOTE nếu class thiểu số có tối thiểu 5 mẫu vì BorderLineSMOTE dùng k_neighbors = 5
        # Nếu class thiểu số ít hơn hoặc = 5 thì ko đủ hàng xóm để nội suy -> lỗi
        if min_class_count > 5:
            smote = BorderlineSMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
            x_resampled, y_resampled = smote.fit_resample(fused_df[fused_columns], fused_df[TARGET_COLUMN])
            fused_df = pd.concat(
                [
                    pd.DataFrame(x_resampled, columns=fused_columns),
                    pd.Series(y_resampled, name=TARGET_COLUMN),
                ],
                axis=1,
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(output_csv, index=False)
    return fused_df
