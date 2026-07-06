from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from fkg_mm.config import IMAGE_ID_COLUMN, TARGET_COLUMN


def _evaluate_feature_set(x_values: np.ndarray, y_values: np.ndarray, cv: int = 5) -> float:
    # Đánh giá một tập feature bằng CV accuracy (wrapper objective)
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, x_values, y_values, cv=cv, scoring="accuracy")
    return float(scores.mean())


def _add_best_feature(
    x_values: np.ndarray,
    selected_indices: list[int],
    y_values: np.ndarray,
    remaining_indices: list[int],
) -> tuple[int | None, float]:
    # Tìm feature thêm vào giúp tăng điểm tốt nhất ở bước hiện tại
    best_score = -np.inf
    best_feature = None
    for idx in remaining_indices:
        temp_indices = selected_indices + [idx]
        score = _evaluate_feature_set(x_values[:, temp_indices], y_values)
        if score > best_score:
            best_score = score
            best_feature = idx
    return best_feature, float(best_score)


def _wrapper_multimodal_selection(
    x_img: np.ndarray,
    x_tab: np.ndarray,
    y_values: np.ndarray,
    max_img: int = 7,
    max_tab: int = 9,
    min_img: int = 2,
    min_tab: int = 2,
) -> tuple[np.ndarray, list[int], list[int]]:
    # Khởi tạo tập feature đã chọn cho ảnh và bảng
    selected_img: list[int] = []
    selected_tab: list[int] = []
    best_score = -np.inf

    img_indices = list(range(x_img.shape[1]))
    tab_indices = list(range(x_tab.shape[1]))

    # B1: ép chọn tối thiểu min_img/min_tab từ từng nguồn
    for _ in range(min(min_img, len(img_indices))):
        best_feature, _ = _add_best_feature(
            x_img,
            selected_img,
            y_values,
            [i for i in img_indices if i not in selected_img],
        )
        if best_feature is not None:
            selected_img.append(best_feature)

    for _ in range(min(min_tab, len(tab_indices))):
        best_feature, _ = _add_best_feature(
            x_tab,
            selected_tab,
            y_values,
            [i for i in tab_indices if i not in selected_tab],
        )
        if best_feature is not None:
            selected_tab.append(best_feature)

    # B2: Sequential Forward Selection giữa 2 nguồn cho đến max hoặc không còn cải thiện
    while len(selected_img) < max_img or len(selected_tab) < max_tab:
        best_new_score = -np.inf
        best_new_feature = None
        best_modality = None

        if len(selected_img) < max_img:
            for i in [k for k in img_indices if k not in selected_img]:
                fused = np.concatenate([x_img[:, selected_img + [i]], x_tab[:, selected_tab]], axis=1)
                score = _evaluate_feature_set(fused, y_values)
                if score > best_new_score:
                    best_new_score = score
                    best_new_feature = i
                    best_modality = "img"

        if len(selected_tab) < max_tab:
            for j in [k for k in tab_indices if k not in selected_tab]:
                fused = np.concatenate([x_img[:, selected_img], x_tab[:, selected_tab + [j]]], axis=1)
                score = _evaluate_feature_set(fused, y_values)
                if score > best_new_score:
                    best_new_score = score
                    best_new_feature = j
                    best_modality = "tab"

        # Nếu không cải thiện điểm thì dừng (early stop)
        if best_new_score > best_score and best_new_feature is not None and best_modality is not None:
            best_score = best_new_score
            if best_modality == "img":
                selected_img.append(int(best_new_feature))
            else:
                selected_tab.append(int(best_new_feature))
        else:
            break

    # Ghép feature đã chọn thành biểu diễn fused cuối cùng
    x_fused = np.concatenate([x_img[:, selected_img], x_tab[:, selected_tab]], axis=1)
    return x_fused, selected_img, selected_tab


def run(
    image_features_csv: Path,
    table_features_csv: Path,
    output_csv: Path,
    max_img: int = 7,
    max_tab: int = 9,
    min_img: int = 2,
    min_tab: int = 2,
    balance: bool = True,
) -> pd.DataFrame:
    # Đọc và ghép dữ liệu từ ảnh và bảng
    image_df = pd.read_csv(image_features_csv)
    table_df = pd.read_csv(table_features_csv)

    merged = image_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
    img_label = f"{TARGET_COLUMN}_img"
    tab_label = f"{TARGET_COLUMN}_tab"
    if not merged[img_label].equals(merged[tab_label]):
        raise ValueError("Image and table labels do not match after merging by image_id.")

    # Tách nhãn + feature matrix từng nguồn
    y = merged[tab_label].astype(int).to_numpy()
    image_feature_cols = [c for c in image_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]
    table_feature_cols = [c for c in table_df.columns if c not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]

    x_img = merged[image_feature_cols].astype(float).to_numpy()
    x_tab = merged[table_feature_cols].astype(float).to_numpy()

    # Chạy Wrapper-based multi-modal feature selection
    x_fused, selected_img, selected_tab = _wrapper_multimodal_selection(
        x_img,
        x_tab,
        y,
        max_img=max_img,
        max_tab=max_tab,
        min_img=min_img,
        min_tab=min_tab,
    )

    columns = [
        *[f"img_wrapper_{idx}" for idx in selected_img],
        *[f"tab_wrapper_{idx}" for idx in selected_tab],
    ]
    fused_df = pd.DataFrame(x_fused, columns=columns)
    fused_df[TARGET_COLUMN] = y

    # Cân bằng lớp bằng BorderlineSMOTE nếu điều kiện dữ liệu cho phép
    if balance and len(np.unique(y)) >= 2 and int(pd.Series(y).value_counts().min()) > 5:
        smote = BorderlineSMOTE(random_state=42, sampling_strategy="auto", k_neighbors=5)
        x_resampled, y_resampled = smote.fit_resample(fused_df[columns], fused_df[TARGET_COLUMN])
        fused_df = pd.concat(
            [pd.DataFrame(x_resampled, columns=columns), pd.Series(y_resampled, name=TARGET_COLUMN)],
            axis=1,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Lưu file fused
    fused_df.to_csv(output_csv, index=False)
    return fused_df
