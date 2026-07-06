from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from fkg_mm.config import DEFAULT_K_TAB, IMAGE_ID_COLUMN, SOURCE_TABLE_COLUMNS, TARGET_COLUMN


SOURCE_TABULAR_HEATMAP_COLUMNS = [
    "patient_age",
    "patient_sex",
    "diabetes_time_y",
    "insuline",
    "diabetes",
    "exam_eye",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "quality",
    TARGET_COLUMN,
]


def build_source_style_tabular(records: pd.DataFrame) -> pd.DataFrame:
    table = records[[IMAGE_ID_COLUMN, *SOURCE_TABULAR_HEATMAP_COLUMNS]].copy()
    # Chuẩn hoá data (cho label tránh nhãn 0 vì có thể hiểu nhầm là missing)
    table[TARGET_COLUMN] = table[TARGET_COLUMN].replace({1: 2, 0: 1})
    table["diabetes_time_y"] = table["diabetes_time_y"].replace({"NA": 0})
    table["insuline"] = table["insuline"].replace({"yes": 2, "no": 1, "No": 1})
    table["diabetes"] = table["diabetes"].replace({"yes": 2, "no": 1, "No": 1})
    table["quality"] = table["quality"].replace({"Adequate": 2, "Inadequate": 1})

    for column in table.columns:
        if column == IMAGE_ID_COLUMN:
            continue
        table[column] = pd.to_numeric(table[column], errors="coerce")
        table[column] = table[column].fillna(table[column].mean())

    return table.drop(columns=[IMAGE_ID_COLUMN])[SOURCE_TABULAR_HEATMAP_COLUMNS]


def extract_tabular_features(
    records: pd.DataFrame,
    output_csv: Path,
    processed_table_csv: Path,
    source_style_table_csv: Path | None = None,
    k_tab: int = DEFAULT_K_TAB,
) -> pd.DataFrame:
    table = records[SOURCE_TABLE_COLUMNS].copy()
    processed_table_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed_table_csv, index=False)

    if source_style_table_csv is not None:
        source_style_table_csv.parent.mkdir(parents=True, exist_ok=True)
        build_source_style_tabular(records).to_csv(source_style_table_csv, index=False)

    image_ids = table[IMAGE_ID_COLUMN].copy()
    table_values = table.iloc[:, 1:].copy()

    # Chuẩn hoá dữ liệu (fill NA) và encode các giá trị trong của từng cột
    for col in table_values.columns:
        encoder = LabelEncoder()
        table_values[col] = table_values[col].replace(["", " ", None], pd.NA)
        table_values[col] = table_values[col].fillna("NA")
        table_values[col] = encoder.fit_transform(table_values[col].astype(str))

    # MinMaxScaler -> Đưa data về [0,1]
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(table_values), columns=table_values.columns)
    
    # Tách features và label
    x_values = scaled.iloc[:, :-1].values
    y_values = scaled.iloc[:, -1].astype(int).values

    # Chọn K (K=9) features tốt nhất dựa trên độ tương quan giữa từng feature với label
    k = min(k_tab, x_values.shape[1])
    selected = SelectKBest(score_func=mutual_info_classif, k=k).fit_transform(x_values, y_values)

    # Chuyển data từ numpy sang dataframe rồi thêm 2 cột là image_id với label
    selected_df = pd.DataFrame(selected, columns=[str(i) for i in range(k)])
    selected_df.insert(0, IMAGE_ID_COLUMN, image_ids.values)
    selected_df[TARGET_COLUMN] = y_values

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_csv, index=False)
    return selected_df
