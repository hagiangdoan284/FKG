from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fkg_mm.config import IMAGE_ID_COLUMN, SOURCE_TABLE_COLUMNS, TARGET_COLUMN


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

# Chuẩn bị bảng record đầu vào
def prepare_records(
    labels_csv: Path,
    image_dir: Path,
    records_csv: Path,
    summary_json: Path,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    # Kiểm tra xem có cột nào bị thiếu ko
    missing_columns = [col for col in SOURCE_TABLE_COLUMNS if col not in labels.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {labels_csv}: {missing_columns}")

    # Copy ra 1 bản mà chỉ lấy các cột mình dùng rồi thêm 2 cột là image_path và image_exists
    records = labels[SOURCE_TABLE_COLUMNS].copy()
    records.insert(
        1,
        "image_path",
        records[IMAGE_ID_COLUMN].map(lambda image_id: str(image_dir / f"{image_id}.jpg")),
    )
    records["image_exists"] = records["image_path"].map(lambda path: Path(path).is_file())

    records_csv.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(records_csv, index=False)

    summary = {
        "labels_csv": str(labels_csv),
        "image_dir": str(image_dir),
        "num_label_rows": int(len(labels)),
        "num_records": int(len(records)),
        "num_existing_images": int(records["image_exists"].sum()),
        "num_missing_images": int((~records["image_exists"]).sum()),
        "target": TARGET_COLUMN,
        "target_distribution": {
            str(key): int(value)
            for key, value in records[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "source_table_columns": SOURCE_TABLE_COLUMNS,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return records

# Dùng để load bản ghi cho các bước sau
def load_records(records_csv: Path, limit: int | None = None) -> pd.DataFrame:
    records = pd.read_csv(records_csv)
    # Nếu tồn tại cột image_exists thì chỉ lấy những giá trị có image_exists = True
    if "image_exists" in records.columns:
        records = records[records["image_exists"].astype(bool)].copy()
    # Lấy limit mẫu dữ liệu nếu mình có set
    if limit is not None:
        records = records.head(limit).copy()
    return records.reset_index(drop=True)
