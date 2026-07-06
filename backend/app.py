from __future__ import annotations

import os
import pickle
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, send_from_directory, session, url_for
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
sys.path.insert(0, str(ROOT))

from fkg_mm.config import IMAGE_FEATURE_COLUMNS, IMAGE_ID_COLUMN, SOURCE_TABLE_COLUMNS, TARGET_COLUMN
from fkg_mm.fkgs.fkgs import calculate_a_fast, calculate_b_fast, calculate_c_fast, min_max_normalize
from fkg_mm.preprocessing.image_features import extract_one_image_features

DEFAULT_MODEL_PATH = ROOT / "outputs" / "fis" / "output" / "feature_selection" / "fuzzy_model.pkl"
DEFAULT_IMAGE_RAW_CSV = ROOT / "outputs" / "features" / "image_features.csv"
DEFAULT_IMAGE_NORM_CSV = ROOT / "outputs" / "features" / "image_fts_norm.csv"
DEFAULT_TABLE_FEATURES_CSV = ROOT / "outputs" / "features" / "table_fts.csv"
DEFAULT_TRAIN_RULE_CSV = ROOT / "outputs" / "fis" / "output" / "feature_selection" / "FRB" / "TrainDataRule.csv"
DEFAULT_RECORDS_CSV = ROOT / "outputs" / "intermediate" / "records.csv"
UPLOAD_DIR = ROOT / "outputs" / "web_uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PredictionResult:
    predicted_class: int
    predicted_text: str
    confidence: float
    feature_hits: int
    total_features: int


class FuzzyWebPredictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        image_raw_csv: Path = DEFAULT_IMAGE_RAW_CSV,
        image_norm_csv: Path = DEFAULT_IMAGE_NORM_CSV,
        table_features_csv: Path = DEFAULT_TABLE_FEATURES_CSV,
        train_rule_csv: Path = DEFAULT_TRAIN_RULE_CSV,
        records_csv: Path = DEFAULT_RECORDS_CSV,
    ) -> None:
        self.model_path = model_path
        self.image_raw_csv = image_raw_csv
        self.image_norm_csv = image_norm_csv
        self.table_features_csv = table_features_csv
        self.train_rule_csv = train_rule_csv
        self.records_csv = records_csv
        self._load_assets()

    def _load_assets(self) -> None:
        with open(self.model_path, "rb") as file:
            model_data = pickle.load(file)

        self.rule_list = np.asarray(model_data["ruleList"], dtype=int)
        self.centers = [np.asarray(center, dtype=float) for center in model_data["centers"]]
        self.feature_count = self.rule_list.shape[1] - 1

        image_raw_df = pd.read_csv(self.image_raw_csv)
        image_norm_df = pd.read_csv(self.image_norm_csv)
        table_df = pd.read_csv(self.table_features_csv)

        self.image_min = image_raw_df[IMAGE_FEATURE_COLUMNS].min()
        self.image_max = image_raw_df[IMAGE_FEATURE_COLUMNS].max()

        merged = image_norm_df.merge(table_df, how="inner", on=IMAGE_ID_COLUMN, suffixes=("_img", "_tab"))
        labels = merged[f"{TARGET_COLUMN}_tab"].astype(int)
        image_cols = [col for col in image_norm_df.columns if col not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]
        table_cols = [col for col in table_df.columns if col not in {IMAGE_ID_COLUMN, TARGET_COLUMN}]

        image_values = merged[image_cols].astype(float).to_numpy()
        table_values = merged[table_cols].astype(float).to_numpy()

        self.image_scaler = StandardScaler().fit(image_values)
        self.table_scaler = StandardScaler().fit(table_values)

        image_k = min(7, image_values.shape[1], self.feature_count)
        table_k = min(max(self.feature_count - image_k, 0), table_values.shape[1])
        if table_k == 0:
            image_k = self.feature_count

        self.image_selector = SelectKBest(score_func=f_classif, k=image_k).fit(self.image_scaler.transform(image_values), labels)
        self.table_selector = None
        if table_k > 0:
            self.table_selector = SelectKBest(score_func=f_classif, k=table_k).fit(self.table_scaler.transform(table_values), labels)

        self.table_baseline = pd.Series(table_values.mean(axis=0), index=table_cols)
        self._prepare_tabular_form_pipeline()
        self._prepare_fkg_matrices()

    def _prepare_tabular_form_pipeline(self) -> None:
        records = pd.read_csv(self.records_csv)
        table = records[SOURCE_TABLE_COLUMNS].copy()
        x_raw = table.iloc[:, 1:-1].copy()
        y_raw = table.iloc[:, -1].copy()

        age_values = pd.to_numeric(x_raw["patient_age"], errors="coerce").dropna()
        if age_values.empty:
            self.patient_age_options: list[int] = []
        else:
            min_age = int(age_values.min())
            max_age = int(age_values.max())
            self.patient_age_options = list(range(min_age, max_age + 1))

        self.form_feature_cols = x_raw.columns.tolist()
        self.feature_encoders: dict[str, LabelEncoder] = {}
        encoded_features = pd.DataFrame(index=x_raw.index)
        for col in self.form_feature_cols:
            encoder = LabelEncoder()
            values = x_raw[col].replace(["", " ", None], pd.NA).fillna("NA").astype(str)
            encoded_features[col] = encoder.fit_transform(values)
            self.feature_encoders[col] = encoder

        y_encoder = LabelEncoder()
        y_values = y_encoder.fit_transform(y_raw.replace(["", " ", None], pd.NA).fillna("NA").astype(str))

        self.form_minmax_scaler = MinMaxScaler().fit(encoded_features.to_numpy())
        x_scaled = self.form_minmax_scaler.transform(encoded_features.to_numpy())
        k_tab = min(9, x_scaled.shape[1])
        self.form_selector = SelectKBest(score_func=mutual_info_classif, k=k_tab).fit(x_scaled, y_values)

    def _prepare_fkg_matrices(self) -> None:
        train_rules_df = pd.read_csv(self.train_rule_csv).astype(float).astype(int)
        self.base_rules = train_rules_df.to_numpy(dtype=int)
        self.n_classes = len(sorted(set(self.base_rules[:, -1].tolist())))

        a_matrix = calculate_a_fast(self.base_rules)
        b_matrix = calculate_b_fast(self.base_rules, a_matrix, self.n_classes)
        c_matrix = calculate_c_fast(self.base_rules, b_matrix, self.n_classes)
        self.c_matrix = min_max_normalize(c_matrix)

        cols_per_class = self.feature_count
        self.lookup: list[dict[int, np.ndarray]] = [dict() for _ in range(self.feature_count)]
        for feature_index in range(self.feature_count):
            feature_values = self.base_rules[:, feature_index]
            for row_index, value in enumerate(feature_values):
                label_index = int(self.base_rules[row_index, -1]) - 1
                if not 0 <= label_index < self.n_classes:
                    continue
                key = int(value)
                class_values = self.lookup[feature_index].setdefault(key, np.zeros(self.n_classes, dtype=float))
                class_values[label_index] = self.c_matrix[row_index, feature_index + label_index * cols_per_class]

    def _normalize_image_features(self, feature_dict: dict[str, float]) -> pd.DataFrame:
        row = pd.DataFrame([feature_dict], columns=IMAGE_FEATURE_COLUMNS)
        denom = self.image_max - self.image_min
        denom = denom.replace(0, 1)
        normalized = (row - self.image_min) / denom
        return normalized.fillna(0.0)

    def _encode_form_value(self, col: str, raw_value: str | None) -> int:
        encoder = self.feature_encoders[col]
        text = "NA" if raw_value is None or str(raw_value).strip() == "" else str(raw_value).strip()
        classes = list(encoder.classes_)
        if text in classes:
            return int(encoder.transform([text])[0])
        if "NA" in classes:
            return int(encoder.transform(["NA"])[0])
        return 0

    def _table_features_from_form(self, form_values: dict[str, str] | None) -> np.ndarray:
        if not form_values:
            return self.table_baseline.values.astype(float)

        encoded = [self._encode_form_value(col, form_values.get(col)) for col in self.form_feature_cols]
        x_scaled = self.form_minmax_scaler.transform(np.asarray([encoded], dtype=float))
        selected = self.form_selector.transform(x_scaled).reshape(-1)
        if selected.shape[0] != self.table_baseline.shape[0]:
            return self.table_baseline.values.astype(float)
        return selected.astype(float)

    def _build_fused_vector(self, image_path: Path, form_values: dict[str, str] | None = None) -> np.ndarray:
        raw_image_features = extract_one_image_features(image_path)
        normalized_img = self._normalize_image_features(raw_image_features)

        img_scaled = self.image_scaler.transform(normalized_img.to_numpy())
        img_selected = self.image_selector.transform(img_scaled)

        parts = [img_selected]
        if self.table_selector is not None:
            table_values = self._table_features_from_form(form_values)
            table_row = pd.DataFrame([table_values], columns=self.table_baseline.index)
            tab_scaled = self.table_scaler.transform(table_row.to_numpy())
            tab_selected = self.table_selector.transform(tab_scaled)
            parts.append(tab_selected)

        fused = np.concatenate(parts, axis=1).reshape(-1)
        if fused.shape[0] != self.feature_count:
            raise ValueError(f"Fused feature size mismatch: expected {self.feature_count}, got {fused.shape[0]}")
        return fused

    def _quantize_to_rule(self, fused_vector: np.ndarray) -> np.ndarray:
        quantized = np.zeros(self.feature_count, dtype=int)
        for idx in range(self.feature_count):
            centers = self.centers[idx]
            nearest = int(np.argmin(np.abs(centers - fused_vector[idx])))
            quantized[idx] = nearest + 1
        return quantized

    def predict_image(self, image_path: Path, form_values: dict[str, str] | None = None) -> PredictionResult:
        fused = self._build_fused_vector(image_path, form_values=form_values)
        quantized = self._quantize_to_rule(fused)

        c_values = np.zeros((self.n_classes, self.feature_count), dtype=float)
        feature_hits = 0
        for feature_index in range(self.feature_count):
            key = int(quantized[feature_index])
            class_values = self.lookup[feature_index].get(key)
            if class_values is not None:
                c_values[:, feature_index] = class_values
                feature_hits += 1

        d_values = c_values.max(axis=1) + c_values.min(axis=1)
        best_index = int(np.argmax(d_values))
        tied_indexes = np.where(np.isclose(d_values, d_values[best_index]))[0]
        c_sum_values = c_values.sum(axis=1)
        if tied_indexes.size > 1:
            best_index = int(tied_indexes[np.argmax(c_sum_values[tied_indexes])])

        sorted_indexes = np.argsort(d_values)[::-1]
        second_index = int(sorted_indexes[1]) if len(sorted_indexes) > 1 else best_index
        predicted = best_index + 1
        d_best = float(d_values[best_index])
        d_second = float(d_values[second_index]) if second_index != best_index else 0.0
        denom = d_best + d_second
        confidence = float(d_best / denom) if denom > 0 else 0.0

        if self.n_classes == 2:
            predicted_text = "Không DR" if predicted == 1 else "Có DR"
        else:
            predicted_text = f"Lớp {predicted}"

        return PredictionResult(
            predicted_class=predicted,
            predicted_text=predicted_text,
            confidence=confidence,
            feature_hits=feature_hits,
            total_features=self.feature_count,
        )


app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SECRET_KEY"] = "fkgs-web-demo-secret"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
predictor = FuzzyWebPredictor()


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result: PredictionResult | None = None
    image_web_path = None
    form_values: dict[str, str] = {}

    if request.method == "POST":
        form_values = {col: request.form.get(col, "") for col in SOURCE_TABLE_COLUMNS[1:-1]}
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "Vui long chon mot tep anh."
        elif not is_allowed(file.filename):
            error = "Dinh dang anh khong ho tro. Vui long dung jpg, jpeg, png hoac bmp."
        else:
            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            saved_path = UPLOAD_DIR / unique_name
            file.save(saved_path)

            try:
                result = predictor.predict_image(saved_path, form_values=form_values)
                image_web_path = f"/uploads/{unique_name}"
                session["last_result"] = {
                    "predicted_class": result.predicted_class,
                    "predicted_text": result.predicted_text,
                    "confidence": result.confidence,
                    "feature_hits": result.feature_hits,
                    "total_features": result.total_features,
                    "image_web_path": image_web_path,
                    "form_values": form_values,
                }
                return redirect(url_for("index", done="1"))
            except Exception as exc:
                error = f"Du doan that bai: {exc}"

    if request.method == "GET" and request.args.get("done") == "1":
        payload = session.pop("last_result", None)
        if payload:
            result = PredictionResult(
                predicted_class=int(payload["predicted_class"]),
                predicted_text=str(payload.get("predicted_text", f"Lớp {int(payload['predicted_class'])}")),
                confidence=float(payload["confidence"]),
                feature_hits=int(payload["feature_hits"]),
                total_features=int(payload["total_features"]),
            )
            image_web_path = str(payload["image_web_path"])
            form_values = {k: str(v) for k, v in dict(payload.get("form_values", {})).items()}

    return render_template(
        "index.html",
        age_options=predictor.patient_age_options,
        error=error,
        result=result,
        image_web_path=image_web_path,
        form_values=form_values,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return ("Not found", 404)
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
