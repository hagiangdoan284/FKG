from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm


@dataclass
class FkgsRunResult:
    # Lưu kết quả của 1 lần chạy FKGS để cuối cùng gom lại thành summary
    ran: int
    error_threshold: float
    turn: int
    sampled_rule_rows: int
    sampling_seconds: float
    train_seconds: float
    test_seconds: float
    total_seconds: float
    accuracy: float
    precision: float
    recall: float


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    # Chuẩn hoá ma trận C về [0, 1] giống bước normalize trước khi FISA
    min_values = np.min(values, axis=0)
    max_values = np.max(values, axis=0)
    denominator = max_values - min_values
    denominator[denominator == 0] = 1
    return (values - min_values) / denominator


def rule_similarity(rule_a: list[int], rule_b: list[int]) -> float:
    # Hai rule khác nhãn thì coi như không tương tự để không loại nhầm rule của class khác
    if rule_a[-1] != rule_b[-1]:
        return -1
    matching = sum(1 for idx in range(len(rule_a) - 1) if rule_a[idx] == rule_b[idx])
    return matching / len(rule_a)


def has_similar_sampled_rule(
    rule: list[int],
    sampled_by_label: dict[int, list[np.ndarray]],
    similarity_threshold: float,
) -> bool:
    # So rule mới với các rule đã chọn cùng label bằng numpy cho nhanh hơn vòng lặp Python
    same_label_rules = sampled_by_label.get(rule[-1])
    if not same_label_rules:
        return False

    existing_rules = np.vstack(same_label_rules)
    rule_values = np.asarray(rule)
    feature_matches = (existing_rules[:, :-1] == rule_values[:-1]).sum(axis=1)
    similarities = feature_matches / len(rule)
    return bool(np.any(similarities >= similarity_threshold))


def sample_rules(
    rules: list[list[int]],
    ran: int,
    error_threshold: float,
    neighbor_window: int = 2,
    rng: random.Random | None = None,
) -> list[list[int]]:
    # Source gốc chọn random trong while nên có thể thử lại mãi các rule đã bị loại
    # Ở đây shuffle trước danh sách index, rồi duyệt có kiểm soát để tránh treo
    rng = rng or random.Random()
    total_rules = len(rules)
    target_size = int(np.ceil(total_rules * ran / 100))
    sampled: list[list[int]] = []
    sampled_by_label: dict[int, list[np.ndarray]] = {}
    attempted_indices: set[int] = set()
    candidate_order = list(range(total_rules))
    rng.shuffle(candidate_order)
    cursor = 0
    similarity_threshold = 1 - error_threshold

    with tqdm(total=target_size, desc=f"sample rules ran={ran} e={error_threshold}") as progress:
        while len(sampled) < target_size and cursor < total_rules:
            index = candidate_order[cursor]
            cursor += 1
            if index in attempted_indices:
                continue
            attempted_indices.add(index)

            candidates = [index]
            # Xét thêm các rule gần vị trí hiện tại trong rule base
            for candidate_idx in range(index - neighbor_window, index + neighbor_window + 1):
                if 0 <= candidate_idx < total_rules and candidate_idx not in attempted_indices:
                    if rule_similarity(rules[candidate_idx], rules[index]) < similarity_threshold:
                        candidates.append(candidate_idx)
                        attempted_indices.add(candidate_idx)

            for candidate_idx in candidates:
                candidate_rule = rules[candidate_idx]
                if has_similar_sampled_rule(candidate_rule, sampled_by_label, similarity_threshold):
                    continue

                sampled.append(candidate_rule)
                sampled_by_label.setdefault(candidate_rule[-1], []).append(np.asarray(candidate_rule))
                progress.update(1)
                if len(sampled) >= target_size:
                    break

        if len(sampled) < target_size:
            # Nếu strict sampling không đủ rule khác nhau, fill phần còn thiếu để vẫn đạt đúng ran%
            for candidate_idx in candidate_order:
                if len(sampled) >= target_size:
                    break
                candidate_rule = rules[candidate_idx]
                if candidate_rule in sampled:
                    continue
                sampled.append(candidate_rule)
                progress.update(1)

    return sampled


def load_rule_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).astype(float).astype(int)


def predict_with_fisa_lookup(
    base: list[list[int]],
    c_matrix: np.ndarray,
    test_rules: list[list[int]],
    n_classes: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Tính D theo từng sample test bằng cách tra C theo các key feature+label trùng với sample
    base_values = np.asarray(base, dtype=int)
    c_values = np.asarray(c_matrix, dtype=float)
    row_count = base_values.shape[0]
    feature_count = base_values.shape[1] - 1
    cols_per_class = feature_count
    if c_values.shape != (row_count, n_classes * cols_per_class):
        raise ValueError("c_matrix must have shape (n_rules, n_classes * n_features) for rule-based FISA mode")

    lookup: list[dict[int, np.ndarray]] = [dict() for _ in range(feature_count)]
    for feature_index in range(feature_count):
        feature_values = base_values[:, feature_index]
        labels = base_values[:, feature_count].astype(int)
        for row_index in range(row_count):
            label_index = labels[row_index] - 1
            if not 0 <= label_index < n_classes:
                continue
            key = int(feature_values[row_index])
            class_values = lookup[feature_index].setdefault(key, np.zeros(n_classes, dtype=float))
            class_values[label_index] = c_values[row_index, label_index * cols_per_class + feature_index]

    predictions = np.zeros(len(test_rules))
    ranks = np.zeros(len(test_rules))

    for idx, rule in enumerate(tqdm(test_rules, desc=desc)):
        class_feature_scores = np.zeros((n_classes, feature_count), dtype=float)
        for feature_index in range(feature_count):
            key = int(rule[feature_index])
            class_values = lookup[feature_index].get(key)
            if class_values is not None:
                class_feature_scores[:, feature_index] = class_values

        d_values = class_feature_scores.max(axis=1) + class_feature_scores.min(axis=1)
        d_sum = float(d_values.sum())
        best_index = int(np.argmax(d_values))
        predictions[idx] = best_index + 1
        ranks[idx] = d_values[best_index] / d_sum if d_sum > 0 else 0.0

    return predictions, ranks


def calculate_a_fast(base_values: np.ndarray) -> np.ndarray:
    # A: với mỗi rule và mỗi cặp feature trong X, đếm tần suất có cùng 2 giá trị đó
    row_count = base_values.shape[0]
    feature_count = base_values.shape[1] - 1
    pairs = list(combinations(range(feature_count), 2))
    a_matrix = np.zeros((row_count, len(pairs)), dtype=float)

    for col_index, pair in enumerate(pairs):
        _, inverse, counts = np.unique(
            base_values[:, pair],
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        a_matrix[:, col_index] = counts[inverse] / row_count

    return a_matrix


def calculate_b_fast(base_values: np.ndarray, a_matrix: np.ndarray, n_classes: int) -> np.ndarray:
    # B: tổng A liên quan tới từng feature, nhân với tần suất feature đó đi cùng từng nhãn
    row_count = base_values.shape[0]
    feature_count = base_values.shape[1] - 1
    cols_per_class = feature_count
    b_matrix = np.zeros((row_count, n_classes * cols_per_class), dtype=float)
    labels = base_values[:, feature_count].astype(int)

    a_row_sum = a_matrix.sum(axis=1)

    for feature_index in range(feature_count):
        _, inverse, counts = np.unique(
            base_values[:, [feature_index, feature_count]],
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        pair_freq = counts[inverse] / row_count
        base_term = a_row_sum * pair_freq
        for class_index in range(n_classes):
            class_label = class_index + 1
            col_index = class_index * cols_per_class + feature_index # B của feature_index cho class_label
            b_matrix[:, col_index] = np.where(labels == class_label, base_term, 0.0)

    return b_matrix


def calculate_c_fast(base_values: np.ndarray, b_matrix: np.ndarray, n_classes: int) -> np.ndarray:
    # C theo từng rule: cộng B của các rule có cùng giá trị feature và cùng label
    row_count = base_values.shape[0]
    feature_count = base_values.shape[1] - 1
    cols_per_class = feature_count
    c_matrix = np.zeros((row_count, n_classes * cols_per_class), dtype=float)
    labels = base_values[:, feature_count].astype(int)

    for feature_index in range(feature_count):
        feature_values = base_values[:, feature_index].astype(int)
        for class_index in range(n_classes):
            class_label = class_index + 1
            b_col = class_index * cols_per_class + feature_index
            sum_by_value: dict[int, float] = {}

            for row_index in range(row_count):
                if labels[row_index] != class_label:
                    continue
                value_key = int(feature_values[row_index])
                sum_by_value[value_key] = sum_by_value.get(value_key, 0.0) + float(b_matrix[row_index, b_col])

            c_col = class_index * cols_per_class + feature_index
            for row_index in range(row_count):
                value_key = int(feature_values[row_index])
                c_matrix[row_index, c_col] = sum_by_value.get(value_key, 0.0)

    return c_matrix


def calculate_fkgs_matrices_fast(
    sampled_rules: list[list[int]],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Tính A/B/C bằng NumPy
    base_values = np.asarray(sampled_rules, dtype=int)

    print(f"  calculateA fast on {len(sampled_rules)} sampled rules...", flush=True)
    start_step = time.time()
    a_matrix = calculate_a_fast(base_values)
    print(f"  calculateA fast done in {time.time() - start_step:.2f}s", flush=True)

    print("  calculateB fast...", flush=True)
    start_step = time.time()
    b_matrix = calculate_b_fast(base_values, a_matrix, n_classes)
    print(f"  calculateB fast done in {time.time() - start_step:.2f}s", flush=True)

    print("  calculateC fast...", flush=True)
    start_step = time.time()
    c_matrix = calculate_c_fast(base_values, b_matrix, n_classes)
    print(f"  calculateC fast done in {time.time() - start_step:.2f}s", flush=True)

    return a_matrix, b_matrix, c_matrix


def run_fkgs_once(
    train_rules: pd.DataFrame,
    test_rules: pd.DataFrame,
    ran: int,
    error_threshold: float,
    turn: int,
    output_dir: Path,
    random_state: int | None = None,
) -> FkgsRunResult:
    # Chạy 1 turn FKGS cho một cặp tham số ran/e
    rng = random.Random(None if random_state is None else random_state + turn)
    train_list = train_rules.values.tolist()
    test_list = test_rules.values.tolist()
    labels = sorted(set(train_rules.iloc[:, -1].tolist()) | set(test_rules.iloc[:, -1].tolist()))
    n_classes = len(labels)

    start_sampling = time.time()
    # Bước 1: lấy ran% rule base sau khi lọc bớt rule quá giống nhau theo error_threshold
    sampled_rules = sample_rules(train_list, ran=ran, error_threshold=error_threshold, rng=rng)
    sampling_seconds = time.time() - start_sampling

    start_train = time.time()
    # Bước 2: xây các ma trận FKGS A/B/C bằng engine được chọn
    _, _, c_matrix = calculate_fkgs_matrices_fast(sampled_rules, n_classes)
    c_matrix = min_max_normalize(c_matrix)
    train_seconds = time.time() - start_train

    start_test = time.time()
    # Bước 3: dự đoán tập test bằng FISA trực tiếp hoặc lookup engine
    print(f"  FISA test on {len(test_list)} test rules with lookup engine...", flush=True)
    predictions, ranks = predict_with_fisa_lookup(
        sampled_rules,
        c_matrix,
        test_list,
        n_classes,
        desc=f"FKGS ran={ran} e={error_threshold} turn={turn}",
    )
    y_true = test_rules.iloc[:, -1].to_numpy()
    test_seconds = time.time() - start_test

    accuracy = accuracy_score(y_true, predictions)
    # Bước 4: tính metric
    precision = precision_score(y_true, predictions, average="macro", zero_division=0)
    recall = recall_score(y_true, predictions, average="macro", zero_division=0)

    config_dir = output_dir / f"ran_{ran}_e_{str(error_threshold).replace('.', '_')}" / f"turn_{turn}"
    config_dir.mkdir(parents=True, exist_ok=True)
    # Lưu prediction từng turn để vẽ confusion matrix và debug sau này
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": predictions.astype(int),
            "rank": ranks,
        }
    ).to_csv(config_dir / "predictions.csv", index=False)

    return FkgsRunResult(
        ran=ran,
        error_threshold=error_threshold,
        turn=turn,
        sampled_rule_rows=len(sampled_rules),
        sampling_seconds=sampling_seconds,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        total_seconds=train_seconds + test_seconds,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
    )


def run_fkgs_experiments(
    train_rule_csv: Path,
    test_rule_csv: Path,
    output_dir: Path,
    ran_values: list[int] | None = None,
    error_thresholds: list[float] | None = None,
    turns: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    # Chạy đủ các cấu hình ran/e và nhiều turn giống kịch bản thực nghiệm của source
    ran_values = ran_values or [15, 20]
    error_thresholds = error_thresholds or [0.2, 0.3]

    # Lấy train rules và test rules
    train_rules = load_rule_csv(train_rule_csv)
    test_rules = load_rule_csv(test_rule_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[FkgsRunResult] = []

    # Tính số lượt chạy
    total_runs = len(ran_values) * len(error_thresholds) * turns
    run_index = 0
    for ran in ran_values:
        for error_threshold in error_thresholds:
            for turn in range(1, turns + 1):
                run_index += 1
                print(f"[{run_index}/{total_runs}] Start FKGS ran={ran}, e={error_threshold}, turn={turn}", flush=True)
                result = run_fkgs_once(
                    train_rules=train_rules,
                    test_rules=test_rules,
                    ran=ran,
                    error_threshold=error_threshold,
                    turn=turn,
                    output_dir=output_dir,
                    random_state=random_state,
                )
                results.append(result)
                print(
                    f"ran={ran}, e={error_threshold}, turn={turn}: "
                    f"acc={result.accuracy:.4f}, precision={result.precision:.4f}, "
                    f"recall={result.recall:.4f}, total={result.total_seconds:.2f}s"
                )

    result_df = pd.DataFrame([asdict(result) for result in results])
    # Lưu kết quả từng turn trước khi gom trung bình/std
    result_df.to_csv(output_dir / "fkgs_turn_results.csv", index=False)

    # Summary theo từng cặp ran/e để đưa vào bảng báo cáo
    summary_df = (
        result_df.groupby(["ran", "error_threshold"])
        .agg(
            sampled_rule_rows_mean=("sampled_rule_rows", "mean"),
            sampling_seconds_mean=("sampling_seconds", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            test_seconds_mean=("test_seconds", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .reset_index()
    )
    summary_df.to_csv(output_dir / "fkgs_summary.csv", index=False)
    with open(output_dir / "fkgs_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary_df.to_dict(orient="records"), file, ensure_ascii=False, indent=2)

    return summary_df

