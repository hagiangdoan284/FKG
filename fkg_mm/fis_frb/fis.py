from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fkg_mm.config import DEFAULT_FIS_CLUSTER, TARGET_COLUMN


def stratify_or_none(values: pd.Series) -> pd.Series | None:
    counts = values.value_counts()
    return values if not counts.empty and int(counts.min()) >= 2 else None


# Sử dụng Fuzzy C-Means. So với K-Means (phân cứng), FCM cho phép 1 điểm thuộc nhiều cụm với mức độ khác nhau.
def fuzzy_c_means_1d(
    values: np.ndarray,
    n_clusters: int,
    initial_centers: np.ndarray,
    m: float = 2.0,
    eps: float = 0.01,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = values.reshape(-1, 1).astype(np.float64)
    centers = np.asarray(initial_centers, dtype=np.float64).reshape(-1, 1)

    # Trường hợp cột hằng (mọi giá trị giống nhau): gán toàn bộ vào cụm 1 để tránh chia 0.
    if np.allclose(x_values, x_values[0]):
        membership = np.zeros((n_clusters, x_values.shape[0]), dtype=np.float64)
        membership[0, :] = 1.0
        centers = np.full((n_clusters, 1), x_values[0, 0], dtype=np.float64)
        return centers, membership

    prev_objective = np.inf
    membership = np.zeros((n_clusters, x_values.shape[0]), dtype=np.float64)

    for _ in range(max_iter):
        # Khoảng cách tuyệt đối từ mỗi điểm tới từng tâm cụm.
        distances = np.abs(x_values - centers.T)
        distances[distances == 0] = np.finfo(float).eps

        # Cập nhật membership theo công thức FCM.
        power = 2 / (m - 1)
        inv_distances = distances ** (-power)
        new_membership = (inv_distances / inv_distances.sum(axis=1, keepdims=True)).T
        new_membership = np.nan_to_num(new_membership, nan=0.0)
        new_membership /= new_membership.sum(axis=0, keepdims=True)

        # Cập nhật tâm cụm theo weighted mean (trọng số là membership^m).
        weights = new_membership**m
        centers = (weights @ x_values) / (weights.sum(axis=1, keepdims=True) + np.finfo(float).eps)

        # Hàm mục tiêu FCM để kiểm tra hội tụ.
        objective = float(np.sum(weights.T * (distances**2)))
        if abs(objective - prev_objective) < eps:
            membership = new_membership
            break
        prev_objective = objective
        membership = new_membership

    return centers, membership


# Khởi tạo tâm cụm ban đầu theo min/max của cột. Thiết kế thủ công cho 1-3 cụm
def initial_centers(min_value: float, max_value: float, n_clusters: int) -> np.ndarray:
    if n_clusters == 1:
        return np.array([(min_value + max_value) / 2])
    if n_clusters == 2:
        return np.array([min_value, max_value])
    if n_clusters == 3:
        return np.array([min_value, (min_value + max_value) / 2, max_value])
    return np.linspace(min_value, max_value, n_clusters)


# Tính sigma cho hàm membership Gaussian từ độ trải tâm cụm. Sigma lớn -> các tập mờ chồng lấn nhiều hơn.
def compute_sigma(center_vector: np.ndarray) -> float:
    if len(center_vector) < 2:
        return 1.0
    max_distance = float(np.max(np.abs(center_vector[:, None] - center_vector[None, :])))
    sigma = abs(max_distance) / (2 * np.sqrt(2 * np.log(2)))
    if sigma <= 0 or not np.isfinite(sigma):
        return 1.0
    while sigma < 1:
        sigma *= 10
    return float(sigma)


# Tính mức thuộc Gaussian của 1 giá trị x vào cụm có nhãn `label` (1-based).
def gaussian_membership(x_value: float, label: float, n_clusters: int, sigma: float, centers: np.ndarray) -> float:
    if not 1 <= label <= n_clusters:
        return 0.0
    center = centers[int(label) - 1]
    return float(np.exp(-((x_value - center) ** 2) / (2 * sigma**2)))


# Sinh bảng rule thô từ dữ liệu train:
# - Chạy FCM theo từng cột.
# - Với mỗi ô dữ liệu, lấy cụm có membership cao nhất (argmax) làm nhãn rời rạc.
def generate_rules(
    train_data: np.ndarray,
    cluster: list[int],
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    n_rows, n_cols = train_data.shape
    rules = np.zeros((n_rows, n_cols))
    centers: list[np.ndarray] = []
    last_membership = np.zeros((n_rows, cluster[-1]))

    for col_idx in range(n_cols):
        start_centers = initial_centers(min_values[col_idx], max_values[col_idx], cluster[col_idx])
        col_centers, membership = fuzzy_c_means_1d(
            train_data[:, col_idx],
            n_clusters=cluster[col_idx],
            initial_centers=start_centers,
        )
        membership_t = membership.T
        centers.append(col_centers.flatten())
        rules[:, col_idx] = np.argmax(membership_t, axis=1) + 1
        last_membership = membership_t

    return rules, centers, last_membership


# Tính độ tin cậy theo từng feature cho từng rule bằng Gaussian membership.
# Điểm càng gần tâm cụm tương ứng thì weight càng cao.
def rule_weight(
    rules: np.ndarray,
    train_features: np.ndarray,
    cluster: list[int],
    centers: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_features = train_features.shape
    sigma = np.zeros(n_features)
    weights = np.zeros((n_rows, n_features))

    for feature_idx in range(n_features):
        sigma[feature_idx] = compute_sigma(centers[feature_idx])
        for row_idx in range(n_rows):
            weights[row_idx, feature_idx] = gaussian_membership(
                train_features[row_idx, feature_idx],
                rules[row_idx, feature_idx],
                cluster[feature_idx],
                sigma[feature_idx],
                centers[feature_idx],
            )
    return weights, sigma


# Gộp các rule trùng điều kiện (antecedent):
# - Key: toàn bộ cột điều kiện.
# - Nếu trùng key thì giữ biến thể có weight tốt hơn.
# - Output chỉ giữ lại [điều kiện..., label].
def reduce_rules(rules_with_weight: np.ndarray) -> np.ndarray:
    rule_dict: dict[tuple[float, ...], list[float]] = {}
    for rule in rules_with_weight:
        condition = tuple(rule[:-3])
        weight = rule[-2]
        label = rule[-3]
        result = [weight, label]
        if condition in rule_dict:
            if rule_dict[condition][0] > result[0]:
                rule_dict[condition] = result
        else:
            rule_dict[condition] = result
    return np.array([[*key, value[1]] for key, value in rule_dict.items()])


# Lọc rule mạnh để làm rule base cho model.
# Cột cuối của `rules_reduce` được dùng như ngưỡng lọc >= 0.9.
def remove_strong_rules(rules_reduce: np.ndarray, n_feature_cols: int) -> np.ndarray:
    strong_rules = [
        tuple(rules_reduce[row_idx])
        for row_idx in range(rules_reduce.shape[0])
        if rules_reduce[row_idx, rules_reduce.shape[1] - 1] >= 0.9
    ]
    if not strong_rules:
        return np.empty((0, n_feature_cols + 1))
    return np.array(list(set(strong_rules)))[:, : n_feature_cols + 1]


# Entry chính cho bước FIS/FRB:
# 1) Đọc dữ liệu fusion.
# 2) Sinh fuzzy rules từ train.
# 3) Giảm trùng/lọc rule model.
# 4) Xuất toàn bộ artifact (.csv/.pkl/.json) cho bước FKGS.
def run_fis_frb(
    input_csv: Path,
    output_dir: Path,
    file_name: str = "feature_selection",
    cluster: list[int] | None = None,
    random_state: int | None = None,
) -> dict[str, Path | int | float]:
    start = time.time()
    cluster = cluster or DEFAULT_FIS_CLUSTER
    data = pd.read_csv(input_csv)

    # Ràng buộc định dạng: cột cuối phải là nhãn mục tiêu.
    if data.columns[-1] != TARGET_COLUMN:
        raise ValueError(f"Expected last column to be {TARGET_COLUMN!r}.")

    # Mỗi cột dữ liệu cần một giá trị số cụm tương ứng trong `cluster`.
    if len(cluster) != data.shape[1]:
        raise ValueError(f"cluster length must be {data.shape[1]}, got {len(cluster)}.")

    input_dir = output_dir / "input" / file_name
    fis_output_dir = output_dir / "output" / file_name
    frb_dir = fis_output_dir / "FRB"
    input_dir.mkdir(parents=True, exist_ok=True)
    frb_dir.mkdir(parents=True, exist_ok=True)

    # Chia train/test (70/30).
    train_df, test_df = train_test_split(
        data,
        train_size=0.7,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_or_none(data.iloc[:, -1]),
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(input_dir / "train_data.csv", index=False)
    test_df.to_csv(input_dir / "test_data.csv", index=False)

    # Min/max lấy trên full data để ổn định biên cụm giữa train/test.
    full_data = data.to_numpy(dtype=float)
    train_data = train_df.to_numpy(dtype=float)
    min_values = np.min(full_data, axis=0)
    max_values = np.max(full_data, axis=0)
    pd.DataFrame(min_values).to_csv(fis_output_dir / "min_vals.csv", index=False)
    pd.DataFrame(max_values).to_csv(fis_output_dir / "max_vals.csv", index=False)

    # Sinh rule thô bằng FCM cho toàn bộ cột.
    rules, centers, last_membership = generate_rules(train_data, cluster, min_values, max_values)
    label_col = train_data.shape[1] - 1

    # Gán nhãn rule từ membership của cột label (1-based index cụm).
    rules[:, label_col] = np.argmax(last_membership, axis=1) + 1

    # Tính weight và sigma cho phần feature (không tính cột label).
    weights, sigma = rule_weight(rules, train_data[:, :-1], cluster, centers)

    # Giữ cấu trúc sigma_M.
    sigma_m = sigma.reshape(-1, 1)
    sigma_m = np.hstack((sigma_m[:, [0]], sigma_m[:, [0]], sigma_m[:, [0]]))

    rule_list_all = pd.DataFrame(rules)
    rule_list_all.to_csv(fis_output_dir / "Rule_List_All.csv", index=False)
    rule_list_all.to_csv(fis_output_dir / "Rule_List.csv", index=False)

    # Gộp rule trùng theo điều kiện.
    rules_with_weight = np.hstack((rules, np.min(weights, axis=1, keepdims=True), train_data[:, [label_col]]))
    rules_reduce = reduce_rules(rules_with_weight)
    pd.DataFrame(rules_reduce).to_csv(fis_output_dir / "Rule_List_reduce.csv", index=False)

    # Lọc rule mạnh để tạo tập rule model.
    rule_list_model = remove_strong_rules(rules_reduce, label_col)
    pd.DataFrame(rule_list_model).to_csv(fis_output_dir / "Rule_List_model.csv", index=False)
    pd.DataFrame(sigma_m).to_csv(fis_output_dir / "Sigma_M.csv", index=False)
    pd.DataFrame(centers).to_csv(fis_output_dir / "Centers.csv", index=False)

    # Tạo bộ Train/Test rule cho bước FKGS (định dạng FRB).
    train_rules, test_rules = train_test_split(
        rule_list_all,
        train_size=0.7,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_or_none(rule_list_all.iloc[:, -1]),
    )
    train_rules = train_rules.reset_index(drop=True)
    test_rules = test_rules.reset_index(drop=True)
    train_rules.to_csv(frb_dir / "TrainDataRule.csv", index=False)
    test_rules.to_csv(frb_dir / "TestDataRule.csv", index=False)

    # Lưu model đóng gói và summary để tiện tái sử dụng/kiểm tra.
    model_data = {
        "ruleList": rule_list_model,
        "sigma_M": sigma_m,
        "centers": centers,
        "min_vals": min_values,
        "max_vals": max_values,
    }
    with open(fis_output_dir / "fuzzy_model.pkl", "wb") as file:
        pickle.dump(model_data, file)

    summary = {
        "input_rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "rule_rows": int(len(rule_list_all)),
        "reduced_rule_rows": int(len(rules_reduce)),
        "model_rule_rows": int(len(rule_list_model)),
        "train_rule_rows": int(len(train_rules)),
        "test_rule_rows": int(len(test_rules)),
        "runtime_seconds": time.time() - start,
    }
    with open(fis_output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return {
        **summary,
        "train_rule_csv": frb_dir / "TrainDataRule.csv",
        "test_rule_csv": frb_dir / "TestDataRule.csv",
    }
