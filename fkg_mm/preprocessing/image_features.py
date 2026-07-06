from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from fkg_mm.config import (
    CLAHE_CLIP_LIMIT,  # Mức giới hạn tương phản của CLAHE; cao hơn -> tăng tương phản mạnh hơn.
    CLAHE_TILE_GRID_SIZE,  # Kích thước lưới chia ảnh cho CLAHE (xử lý cục bộ theo từng tile).
    DENOISE_H,  # Cường độ khử nhiễu cho kênh độ sáng (luminance) trong Non-local Means.
    DENOISE_H_COLOR,  # Cường độ khử nhiễu cho kênh màu trong Non-local Means.
    DENOISE_SEARCH_WINDOW_SIZE,  # Cửa sổ tìm patch tương tự; lớn hơn -> tìm rộng hơn nhưng chậm hơn.
    DENOISE_TEMPLATE_WINDOW_SIZE,  # Kích thước patch mẫu để so sánh tương đồng cục bộ.
    GLCM_ANGLES,  # Danh sách góc dùng khi tạo ma trận GLCM (hướng kết cấu).
    GLCM_DISTANCES,  # Danh sách khoảng cách pixel khi tính đồng xuất hiện trong GLCM.
    GLCM_LEVELS,  # Số mức xám dùng để lượng tử hóa ảnh trước khi tính GLCM.
    GLCM_NORMED,  # Có chuẩn hóa ma trận GLCM về xác suất hay không.
    GLCM_SYMMETRIC,  # Có ép GLCM đối xứng (i,j) = (j,i) hay không.
    IMAGE_FEATURE_COLUMNS,  # Tên các cột feature ảnh đầu ra (GLCM + thống kê).
    IMAGE_ID_COLUMN,  # Tên cột định danh ảnh để nối dữ liệu đa mô thức.
    IMAGE_SIZE,  # Kích thước resize ảnh chuẩn trước khi trích xuất feature.
    OTSU_BLUR_KERNEL,  # Kernel Gaussian blur trước khi threshold Otsu để giảm nhiễu.
    TARGET_COLUMN,  # Tên cột nhãn mục tiêu (phân loại bệnh).
    UNSHARP_AMOUNT,  # Độ mạnh làm nét trong unsharp mask.
    UNSHARP_SIGMA_X,  # Sigma của Gaussian blur trong unsharp mask.
)

# Function tăng tương phản từng phần cho ảnh, cái function này phù hợp với
# các bài toán y tế hơn là HE (Histogram Equalization)
def apply_clahe(image: np.ndarray) -> np.ndarray:
    # Tăng tương phản cục bộ trên kênh sáng (L) trong không gian LAB rồi đổi về BGR.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Function unsharp masking dùng làm nét ảnh
def apply_unsharp_mask(image: np.ndarray, amount: float = UNSHARP_AMOUNT) -> np.ndarray:
    # Làm nét bằng cách lấy ảnh gốc trừ ảnh blur rồi cộng ngược lại theo hệ số amount.
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=UNSHARP_SIGMA_X)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

# Function sử dụng otsu để tìm threshold phù hợp
def segment_by_otsu(gray_image: np.ndarray) -> np.ndarray:
    # Làm mượt nhẹ rồi dùng Otsu để tự chọn ngưỡng, trả về mask nhị phân 0/255.
    blurred = cv2.GaussianBlur(gray_image, OTSU_BLUR_KERNEL, 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask


def source_style_gray(image: np.ndarray) -> np.ndarray:
    # Chuyển ảnh RGB sang ảnh xám kiểu source style và ép về uint8.
    return img_as_ubyte(color.rgb2gray(image))

# Function tiền xử lý ảnh giúp quá trình tách object khỏi background tốt hơn
def preprocess_fundus_image(image: np.ndarray) -> np.ndarray:
    # Pipeline tiền xử lý ảnh fundus: làm nét -> khử nhiễu màu -> tăng tương phản cục bộ.
    # Làm nét ảnh
    sharpened = apply_unsharp_mask(image)

    # Khử nhiễu ảnh
    denoised = cv2.fastNlMeansDenoisingColored(
        sharpened,
        None,
        DENOISE_H, # Điều khiển khử nhiễu độ sáng
        DENOISE_H_COLOR, # Điều khiển khử nhiễu màu
        DENOISE_TEMPLATE_WINDOW_SIZE, # Kích thước patch so sánh
        DENOISE_SEARCH_WINDOW_SIZE, # Vùng tìm patch tương tự
    )
    # Tăng tương phản từng phần cho ảnh
    return apply_clahe(denoised)

# Function để chuẩn hoá data của các features
def glcm_mean(matrix: np.ndarray, prop: str) -> float:
    # Tính trung bình thuộc tính GLCM theo mọi góc/khoảng cách đã cấu hình.
    return float(np.mean(graycoprops(matrix, prop)))


def extract_one_image_features(image_path: Path) -> dict[str, float]:
    # Đọc 1 ảnh, tiền xử lý, tạo mask, tính GLCM + thống kê pixel và trả về dict feature.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize ảnh về kích thước 256x256
    image = cv2.resize(image, IMAGE_SIZE)

    # Tiền xử lý ảnh
    image = preprocess_fundus_image(image)

    # Đơn giản hoá ảnh (chuyển từ hệ 3 màu sang hệ 1 màu)
    otsu_gray = source_style_gray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dùng thuật toán otsu để tự động tìm ngưỡng tách phần mắt với phần background
    otsu_mask = segment_by_otsu(otsu_gray)

    # Dùng phần mask đã tìm đc ở trên và áp lên ảnh xám -> chỉ còn vùng mắt, các vùng khác bị xoá thành màu đen
    masked_gray = img_as_ubyte(cv2.bitwise_and(gray, gray, mask=otsu_mask))

    # Tính GLCM matrix (ma trận này đếm tần suất xuất hiện của các cặp pixel có độ sáng nhất định nằm cạnh nhau giúp trích xuất các đặc trưng như độ tương phản, độ đồng nhất, ...)
    matrix = graycomatrix(
        masked_gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        normed=GLCM_NORMED,
        symmetric=GLCM_SYMMETRIC,
    )

    # Lấy pixel phần ngoài vùng mắt 
    masked_pixels = image[otsu_mask == 0]
    if masked_pixels.size == 0:
        masked_pixels = image.reshape(-1, image.shape[-1])

    return {
        # Features của GLCM (vùng mask = 255)
        "glcm_contrast": glcm_mean(matrix, "contrast"),
        "glcm_dissimilarity": glcm_mean(matrix, "dissimilarity"),
        "glcm_homogeneity": glcm_mean(matrix, "homogeneity"),
        "glcm_energy": glcm_mean(matrix, "energy"),
        "glcm_correlation": glcm_mean(matrix, "correlation"),
        "glcm_asm": glcm_mean(matrix, "ASM"),
        # Features của Statistical (vùng mask = 0)
        "stat_mean": float(np.mean(masked_pixels)),
        "stat_variance": float(np.var(masked_pixels)),
        "stat_std": float(np.std(masked_pixels)),
        "stat_rms": float(np.sqrt(np.mean(np.square(masked_pixels)))),
    }


# Chuẩn hoá dữ liệu về miền [0,1]
def min_max_normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hóa từng cột feature ảnh về [0, 1]; cột hằng sẽ được gán 0.0.
    normalized = df.copy()
    for col in IMAGE_FEATURE_COLUMNS:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            normalized[col] = 0.0
        else:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
    return normalized


def extract_image_features(
    records: pd.DataFrame,
    raw_output_csv: Path,
    normalized_output_csv: Path,
) -> pd.DataFrame:
    # Lặp toàn bộ records để trích xuất feature ảnh, lưu bản raw + bản normalize rồi trả về DataFrame normalize.
    rows: list[dict[str, object]] = []
    start = time.time()

    # in dqdm để hiển thị progress trên terminal
    for record in tqdm(records.itertuples(index=False), total=len(records), desc="image features"):
        # Lấy path của ảnh từ mẫu dữ liệu hiện tại
        image_path = Path(getattr(record, "image_path"))
        features = extract_one_image_features(image_path)
        rows.append(
            {
                IMAGE_ID_COLUMN: getattr(record, IMAGE_ID_COLUMN),
                **features,
                TARGET_COLUMN: int(getattr(record, TARGET_COLUMN)),
            }
        )

    # Lưu file vào tiện debug/check lại sau này
    features_df = pd.DataFrame(rows, columns=[IMAGE_ID_COLUMN, *IMAGE_FEATURE_COLUMNS, TARGET_COLUMN])
    raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(raw_output_csv, index=False)

    # Chuẩn hoá lại data về [0,1] cho đồng nhất rồi lưu lại để sử dụng
    normalized = min_max_normalize_features(features_df)
    normalized_output_csv.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(normalized_output_csv, index=False)

    print(f"Extracted {len(features_df)} image feature rows in {time.time() - start:.2f}s")
    return normalized
