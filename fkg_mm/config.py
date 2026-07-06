from __future__ import annotations

import numpy as np

TARGET_COLUMN = "diabetic_retinopathy"
IMAGE_ID_COLUMN = "image_id"

SOURCE_TABLE_COLUMNS = [
    IMAGE_ID_COLUMN,
    "patient_age",
    "diabetes_time_y",
    "insuline",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "quality",
    TARGET_COLUMN,
]

TABULAR_FEATURE_COLUMNS = [
    "patient_age",
    "diabetes_time_y",
    "insuline",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "quality",
]

IMAGE_FEATURE_COLUMNS = [
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_asm",
    "stat_mean",
    "stat_variance",
    "stat_std",
    "stat_rms",
]

IMAGE_SIZE = (256, 256)
UNSHARP_AMOUNT = 1.5
UNSHARP_SIGMA_X = 3
DENOISE_H = 10
DENOISE_H_COLOR = 10
DENOISE_TEMPLATE_WINDOW_SIZE = 7
DENOISE_SEARCH_WINDOW_SIZE = 21
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)
OTSU_BLUR_KERNEL = (5, 5)
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 256
GLCM_NORMED = True
GLCM_SYMMETRIC = True

DEFAULT_K_TAB = 9
DEFAULT_K_IMG = 7
DEFAULT_FIS_CLUSTER = [5] * (DEFAULT_K_IMG + DEFAULT_K_TAB) + [2]
