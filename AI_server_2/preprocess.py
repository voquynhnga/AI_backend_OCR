"""Preprocess crop dong truoc khi day vao VietOCR.

Pipeline:
    1. Xoa duong ke ngang trong crop (inpaint).
    2. CLAHE tren kenh L cua LAB de tang tuong phan cuc bo (chi khi can).
    3. Tang nhe sharpen (unsharp mask) — tat theo mac dinh.
    4. Pad trang xung quanh de model nhin "thoang" hon.

Mac dinh: chi remove_rules=True va pad=True (an toan, khong lam mo anh).
"""

from __future__ import annotations

import cv2
import numpy as np


def _remove_ruled_lines_from_crop(image: np.ndarray) -> np.ndarray:
    """Xoa duong ke ngang (lien tuc hoac cham cham) khoi anh BGR."""
    h, w = image.shape[:2]
    if w < 60 or h < 12:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        25, 10,
    )
    closed = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
    )
    row_density = closed.sum(axis=1) / 255.0
    rule_rows = np.where(row_density >= 0.5 * w)[0]
    if rule_rows.size == 0:
        return image
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rule_rows, :] = 255
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
        iterations=1,
    )
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def _needs_clahe(image: np.ndarray, std_thresh: float = 65.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)) < std_thresh


def _clahe_bgr(image: np.ndarray, clip: float = 1.5, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _unsharp(image: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)


def _pad_white(image: np.ndarray, px: int = 4, py: int = 2) -> np.ndarray:
    return cv2.copyMakeBorder(
        image, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def preprocess_line_crop(
    crop_bgr: np.ndarray,
    deskew: bool = False,
    remove_rules: bool = True,
    clahe: bool = False,
    sharpen: bool = False,
    pad: bool = True,
) -> np.ndarray:
    """Chay preprocess nhe tren mot crop dong text truoc khi day VietOCR."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    out = crop_bgr
    if remove_rules:
        out = _remove_ruled_lines_from_crop(out)
    if clahe and _needs_clahe(out):
        out = _clahe_bgr(out)
    if sharpen:
        out = _unsharp(out)
    if pad:
        out = _pad_white(out)
    return out
