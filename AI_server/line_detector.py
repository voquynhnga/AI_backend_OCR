"""Tach dong cho anh viet tay (giay ke ngang hoac giay trang).

Pipeline (CC + y-clustering):
    1. Grayscale -> adaptive threshold (chu = trang).
    2. Xoa duong ke ngang bang morphology open kernel ngang dai.
    3. Mo nho (1x3) de diet tan du rule-line.
    4. Connected components -> bbox cua tung "blob" (chu / dau / cum nho).
    5. Loc CC: theo dien tich, chieu cao, ti le.
    6. Cluster theo y-center: gap_threshold = median(height) * factor.
    7. Moi cluster -> 1 bounding box dong = union cac CC trong cluster.
    8. Loc dong: so luong CC toi thieu, chieu rong, chieu cao.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def pad(self, px: int, py: int, max_w: int, max_h: int) -> "Box":
        return Box(
            max(0, self.x1 - px),
            max(0, self.y1 - py),
            min(max_w, self.x2 + px),
            min(max_h, self.y2 + py),
        )


def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # blur mạnh hơn cho bút chì mờ
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=71, C=4,  # blockSize lớn hơn, C nhỏ hơn
    )
    return bw


def _remove_ruled_lines(bw: np.ndarray, page_w: int) -> np.ndarray:
    """Xoa duong ke ngang dai. Kernel dai = it nhat 1/12 chieu rong trang."""
    kernel_len = max(40, page_w // 12)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    horizontal = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
        iterations=1,
    )
    cleaned = cv2.subtract(bw, horizontal)
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    )
    return cleaned


   

# ── Merge boxes bị overlap theo trục Y ──────────────────────────
def _merge_overlapping_boxes(boxes: List[Box], overlap_ratio: float = 0.5) -> List[Box]:
    """Merge các box có overlap theo chiều Y vượt quá overlap_ratio."""
    if not boxes:
        return boxes
    
    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(boxes)
        
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            group = [a]
            used[i] = True
            
            for j, b in enumerate(boxes):
                if used[j]:
                    continue
                # Tính overlap Y giữa a và b
                y_overlap = min(a.y2, b.y2) - max(a.y1, b.y1)
                min_h = min(a.h, b.h)
                if min_h > 0 and y_overlap / min_h >= overlap_ratio:
                    group.append(b)
                    used[j] = True
                    merged = True
            
            # Union tất cả box trong group
            x1 = min(g.x1 for g in group)
            y1 = min(g.y1 for g in group)
            x2 = max(g.x2 for g in group)
            y2 = max(g.y2 for g in group)
            result.append(Box(x1, y1, x2, y2))
        
        boxes = sorted(result, key=lambda b: (b.y1, b.x1))
    
    return boxes




class RuledPaperLineDetector:
    """Tach dong dua tren connected-components + y-clustering."""

    def __init__(
        self,
        min_cc_area: int = 30,
        min_cc_height: int = 8,
        max_cc_height_ratio: float = 0.25,
        max_cc_width_ratio: float = 0.85,
        y_gap_factor: float = 0.7,
        min_components_per_line: int = 1,
        min_strength_ratio: float = 0.02,
        min_line_height: int = 18,
        min_line_width_ratio: float = 0.02,
        pad_x: int = 40,
        pad_y: int = 10,
    ):
        self.min_cc_area = min_cc_area
        self.min_cc_height = min_cc_height
        self.max_cc_height_ratio = max_cc_height_ratio
        self.max_cc_width_ratio = max_cc_width_ratio
        self.y_gap_factor = y_gap_factor
        self.min_components_per_line = min_components_per_line
        self.min_strength_ratio = min_strength_ratio
        self.min_line_height = min_line_height
        self.min_line_width_ratio = min_line_width_ratio
        self.pad_x = pad_x
        self.pad_y = pad_y

    def detect(self, image_bgr: np.ndarray) -> List[Box]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        H, W = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bw = _binarize(gray)
        bw = _remove_ruled_lines(bw, page_w=W)

        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        max_h = int(H * self.max_cc_height_ratio)
        max_w = int(W * self.max_cc_width_ratio)

        ccs: List[Tuple[int, int, int, int, float]] = []
        for i in range(1, n_lbl):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w_ = int(stats[i, cv2.CC_STAT_WIDTH])
            h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_cc_area:
                continue
            if h_ < self.min_cc_height:
                continue
            if h_ > max_h or w_ > max_w:
                continue
            # Loai CC dang "soi" ngang (rule-line residue)
            if w_ > 8 * h_ and h_ < 12:
                continue
            ccs.append((x, y, x + w_, y + h_, y + h_ / 2.0))

        if not ccs:
            return []

        heights = np.array([b[3] - b[1] for b in ccs], dtype=np.float32)
        ref_h = float(np.percentile(heights, 70))

        proj = np.zeros(H, dtype=np.float32)
        for x1, y1, x2, y2, yc in ccs:
            yc_int = int(round(yc))
            if 0 <= yc_int < H:
                weight = float(np.sqrt(max(1, (x2 - x1) * (y2 - y1))))
                proj[yc_int] += weight

        sigma = max(2.0, ref_h / 3.0)
        ksize = int(sigma * 4) | 1
        proj_s = cv2.GaussianBlur(
            proj.reshape(-1, 1), (1, ksize), sigma,
        ).flatten()

        peak_thresh = max(2.0, float(np.max(proj_s)) * 0.03)
        peaks: List[int] = []
        min_dist = max(int(ref_h * 1.0), 12)
        last_peak = -10**9
        for y in range(1, H - 1):
            if proj_s[y] < peak_thresh:
                continue
            if proj_s[y] >= proj_s[y - 1] and proj_s[y] >= proj_s[y + 1]:
                if y - last_peak >= min_dist:
                    peaks.append(y)
                    last_peak = y
                elif peaks and proj_s[y] > proj_s[peaks[-1]]:
                    peaks[-1] = y
                    last_peak = y

        if not peaks:
            return []

        clusters: List[List[Tuple[int, int, int, int, float]]] = [[] for _ in peaks]
        peaks_arr = np.array(peaks, dtype=np.float32)
        max_dist_to_peak = ref_h * 3.0
        for cb in ccs:
            dists = np.abs(peaks_arr - cb[4])
            k = int(np.argmin(dists))
            if dists[k] <= max_dist_to_peak:
                clusters[k].append(cb)

        min_w_line = int(W * self.min_line_width_ratio)
        cluster_strength = [
            sum((c[2] - c[0]) * (c[3] - c[1]) for c in cl)
            for cl in clusters
        ]
        max_strength = max(cluster_strength) if cluster_strength else 0

        boxes: List[Box] = []
        for cluster, strength in zip(clusters, cluster_strength):
            if len(cluster) < self.min_components_per_line:
                continue
            if max_strength > 0 and strength < max_strength * self.min_strength_ratio:
                continue
            x1 = min(c[0] for c in cluster)
            y1 = min(c[1] for c in cluster)
            x2 = max(c[2] for c in cluster)
            y2 = max(c[3] for c in cluster)
            
            if (x2 - x1) < min_w_line:
                continue
                
            line_h = y2 - y1
            if line_h < self.min_line_height:
                continue

            # ========================================================
            # FIX LỖI CHỮ NGHIÊNG (SLANT COMPENSATION)
            # ========================================================
            # Bù đắp độ nghiêng: nới rộng lề X tỷ lệ thuận với chiều cao dòng.
            # 0.35 tương đương bù đắp cho góc nghiêng khoảng ~20 độ (tan(20°) ≈ 0.36)
            dynamic_pad_x = int(line_h * 0.35)
            
            # Lấy giá trị lớn nhất giữa pad_x mặc định và pad_x động
            actual_pad_x = max(self.pad_x, dynamic_pad_x)
            
            # Nới lề y thêm một chút (thường chữ nghiêng cũng hay có nét móc dài)
            actual_pad_y = max(self.pad_y, int(line_h * 0.1))

            boxes.append(Box(x1, y1, x2, y2).pad(actual_pad_x, actual_pad_y, W, H))

        boxes.sort(key=lambda b: (b.y1, b.x1))
        boxes = _merge_overlapping_boxes(boxes, overlap_ratio=0.35)
        # Lọc bỏ box không phải chữ dựa trên mật độ pixel
        valid_boxes = []
        for box in boxes:
            region = image_bgr[box.y1:box.y2, box.x1:box.x2]
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            density = bw.mean() / 255.0  # tỉ lệ pixel chữ
            
            # Chữ viết tay thường có density 3%-25%
            # Vân gỗ/nhiễu có density rất cao (>35%) hoặc rất thấp
            if 0.005 <= density <= 0.3:
                valid_boxes.append(box)
            else:
                print(f"[DEBUG] Filtered box ({box.x1},{box.y1})→({box.x2},{box.y2}): density={density:.3f}")

        valid_boxes = [b for b in valid_boxes if b.w > W * 0.02 and b.h > 15]


        boxes = valid_boxes
        return boxes