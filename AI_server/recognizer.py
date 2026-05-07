"""CRNN (CNNBiLSTMCTC) recognizer — inference wrapper.

Preprocessing khớp với training:
    - Grayscale
    - Resize cố định (W=960, H=100)  ← giống HandwritingDataset
    - Normalize [0, 1]
    - CTC greedy decode (blank_idx=0)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .define import CNNBiLSTMCTC, ensure_binary_black_bg

logger = logging.getLogger(__name__)

INPUT_HEIGHT = 64
INPUT_WIDTH  = 2048


class CRNNRecognizer:
    """Load CNNBiLSTMCTC từ checkpoint và chạy inference trên batch BGR crops."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        if not isinstance(ckpt, dict) or "char_to_idx" not in ckpt:
            raise KeyError(
                "Checkpoint thiếu key 'char_to_idx'. "
                "Cần re-train hoặc lưu vocab vào checkpoint."
            )

        self.char_to_idx: Dict[str, int] = ckpt["char_to_idx"]
        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}
        num_classes = len(self.char_to_idx)

        self.model = CNNBiLSTMCTC(num_classes=num_classes).to(self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
        self.model.eval()

        logger.info(
            "CRNNRecognizer loaded từ %s trên %s (num_classes=%d)",
            checkpoint_path, self.device, num_classes,
        )

    # ── Preprocessing ──────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_crop(bgr: np.ndarray) -> torch.Tensor:
        TARGET_H = 64
        TARGET_W = 2048

        # 1. Grayscale
        if bgr.ndim == 2:
            gray = bgr
        elif bgr.shape[2] == 4:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2. Binary hóa
        binary = ensure_binary_black_bg(gray)

        # 3. Deskew từng dòng
        binary = CRNNRecognizer._deskew_line(binary)

        # 4. Crop tight sau deskew (bỏ vùng đen thừa)
        binary = CRNNRecognizer._tight_crop(binary)

        # 5. Scale giữ tỉ lệ theo H, pad W
        h, w = binary.shape
        scale = TARGET_H / h
        new_w = int(w * scale)

        if new_w > TARGET_W:
            scale = TARGET_W / w
            new_h = int(h * scale)
            new_w = TARGET_W
        else:
            new_h = TARGET_H

        resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        pad_top = (TARGET_H - new_h) // 2
        canvas[pad_top:pad_top + new_h, :new_w] = resized

        arr = canvas.astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


    @staticmethod
    def _deskew_line(binary: np.ndarray) -> np.ndarray:
        """Deskew dựa trên horizontal projection — chính xác hơn minAreaRect."""
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            return binary

        # Dùng horizontal projection để tìm góc nghiêng thực của DÒNG
        # Thay vì minAreaRect (dễ bị lệch do nét chữ dọc)
        best_angle = 0.0
        best_score = -1.0

        for angle_deg in np.arange(-10, 10.1, 0.5):
            h, w = binary.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Projection theo hàng ngang
            proj = rotated.sum(axis=1).astype(np.float32)
            # Score = variance cao → chữ tập trung vào ít hàng → thẳng
            score = float(np.var(proj))
            if score > best_score:
                best_score = score
                best_angle = angle_deg

        # Chỉ rotate nếu góc đáng kể
        if abs(best_angle) < 0.3:
            return binary

        h, w = binary.shape
        pad = int(h * abs(np.tan(np.radians(best_angle))) * 1.5) + 10
        padded = cv2.copyMakeBorder(binary, 0, 0, pad, pad,
                                    cv2.BORDER_CONSTANT, value=0)
        ph, pw = padded.shape
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), best_angle, 1.0)
        rotated = cv2.warpAffine(padded, M, (pw, ph),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotated


    @staticmethod
    def _tight_crop(binary: np.ndarray) -> np.ndarray:
        """Crop bỏ vùng đen thừa xung quanh chữ, giữ lại padding nhỏ."""
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) == 0:
            return binary

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        PAD = 4
        h, w = binary.shape
        y1 = max(0, y_min - PAD)
        y2 = min(h, y_max + PAD)
        x1 = max(0, x_min - PAD)
        x2 = min(w, x_max + PAD)

        return binary[y1:y2, x1:x2]

    def _ctc_greedy_decode(self, log_probs: torch.Tensor) -> List[str]:
        """log_probs: (T, B, C) log-probabilities từ model.forward() → List[str] độ dài B.

        Blank index = 0. Loại bỏ blank và token lặp liên tiếp.
        """
        # Model đã trả log-probs (F.log_softmax bên trong define.py)
        best_idx  = log_probs.argmax(dim=2)             # (T, B)
        best_idx  = best_idx.transpose(0, 1).cpu().numpy()  # (B, T)

        print(f"[CTC DEBUG] best_idx shape: {best_idx.shape}")
        print(f"[CTC DEBUG] unique indices: {np.unique(best_idx)}")
        print(f"[CTC DEBUG] blank (0) count: {(best_idx == 0).sum()}")
        print(f"[CTC DEBUG] non-blank count: {(best_idx != 0).sum()}")

        results: List[str] = []
        for row_idx, row in enumerate(best_idx):
            chars: List[str] = []
            prev = -1
            for t, v in enumerate(row.tolist()):
                if v != 0 and v != prev:
                    char = self.idx_to_char.get(v, "")
                    if char:
                        chars.append(char)
                prev = v
            
            result = "".join(chars)
            print(f"[CTC DEBUG] Row {row_idx} decoded: '{result}'")
            results.append(result)
        
        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_batch(self, bgr_crops: List[np.ndarray]) -> List[str]:
        """Nhận list BGR crops, trả về list string đã decode."""
        if not bgr_crops:
            return []

        print(f"[CRNN DEBUG] predict_batch received {len(bgr_crops)} crops")
        
        tensors = []
        debug_dir = Path(__file__).parent / "debug_crops"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        for i, c in enumerate(bgr_crops):
            print(f"[CRNN DEBUG]   Crop {i}: input shape {c.shape}")
            t = self._preprocess_crop(c)
            tensors.append(t)
            print(f"[CRNN DEBUG]   Crop {i}: preprocessed shape {t.shape}, min={t.min():.4f}, max={t.max():.4f}, mean={t.mean():.4f}")

            # Save preprocessed image to disk for inspection
            try:
                arr = t.squeeze().squeeze().cpu().numpy()  # (H, W) in [0,1]
                arr_u8 = (arr * 255.0).clip(0,255).astype(np.uint8)
                ts = int(time.time() * 1000)
                out_path = debug_dir / f"pre_{ts}_{i}.png"
                cv2.imwrite(str(out_path), arr_u8)
                print(f"[CRNN DEBUG]   Saved preprocessed crop to: {out_path}")
            except Exception as ex:
                print(f"[CRNN DEBUG]   Failed saving preprocessed crop: {ex}")
        
        batch = torch.cat(tensors, dim=0).to(self.device)  # (B, 1, H, W)
        print(f"[CRNN DEBUG] Batch shape before model: {batch.shape}")
        
        log_probs = self.model(batch)  # (T, B, num_classes) log-probs
        log_probs = F.log_softmax(log_probs, dim=2)
        print(f"[CRNN DEBUG] Log probs shape: {log_probs.shape}")
        print(f"[CRNN DEBUG] Log probs stats: min={log_probs.min():.4f}, max={log_probs.max():.4f}")
        
        results = self._ctc_greedy_decode(log_probs)
        print(f"[CRNN DEBUG] Decoded results: {results}")
        
        return results
