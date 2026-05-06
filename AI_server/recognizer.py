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
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .define import CNNBiLSTMCTC

logger = logging.getLogger(__name__)

INPUT_HEIGHT = 100
INPUT_WIDTH  = 960


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
        """BGR uint8 → (1, 1, H=100, W=960) float32 tensor trong [0, 1]."""
        if bgr.ndim == 2:
            gray = bgr
        elif bgr.shape[2] == 4:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
        arr = resized.astype(np.float32) / 255.0
        # (H, W) → (1, 1, H, W)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # ── CTC Decode ─────────────────────────────────────────────────────────────

    def _ctc_greedy_decode(self, log_probs: torch.Tensor) -> List[str]:
        """log_probs: (T, B, C) log-probabilities từ model.forward() → List[str] độ dài B.

        Blank index = 0. Loại bỏ blank và token lặp liên tiếp.
        """
        # Model đã trả log-probs (F.log_softmax bên trong define.py)
        best_idx  = log_probs.argmax(dim=2)             # (T, B)
        best_idx  = best_idx.transpose(0, 1).cpu().numpy()  # (B, T)

        results: List[str] = []
        for row in best_idx:
            chars: List[str] = []
            prev = -1
            for v in row.tolist():
                if v != 0 and v != prev:
                    char = self.idx_to_char.get(v, "")
                    if char:
                        chars.append(char)
                prev = v
            results.append("".join(chars))
        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_batch(self, bgr_crops: List[np.ndarray]) -> List[str]:
        """Nhận list BGR crops, trả về list string đã decode."""
        if not bgr_crops:
            return []

        tensors = [self._preprocess_crop(c) for c in bgr_crops]
        batch    = torch.cat(tensors, dim=0).to(self.device)  # (B, 1, H, W)
        log_probs = self.model(batch)                          # (T, B, num_classes) log-probs
        return self._ctc_greedy_decode(log_probs)
