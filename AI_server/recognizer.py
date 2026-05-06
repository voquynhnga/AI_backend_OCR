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

        if bgr.ndim == 2:
            gray = bgr
        elif bgr.shape[2] == 4:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gray = ensure_binary_black_bg(gray)

        h, w = gray.shape

        # Scale để chữ chiếm tỉ lệ tương tự ảnh training
        scale_h = TARGET_H / h
        scale_w = TARGET_W / w
        
        # Dùng scale nhỏ hơn để vừa khung, KHÔNG bóp méo
        scale = min(scale_h, scale_w * 0.5)  # *0.5 để chữ không quá rộng
        
        new_h = min(int(h * scale), TARGET_H)
        new_w = min(int(w * scale), TARGET_W)

        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad: căn giữa theo H, căn trái theo W (giống training)
        canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        pad_top = (TARGET_H - new_h) // 2
        
        # Dịch ảnh sang phải 30 pixel thay vì dán sát lề 0
        pad_left = 30
        
        # Đảm bảo phần dán vào không bị tràn khỏi TARGET_W (2048)
        actual_w = min(new_w, TARGET_W - pad_left)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + actual_w] = resized[:, :actual_w]

        arr = canvas.astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

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
