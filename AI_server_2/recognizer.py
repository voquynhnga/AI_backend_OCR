"""VietOCR vgg_transformer recognizer wrapper.

VietOCR xu ly noi bo: resize giu aspect ratio den height=32, ToTensor, khong normalize.
Chi can truyen PIL.Image RGB la xong.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .preprocess import preprocess_line_crop

logger = logging.getLogger(__name__)


class VietOCRRecognizer:
    def __init__(
        self,
        weights: Optional[str | os.PathLike] = None,
        device: Optional[str] = None,
        config_name: str = "vgg_transformer",
        beam_search: bool = False,
        preprocess: bool = False,
    ):
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as e:
            raise ImportError(
                "Chua cai vietocr. Chay: pip install vietocr"
            ) from e

        cfg = Cfg.load_config_from_name(config_name)
        cfg["cnn"]["pretrained"] = False
        cfg["predictor"]["beamsearch"] = beam_search
        cfg["device"] = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights is not None:
            wp = Path(weights)
            if not wp.exists():
                raise FileNotFoundError(f"Khong tim thay weights: {wp}")
            cfg["weights"] = str(wp)

        self.device = torch.device(cfg["device"])
        self.config_name = config_name
        self.preprocess = preprocess
        self.predictor = Predictor(cfg)
        logger.info(
            "Loaded VietOCR %s tren %s (weights=%s, preprocess=%s)",
            config_name, self.device, cfg.get("weights"), preprocess,
        )

    @staticmethod
    def _bgr_to_pil(crop: np.ndarray) -> Image.Image:
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def predict_batch(self, bgr_crops: List[np.ndarray]) -> List[str]:
        if not bgr_crops:
            return []
        out: List[str] = []
        for crop in bgr_crops:
            try:
                proc = preprocess_line_crop(crop) if self.preprocess else crop
                pil = self._bgr_to_pil(proc)
                text = self.predictor.predict(pil)
            except Exception as e:
                logger.warning("VietOCR predict fail: %s", e)
                text = ""
            out.append(text)
        return out
