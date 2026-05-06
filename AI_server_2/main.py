"""FastAPI server — VietOCR vgg_transformer Handwriting OCR.

Endpoints:
    GET  /health    → liveness check
    POST /predict   → upload ảnh, trả về text từng dòng (JSON giống AI_server)
    POST /grade     → chấm điểm bài làm bằng Gemini (JSON)

Env vars:
    VIETOCR_WEIGHTS   path tới file .pth          (default: ./vgg_transformer.pth trong PROJECT_DIR)
    VIETOCR_CONFIG    tên config VietOCR           (default: vgg_transformer)
    VIETOCR_PREPROCESS  "1" để bật preprocess      (default: 0 = tắt, raw crop)
    HOST              bind host                   (default: 0.0.0.0)
    PORT              bind port                   (default: 8000)
    GEMINI_API_KEY    Gemini API key
    GEMINI_MODEL      model name                  (default: gemini-2.0-flash-latest)
    GEMINI_TIMEOUT    timeout giây                (default: 25)
    GEMINI_RETRIES    số lần retry                (default: 3)
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .line_detector import Box, RuledPaperLineDetector
from .recognizer import VietOCRRecognizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config từ env ─────────────────────────────────────────────────────────────
_PROJECT_DIR   = Path(__file__).resolve().parent.parent   # D:\ki8\xla\BE_server
_DEFAULT_WEIGHTS = _PROJECT_DIR / "vgg_transformer.pth"

WEIGHTS_PATH  = Path(os.getenv("VIETOCR_WEIGHTS", str(_DEFAULT_WEIGHTS)))
CONFIG_NAME   = os.getenv("VIETOCR_CONFIG",     "vgg_transformer")
PREPROCESS    = os.getenv("VIETOCR_PREPROCESS", "0") == "1"
DEVICE        = os.getenv("VIETOCR_DEVICE",     "")  # "" → auto

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDpa7foCo-Yltiovn1tZFAn5Uur8zKsETk")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-2.0-flash-latest")
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "25"))
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES",   "3"))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="VietOCR Handwriting OCR",
    description="VietOCR vgg_transformer nhận diện chữ viết tay tiếng Việt",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model instances ─────────────────────────────────────────────────────
_line_detector: RuledPaperLineDetector | None = None
_recognizer:    VietOCRRecognizer | None      = None


@app.on_event("startup")
def _load_models() -> None:
    global _line_detector, _recognizer

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Không tìm thấy weights: {WEIGHTS_PATH}\n"
            f"Hãy đặt file vgg_transformer.pth vào {_PROJECT_DIR} "
            f"hoặc set env VIETOCR_WEIGHTS."
        )

    logger.info("Loading line detector …")
    _line_detector = RuledPaperLineDetector()

    logger.info(
        "Loading VietOCR %s từ %s (preprocess=%s) …",
        CONFIG_NAME, WEIGHTS_PATH, PREPROCESS,
    )
    _recognizer = VietOCRRecognizer(
        weights=WEIGHTS_PATH,
        device=DEVICE or None,
        config_name=CONFIG_NAME,
        preprocess=PREPROCESS,
    )
    logger.info("Models ready.")


# ── Pydantic schemas (giống AI_server) ────────────────────────────────────────
class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class LineResult(BaseModel):
    line_index: int
    text: str
    bbox: BBox


class PredictResponse(BaseModel):
    num_lines: int
    lines: List[LineResult]
    full_text: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "model": CONFIG_NAME,
        "weights": str(WEIGHTS_PATH),
        "device": str(_recognizer.device) if _recognizer else "not_loaded",
    }


@app.post("/predict", response_model=PredictResponse, tags=["ocr"])
async def predict(file: UploadFile = File(..., description="Ảnh chứa chữ viết tay")) -> PredictResponse:
    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh. Hãy gửi file jpg/png hợp lệ.")

    boxes = _line_detector.detect(img)
    if not boxes:
        logger.info("[predict] %s → 0 dòng", file.filename)
        return PredictResponse(num_lines=0, lines=[], full_text="")

    crops = [img[b.y1:b.y2, b.x1:b.x2] for b in boxes]
    texts = _recognizer.predict_batch(crops)

    lines = [
        LineResult(
            line_index=i,
            text=text,
            bbox=BBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2),
        )
        for i, (b, text) in enumerate(zip(boxes, texts))
    ]

    logger.info("[predict] %s → %d dòng: %s", file.filename, len(lines), texts)

    return PredictResponse(
        num_lines=len(lines),
        lines=lines,
        full_text="\n".join(r.text for r in lines),
    )


# ── /grade helpers ───────────────────────────────────────────────────────────
_GRADE_SYSTEM = (
    "Bạn là giáo viên đang chấm bài kiểm tra tự luận tiếng Việt. "
    "So sánh BÀI LÀM của học sinh với ĐÁP ÁN MẪU và cho điểm trên thang điểm đã cho. "
    "Cho điểm công bằng, ưu tiên ý đúng (không bắt bẻ chính tả nhỏ). "
    "Phân bổ đều cho từng câu trong tổng số câu.\n"
    "BẮT BUỘC trả về DUY NHẤT một JSON hợp lệ, KHÔNG kèm markdown, theo schema:\n"
    "{\n"
    '  "diem": <number, 0..thang_diem, có thể có 0.5>,\n'
    '  "nhan_xet": "<string ngắn gọn, tiếng Việt, có dấu>"\n'
    "}"
)


def _grade_with_gemini(
    *,
    hoc_sinh: str,
    so_cau: int,
    thang_diem: float,
    dap_an_mau: str,
    bai_lam: str,
) -> dict:
    user_prompt = (
        f"HỌC SINH: {hoc_sinh}\n"
        f"SỐ CÂU: {so_cau}\n"
        f"THANG ĐIỂM: {thang_diem}\n\n"
        f"=== ĐÁP ÁN MẪU ===\n{dap_an_mau}\n\n"
        f"=== BÀI LÀM CỦA HỌC SINH ===\n{bai_lam}\n\n"
        "Hãy chấm và trả JSON đúng schema."
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": _GRADE_SYSTEM + "\n\n" + user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-goog-api-key": GEMINI_API_KEY,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Gemini HTTP {e.code}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini request fail: {e}") from e

    data = json.loads(raw.decode("utf-8"))
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(text)


class GradeRequest(BaseModel):
    hoc_sinh:   str   = Field(..., description="Tên học sinh")
    so_cau:     int   = Field(..., gt=0, description="Số câu hỏi")
    thang_diem: float = Field(..., gt=0, description="Thang điểm tối đa")
    dap_an_mau: str   = Field(..., description="Đáp án mẫu")
    bai_lam:    str   = Field(..., description="Bài làm của học sinh")


@app.post("/grade", tags=["ocr"])
async def grade(req: GradeRequest) -> dict:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Chưa cấu hình GEMINI_API_KEY")

    logger.info(
        "[/grade] hoc_sinh=%r so_cau=%d thang_diem=%s",
        req.hoc_sinh, req.so_cau, req.thang_diem,
    )
    t0 = time.perf_counter()
    last_err: Exception | None = None
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            out = _grade_with_gemini(
                hoc_sinh=req.hoc_sinh,
                so_cau=req.so_cau,
                thang_diem=req.thang_diem,
                dap_an_mau=req.dap_an_mau,
                bai_lam=req.bai_lam,
            )
            dt = (time.perf_counter() - t0) * 1000
            logger.info(
                "[/grade] DONE %.0f ms → %s",
                dt, json.dumps(out, ensure_ascii=False),
            )
            return out
        except Exception as e:
            last_err = e
            logger.warning("[/grade] lần %d/%d fail: %s", attempt, GEMINI_RETRIES, e)
            msg = str(e)
            if any(c in msg for c in ("HTTP 429", "HTTP 401", "HTTP 403")):
                logger.warning("[/grade] lỗi quota/auth → bỏ retry")
                break

    raise HTTPException(status_code=502, detail=f"Gemini fail: {last_err}")


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("AI_server_2.main:app", host=host, port=port, reload=False)
