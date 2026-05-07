"""FastAPI server — CRNN Handwriting OCR.

Endpoints:
    GET  /health    → liveness check
    POST /predict   → upload ảnh, trả về text từng dòng
    POST /grade     → chấm điểm bài làm bằng Gemini (JSON)

Env vars:
    CRNN_CHECKPOINT   path tới file .pth          (default: ./checkpoints/best_model.pth)
    CRNN_DEVICE       "cpu" | "cuda" | ""         (default: auto)
    HOST              bind host                   (default: 0.0.0.0)
    PORT              bind port                   (default: 8000)
    GEMINI_API_KEY    Gemini API key
    GEMINI_MODEL      model name                  (default: gemini-flash-latest)
    GEMINI_TIMEOUT    timeout giây                (default: 25)
    GEMINI_RETRIES    số lần retry                (default: 3)
"""
from __future__ import annotations

import json
import logging
import os
import sys
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


_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from .line_detector import RuledPaperLineDetector  # noqa: E402
from .recognizer import CRNNRecognizer                       # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
# Tat debug spam cua thu vien ngoai
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config từ env ─────────────────────────────────────────────────────────────
_DEFAULT_CKPT = Path(__file__).parent / "checkpoints" / "best_model.pth"
CHECKPOINT    = Path(os.getenv("CRNN_CHECKPOINT", str(_DEFAULT_CKPT)))
DEVICE        = os.getenv("CRNN_DEVICE", "")          # "" → auto

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDpa7foCo-Yltiovn1tZFAn5Uur8zKsETk")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-flash-latest")
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "25"))
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES",   "3"))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CRNN Handwriting OCR",
    description="CNN-BiLSTM-CTC model nhận diện chữ viết tay tiếng Việt",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model instances ─────────────────────────────────────────────────────
_line_detector: RuledPaperLineDetector | None = None
_recognizer:    CRNNRecognizer | None         = None


def deskew_page(img_bgr: np.ndarray) -> np.ndarray:
    """Làm thẳng ảnh toàn trang trước khi detect dòng."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Binary, chữ trắng nền đen
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm góc nghiêng qua tất cả pixel chữ
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 100:
        return img_bgr  # không đủ pixel để tính
    
    angle = cv2.minAreaRect(coords)[2]
    
    # Chuẩn hóa góc về (-45, 45)
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Chỉ deskew nếu nghiêng đáng kể (tránh rotate ảnh thẳng)
    if abs(angle) < 0.3:
        return img_bgr
    
    print(f"[DEBUG] Deskewing page by {angle:.2f} degrees")
    
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    
    # Tính kích thước ảnh mới để không bị crop góc
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(
        img_bgr, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # nền trắng
    )
    return rotated


@app.on_event("startup")
def _load_models() -> None:
    global _line_detector, _recognizer

    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Không tìm thấy checkpoint: {CHECKPOINT}\n"
            f"Hãy đặt file vào đó hoặc set env CRNN_CHECKPOINT."
        )

    logger.info("Loading line detector …")
    _line_detector = RuledPaperLineDetector()

    logger.info("Loading CRNN model từ %s …", CHECKPOINT)
    _recognizer = CRNNRecognizer(
        checkpoint_path=CHECKPOINT,
        device=DEVICE or None,
    )
    logger.info("Models ready.")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
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
    print("[DEBUG] /health endpoint called")
    result = {
        "status": "ok",
        "checkpoint": str(CHECKPOINT),
        "device": str(_recognizer.device) if _recognizer else "not_loaded",
    }
    print(f"[DEBUG] /health response: {result}")
    return result


@app.post("/predict", response_model=PredictResponse, tags=["ocr"])
async def predict(file: UploadFile = File(..., description="Ảnh chứa chữ viết tay")) -> PredictResponse:
    print(f"\n{'='*80}")
    print(f"[DEBUG] /predict endpoint called with file: {file.filename}")
    print(f"{'='*80}")
    try:
        data = await file.read()
        print(f"[DEBUG] File size: {len(data)} bytes")
        arr  = np.frombuffer(data, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        print(f"[DEBUG] Image decoded, shape: {img.shape if img is not None else 'None'}")
        if img is not None:
            print(f"[DEBUG] Image dtype: {img.dtype}, min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}")

        if img is None:
            print("[DEBUG] Image decode failed")
            raise HTTPException(status_code=400, detail="Không đọc được ảnh...")
        
        # --- THÊM VÀO ĐÂY ---
        print("[DEBUG] Deskewing image before detection...")
        img = deskew_page(img)  # Chú ý: phải gán lại img = ...
        # --------------------

        print("[DEBUG] Starting line detection...")
        boxes = _line_detector.detect(img)
        print(f"[DEBUG] Detected {len(boxes)} lines")
        
        for i, b in enumerate(boxes):
            print(f"[DEBUG]   Line {i}: bbox=({b.x1}, {b.y1}) → ({b.x2}, {b.y2}), size=({b.x2-b.x1}, {b.y2-b.y1})")
        
        if not boxes:
            logger.info("[predict] %s → 0 dòng", file.filename)
            print("[DEBUG] No lines detected, returning empty response")
            return PredictResponse(num_lines=0, lines=[], full_text="")
        
        deskew_page(img)
        print(f"\n[DEBUG] Extracting {len(boxes)} crops...")
        crops = []
        pad_x = 45  # Nới rộng lề trái/phải 20 pixel để không lẹm chữ
        pad_y = 15 # Nới trên/dưới 5 pixel
        
        for i, b in enumerate(boxes):
            # Tính toán tọa độ mới, đảm bảo không vượt quá kích thước ảnh gốc
            x1_safe = max(0, b.x1 - pad_x)
            y1_safe = max(0, b.y1 - pad_y)
            x2_safe = min(img.shape[1], b.x2 + pad_x)
            y2_safe = min(img.shape[0], b.y2 + pad_y)
            
            crop = img[y1_safe:y2_safe, x1_safe:x2_safe]
            crops.append(crop)
            print(f"[DEBUG]   Crop {i}: shape={crop.shape}, dtype={crop.dtype}, min={crop.min()}, max={crop.max()}, mean={crop.mean():.2f}")
        
        print(f"\n[DEBUG] Running CRNN prediction on {len(crops)} crops...")
        texts = _recognizer.predict_batch(crops)
        
        for i, text in enumerate(texts):
            print(f"[DEBUG]   Prediction {i}: '{text}'")

        lines = [
            LineResult(line_index=i, text=text, bbox=BBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2))
            for i, (b, text) in enumerate(zip(boxes, texts))
        ]

        logger.info("[predict] %s → %d dòng: %s", file.filename, len(lines), texts)
        print(f"\n[DEBUG] Returning {len(lines)} lines")
        full_text_joined = "\n".join(r.text for r in lines)
        print(f"[DEBUG] Full text: {repr(full_text_joined)}")
        print(f"{'='*80}\n")

        return PredictResponse(
            num_lines=len(lines),
            lines=lines,
            full_text="\n".join(r.text for r in lines),
        )
    except Exception as e:
        print(f"[DEBUG] ERROR in /predict: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


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
    uvicorn.run("AI_server.main:app", host=host, port=port, reload=False)
