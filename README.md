# BE — Vietnamese Handwriting OCR Backend

Thư mục này chứa 2 server FastAPI nhận diện chữ viết tay tiếng Việt.

---

## Cấu trúc

```
BE/
├── AI_server/          # Server 1: CRNN CNNBiLSTMCTC 
│   ├── checkpoints/
│   │   └── best_model.pth
│   └── run_local.bat
└── AI_server_2/        # Server 2: VietOCR vgg_transformer 
    └── run_local.bat
```

---

## Yêu cầu

```
Python venv: C:\venvs\be_server
fastapi, uvicorn, opencv-python-headless, torch, vietocr, pillow
```

---

## Cách chạy

### AI_server_2 — VietOCR (khuyến nghị)

```bat
AI_server_2\run_local.bat
```

Hoặc từ PowerShell tại `D:\ki8\xla\BE_server`:

```powershell
C:\venvs\be_server\Scripts\python.exe -m uvicorn AI_server_2.main:app --host 0.0.0.0 --port 8000
```

### AI_server — CRNN (thực nghiệm)

```bat
AI_server\run_local.bat
```

> **Lưu ý:** Weights hiện tại của AI_server bị degenerate (lỗi training). Kết quả nhận dạng không chính xác. Dùng AI_server_2 thay thế.

---

## API Endpoints

Cả hai server có cùng format JSON.

### `GET /health`

```json
{
  "status": "ok",
  "model": "vgg_transformer",
  "device": "cpu"
}
```

### `POST /predict`

Upload ảnh multipart (`file`), trả về các dòng đã nhận dạng:

```json
{
  "num_lines": 3,
  "lines": [
    {
      "line_index": 0,
      "text": "Câu 1. Phân loại các loại áo quần",
      "bbox": { "x1": 10, "y1": 5, "x2": 980, "y2": 60 }
    }
  ],
  "full_text": "Câu 1. Phân loại các loại áo quần\n..."
}
```

### `POST /grade`

Chấm điểm bài làm bằng Gemini API:

```json
// Request
{
  "hoc_sinh": "Nguyễn Văn A",
  "so_cau": 4,
  "thang_diem": 10,
  "dap_an_mau": "...",
  "bai_lam": "..."
}

// Response
{
  "diem": 7.5,
  "nhan_xet": "Học sinh trả lời đúng 3/4 câu, câu 2 còn thiếu ý."
}
```

---

## Biến môi trường

| Biến | Server | Mặc định | Mô tả |
|------|--------|----------|-------|
| `VIETOCR_WEIGHTS` | AI_server_2 | `./vgg_transformer.pth` | Đường dẫn weights |
| `VIETOCR_PREPROCESS` | AI_server_2 | `0` | `"1"` để bật preprocess crop |
| `CRNN_CHECKPOINT` | AI_server | `./checkpoints/best_model.pth` | Đường dẫn checkpoint |
| `GEMINI_API_KEY` | cả hai | (hardcoded) | Gemini API key |
| `PORT` | cả hai | `8000` | Port lắng nghe |

---

## Swagger UI

Sau khi chạy server: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
