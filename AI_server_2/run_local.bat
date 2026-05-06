@echo off
REM =======================================================================
REM  AI_server_2 launcher (LOCAL) - dung VietOCR vgg_transformer
REM  Pipeline: page -> line detect (RuledPaper) -> crop -> VietOCR Transformer
REM  Cach dung: nhay dup chuot vao file nay, hoac chay tu PowerShell:
REM      d:\ki8\xla\BE_server\AI_server_2\run_local.bat
REM =======================================================================

setlocal

set "VENV_PY=C:\venvs\be_server\Scripts\python.exe"
set "PROJECT_DIR=d:\ki8\xla\BE_server"
set "HOST=0.0.0.0"
set "PORT=8000"

REM Tu dong lay IP WiFi hien tai (ten adapter "Wi-Fi")
for /f "tokens=2 delims=:" %%a in ('netsh interface ip show address "Wi-Fi" ^| findstr /i "IP Address"') do (
    for /f "tokens=1" %%b in ("%%a") do set "LAN_IP=%%b"
)

REM Weights cua VietOCR
set "VIETOCR_WEIGHTS=%PROJECT_DIR%\vgg_transformer.pth"
set "VIETOCR_CONFIG=vgg_transformer"

REM Device: de trong de auto-detect (dung cuda neu co, nguoc lai dung cpu)
set "VIETOCR_DEVICE="

REM Preprocess: "1" de bat preprocess crop (mac dinh tat - raw crop tot hon)
set "VIETOCR_PREPROCESS=0"

REM ---- Kiem tra ----
if not exist "%VENV_PY%" (
    echo [ERROR] Khong tim thay Python venv tai: %VENV_PY%
    pause
    exit /b 1
)

if not exist "%VIETOCR_WEIGHTS%" (
    echo [ERROR] Khong tim thay weights tai: %VIETOCR_WEIGHTS%
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

REM ---- Giai phong port neu dang bi chiem ----
echo [INFO] Kiem tra port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% "') do (
    echo [INFO] Kill PID %%a dang chiem port %PORT%
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo.
echo ============================================================
echo  AI_server_2 (VietOCR vgg_transformer)
echo  Weights    : %VIETOCR_WEIGHTS%
echo  Device     : %VIETOCR_DEVICE% (trong = auto)
echo  LocalURL   : http://127.0.0.1:%PORT%
echo  LAN URL    : http://%LAN_IP%:%PORT%
echo  Health     : http://127.0.0.1:%PORT%/health
echo  Predict    : POST http://%LAN_IP%:%PORT%/predict
echo  Swagger    : http://%LAN_IP%:%PORT%/docs
echo ============================================================
echo.

"%VENV_PY%" -m uvicorn AI_server_2.main:app --host %HOST% --port %PORT%

endlocal
pause
