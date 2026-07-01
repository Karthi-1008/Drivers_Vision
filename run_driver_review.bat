@echo off
setlocal

if "%~1"=="" (
    echo Usage: run_driver_review.bat "C:\path\to\frames"
    exit /b 1
)

python "%~dp0predict_driver_review.py" --frames "%~1"
