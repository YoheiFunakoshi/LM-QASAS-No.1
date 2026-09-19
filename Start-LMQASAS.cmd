@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 専用Python環境がありません。docs\SETUP.md を確認してください。
  pause
  exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" "scripts\start_app.py"
if errorlevel 1 pause
