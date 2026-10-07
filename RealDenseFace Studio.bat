@echo off
rem Starts RealDenseFace Studio. Run install.bat first.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo RealDenseFace Studio is not installed yet. Run install.bat first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" app.py
