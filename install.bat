@echo off
rem Installs RealDenseFace Studio into this folder (see install.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
