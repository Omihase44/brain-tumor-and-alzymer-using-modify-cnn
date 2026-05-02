@echo off
REM Medical Imaging Platform - Quick Start Terminal Monitor
REM This script starts both Django and Flask applications with terminal output

echo ========================================
echo 🏥 MEDICAL IMAGING PLATFORM
echo ========================================
echo Starting applications with terminal output...
echo.
echo Django Web App will run on: http://127.0.0.1:8000/
echo Flask API Backend will run on: http://127.0.0.1:5000/
echo.
echo Press Ctrl+C to stop all applications
echo ========================================

REM Start the terminal control script
python terminal_control.py

pause