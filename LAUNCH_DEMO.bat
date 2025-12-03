@echo off
echo ============================================================
echo  OPTICAL COMMUNICATIONS ML - Interactive Web Demo
echo ============================================================
echo.

echo [1/2] Activating conda environment 'torch'...
call conda activate torch

echo.
echo [2/2] Launching interactive web demo...
echo.
echo The demo will open in your web browser automatically!
echo.
echo Controls:
echo  - Use sidebar to adjust parameters
echo  - See live visualizations
echo  - Compare model performance
echo.
echo To stop: Press Ctrl+C in this window
echo.
echo ============================================================

streamlit run demo_app.py

pause
