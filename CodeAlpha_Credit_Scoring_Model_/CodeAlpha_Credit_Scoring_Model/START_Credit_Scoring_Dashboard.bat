@echo off
setlocal
cd /d "%~dp0"

title CodeAlpha Credit Scoring Dashboard

echo ============================================
echo   CodeAlpha Credit Scoring Dashboard
echo ============================================
echo.

REM Check Python
py --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found.
    echo Please install Python first.
    pause
    exit /b 1
)

REM Check Streamlit and install project dependencies automatically if needed
py -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit is not installed.
    echo Installing project requirements...
    py -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installation failed. Please run:
        echo py -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo Starting the dashboard...
start "" cmd /c "py -m streamlit run app.py --server.headless true"

echo Waiting for the dashboard...
timeout /t 5 /nobreak >nul

echo Opening Chrome...
start "" "http://localhost:8501"

echo.
echo Dashboard is running at:
echo http://localhost:8501
echo.
echo Keep this window open while using the dashboard.
echo Close the Streamlit window to stop the application.
pause
