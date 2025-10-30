@echo off
REM ============================================
REM Lithophane Generator - Run Script
REM This script activates venv and runs the app
REM ============================================

echo.
echo ========================================
echo Lithophane Generator
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Check if viewer.py exists
if not exist "viewer.py" (
    echo [ERROR] viewer.py not found in current directory!
    echo Please make sure you're running this script from the project root.
    pause
    exit /b 1
)

REM Run the application
echo [INFO] Starting Lithophane Generator...
echo.
python viewer.py

REM Check if the application exited with an error
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code: %errorlevel%
    echo.
)

echo.
echo Application closed.
pause
