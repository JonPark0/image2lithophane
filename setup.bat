@echo off
REM ============================================
REM Lithophane Generator - Setup Script
REM This script sets up the virtual environment
REM and installs all required dependencies
REM ============================================

echo.
echo ========================================
echo Lithophane Generator - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/4] Python found:
python --version
echo.

REM Check if venv already exists
if exist "venv\" (
    echo [2/4] Virtual environment already exists.
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 (
        echo Keeping existing virtual environment...
        goto :install_packages
    )
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo [2/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

:install_packages
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

echo [4/4] Installing required packages...
echo This may take a few minutes...
echo.

python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install required packages!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo You can now run the application using:
echo   run.bat
echo.
echo Or manually:
echo   venv\Scripts\activate
echo   python viewer.py
echo.
pause
