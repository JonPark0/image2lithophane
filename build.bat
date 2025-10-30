@echo off
REM ============================================
REM Lithophane Generator - Build Script
REM Builds standalone executable using PyInstaller
REM ============================================

echo.
echo ========================================
echo Lithophane Generator - Build
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo [WARNING] Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [1/5] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo [2/5] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

REM Install PyInstaller
echo.
echo [3/5] Installing PyInstaller...
pip install pyinstaller
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyInstaller!
    pause
    exit /b 1
)

REM Clean previous build
echo.
echo [4/5] Cleaning previous build...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "*.spec" del /q *.spec

REM Build executable
echo.
echo [5/5] Building executable...
echo This may take several minutes...
echo.

pyinstaller --name="LithophaneGenerator" ^
    --windowed ^
    --onefile ^
    --add-data="README.md;." ^
    --hidden-import=vtkmodules ^
    --hidden-import=vtkmodules.all ^
    --hidden-import=vtkmodules.qt.QVTKRenderWindowInteractor ^
    --hidden-import=vtkmodules.util ^
    --hidden-import=vtkmodules.util.numpy_support ^
    --collect-all=vtk ^
    --collect-all=vtkmodules ^
    viewer.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executable location:
echo   dist\LithophaneGenerator.exe
echo.
echo File size:
dir dist\LithophaneGenerator.exe | findstr "LithophaneGenerator.exe"
echo.
echo You can now distribute this single .exe file!
echo.
pause
