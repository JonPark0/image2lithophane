# ============================================
# Lithophane Generator - Build Script (PowerShell)
# Builds standalone executable using PyInstaller
# ============================================

Write-Host ""
Write-Host "========================================"
Write-Host "Lithophane Generator - Build"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion"
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "[WARNING] Virtual environment not found!" -ForegroundColor Yellow
    Write-Host "Creating virtual environment..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "[1/5] Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate virtual environment!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "[2/5] Installing dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install PyInstaller
Write-Host ""
Write-Host "[3/5] Installing PyInstaller..."
pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PyInstaller!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Clean previous build
Write-Host ""
Write-Host "[4/5] Cleaning previous build..."
if (Test-Path "build") { Remove-Item -Recurse -Force build }
if (Test-Path "dist") { Remove-Item -Recurse -Force dist }
if (Test-Path "*.spec") { Remove-Item -Force *.spec }

# Build executable
Write-Host ""
Write-Host "[5/5] Building executable..."
Write-Host "This may take several minutes..."
Write-Host ""

pyinstaller --name="LithophaneGenerator" `
    --windowed `
    --onefile `
    --add-data="README.md;." `
    --hidden-import=vtkmodules `
    --hidden-import=vtkmodules.all `
    --hidden-import=vtkmodules.qt.QVTKRenderWindowInteractor `
    --hidden-import=vtkmodules.util `
    --hidden-import=vtkmodules.util.numpy_support `
    --collect-all=vtk `
    --collect-all=vtkmodules `
    viewer.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "Executable location:"
Write-Host "  dist\LithophaneGenerator.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "File size:"
Get-Item "dist\LithophaneGenerator.exe" | ForEach-Object {
    $size = "{0:N2}" -f ($_.Length / 1MB)
    Write-Host "  $size MB"
}
Write-Host ""
Write-Host "You can now distribute this single .exe file!"
Write-Host ""
Read-Host "Press Enter to exit"
