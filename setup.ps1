# ============================================
# Lithophane Generator - Setup Script (PowerShell)
# This script sets up the virtual environment
# and installs all required dependencies
# ============================================

Write-Host ""
Write-Host "========================================"
Write-Host "Lithophane Generator - Setup"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[1/4] Python found: $pythonVersion"
    Write-Host ""
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/"
    Write-Host "Make sure to check 'Add Python to PATH' during installation."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if venv already exists
if (Test-Path "venv") {
    Write-Host "[2/4] Virtual environment already exists."
    $recreate = Read-Host "Do you want to recreate it? (y/N)"
    if ($recreate -eq "y" -or $recreate -eq "Y") {
        Write-Host "Removing old virtual environment..."
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "Keeping existing virtual environment..."
        $skipVenvCreation = $true
    }
}

if (-not $skipVenvCreation) {
    Write-Host "[2/4] Creating virtual environment..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created successfully."
    Write-Host ""
}

Write-Host "[3/4] Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host ""
    Write-Host "If you see an error about execution policy, run PowerShell as Administrator and execute:"
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

Write-Host "[4/4] Installing required packages..."
Write-Host "This may take a few minutes..."
Write-Host ""

python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}

pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install required packages!" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "You can now run the application using:"
Write-Host "  .\run.ps1"
Write-Host ""
Write-Host "Or manually:"
Write-Host "  venv\Scripts\Activate.ps1"
Write-Host "  python viewer.py"
Write-Host ""
Read-Host "Press Enter to exit"
