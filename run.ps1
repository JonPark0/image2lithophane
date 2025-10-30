# ============================================
# Lithophane Generator - Run Script (PowerShell)
# This script activates venv and runs the app
# ============================================

Write-Host ""
Write-Host "========================================"
Write-Host "Lithophane Generator"
Write-Host "========================================"
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first to set up the environment."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Python is installed
try {
    python --version | Out-Null
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "[INFO] Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host ""
    Write-Host "If you see an error about execution policy, run PowerShell as Administrator and execute:"
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if viewer.py exists
if (-not (Test-Path "viewer.py")) {
    Write-Host "[ERROR] viewer.py not found in current directory!" -ForegroundColor Red
    Write-Host "Please make sure you're running this script from the project root."
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the application
Write-Host "[INFO] Starting Lithophane Generator..."
Write-Host ""
python viewer.py

# Check if the application exited with an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Application exited with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
}

Write-Host ""
Write-Host "Application closed."
Read-Host "Press Enter to exit"
