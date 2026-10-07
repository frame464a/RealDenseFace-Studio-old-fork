# RealDenseFace Studio installer (Windows, NVIDIA GPU).
# Creates a self-contained Python environment in this folder; nothing is installed system-wide.
# Run via install.bat.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot

# Prebuilt CUDA extension wheels. A local .\wheels folder takes precedence (offline / testing).
$ReleaseUrl = "https://github.com/frame464a/RealDenseFace-Studio/releases/download/v0.1.0"
$Wheels = @(
    "flame_solver-0.0.0-cp311-cp311-win_amd64.whl",
    "nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl",
    "chumpy-0.70-py3-none-any.whl"
)
$TorchIndex = "https://download.pytorch.org/whl/cu128"

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) {
    Write-Host ""
    Write-Host "ERROR: $text" -ForegroundColor Red
    exit 1
}

Write-Host "RealDenseFace Studio installer" -ForegroundColor White
Write-Host "Install folder: $PSScriptRoot"

# --- GPU check ------------------------------------------------------------------------------
Step "Checking for an NVIDIA GPU"
$gpu = $null
try { $gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null } catch {}
if (-not $gpu) {
    Fail "No NVIDIA GPU / driver found (nvidia-smi failed). RealDenseFace Studio needs an NVIDIA RTX GPU with a recent driver."
}
Write-Host "Found: $($gpu | Select-Object -First 1)"
$driver = [double](($gpu | Select-Object -First 1).Split(",")[1].Trim().Split(".")[0])
if ($driver -lt 570) {
    Fail "Your NVIDIA driver is too old for CUDA 12.8 (need 570 or newer). Update it from nvidia.com and run the installer again."
}

# --- uv (Python package manager) ---------------------------------------------------------------
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot "tools\python"
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot "tools\cache"
$uv = Join-Path $PSScriptRoot "tools\uv.exe"
if (-not (Test-Path $uv)) {
    Step "Downloading uv (Python package manager)"
    $env:UV_INSTALL_DIR = Join-Path $PSScriptRoot "tools"
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    if (-not (Test-Path $uv)) { Fail "Could not download uv." }
}

# --- Python environment -------------------------------------------------------------------------
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Step "Creating Python 3.11 environment"
    & $uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the Python environment." }
}

Step "Installing PyTorch with CUDA 12.8 (about 3 GB, this takes a while)"
& $uv pip install --python $py torch==2.11.0 torchvision==0.26.0 --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed." }

Step "Installing dependencies"
& $uv pip install --python $py -r requirements-studio.txt
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed." }

Step "Installing prebuilt GPU components"
$localWheels = Join-Path $PSScriptRoot "wheels"
New-Item -ItemType Directory -Force $localWheels | Out-Null
$wheelPaths = @()
foreach ($wheel in $Wheels) {
    $path = Join-Path $localWheels $wheel
    if (-not (Test-Path $path)) {
        Write-Host "Downloading $wheel"
        Invoke-WebRequest "$ReleaseUrl/$wheel" -OutFile $path
    }
    $wheelPaths += $path
}
& $uv pip install --python $py --no-deps $wheelPaths
if ($LASTEXITCODE -ne 0) { Fail "Installing the GPU components failed." }

Step "Checking the installation"
& $py -c "import torch, flame_solver, nvdiffrast.torch, pxr, PySide6; assert torch.cuda.is_available(), 'CUDA not available'; print('OK:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { Fail "The installation check failed. See the messages above." }

# --- Shortcut ---------------------------------------------------------------------------------
Step "Creating desktop shortcut"
try {
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "RealDenseFace Studio.lnk"))
    $lnk.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $lnk.Arguments = "app.py"
    $lnk.WorkingDirectory = $PSScriptRoot
    $lnk.Save()
    Write-Host "Shortcut created on your desktop."
} catch {
    Write-Host "Could not create a desktop shortcut; start the app with 'RealDenseFace Studio.bat'."
}

Write-Host ""
Write-Host "Done! Start RealDenseFace Studio from the desktop shortcut or 'RealDenseFace Studio.bat'." -ForegroundColor Green
Write-Host "On first start it will download the model weights and help you get the FLAME head model."
