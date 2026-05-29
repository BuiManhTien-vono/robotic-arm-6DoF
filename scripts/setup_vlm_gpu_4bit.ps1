param(
    [string]$CudaWheel = "cu121"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$MinimumDriver = [version]"527.41"
$DriverText = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
if (-not $DriverText) {
    throw "nvidia-smi did not return a driver version. Install/update NVIDIA driver first."
}
$DriverVersion = [version]$DriverText
if ($DriverVersion -lt $MinimumDriver) {
    throw "NVIDIA driver $DriverText is too old for CUDA 12.1 PyTorch wheels. Update NVIDIA driver first; minimum is $MinimumDriver, latest Studio/Game Ready is recommended."
}

$BasePython = Join-Path $ProjectRoot ".venv_vlm\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Missing base Python: $BasePython"
}

$GpuVenv = Join-Path $ProjectRoot ".venv_vlm_gpu"
$GpuPython = Join-Path $GpuVenv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $GpuPython)) {
    & $BasePython -m venv $GpuVenv
}

& $GpuPython -m pip install --upgrade pip setuptools wheel
& $GpuPython -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CudaWheel"
& $GpuPython -m pip install -r ".\requirements_vlm_local.txt"
& $GpuPython -m pip install "bitsandbytes>=0.43.3"

@'
import importlib.util
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print("bitsandbytes", importlib.util.find_spec("bitsandbytes") is not None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is still not available in .venv_vlm_gpu.")
'@ | & $GpuPython -
