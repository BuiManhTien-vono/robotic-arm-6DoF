param(
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6",
    [string]$VcVars = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Repo ".venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    throw "Missing .venv. Create it first with: uv venv --python cpython-3.10.20-windows-x86_64-none .venv"
}

if (!(Test-Path $CudaPath)) {
    throw "CUDA Toolkit not found at $CudaPath. Install CUDA Toolkit 11.6 or pass -CudaPath to the matching toolkit."
}

if (!(Test-Path $VcVars)) {
    throw "MSVC vcvars64.bat not found at $VcVars. Install Visual Studio C++ Build Tools or pass -VcVars."
}

$env:CUDA_PATH = $CudaPath
$env:PATH = "$CudaPath\bin;$CudaPath\libnvvp;$env:PATH"

cmd /c "`"$VcVars`" && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}

Write-Host "Python:" (& $VenvPython -c "import sys; print(sys.version)")
Write-Host "CUDA_PATH: $env:CUDA_PATH"
nvcc --version
cl

& $VenvPython -m pip install --timeout 1000 --retries 10 torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
& $VenvPython -m pip install --timeout 1000 --retries 10 tensorboard numpy==1.24.4 scipy==1.10.1 open3d==0.19.0 Pillow tqdm

Push-Location (Join-Path $Repo "pointnet2")
& $VenvPython setup.py install
Pop-Location

Push-Location (Join-Path $Repo "knn")
& $VenvPython setup.py install
Pop-Location

& $VenvPython -m pip install "git+https://github.com/graspnet/graspnetAPI.git"
& $VenvPython -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)"
