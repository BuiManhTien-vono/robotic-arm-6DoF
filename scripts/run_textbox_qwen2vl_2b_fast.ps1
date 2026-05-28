$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$GraspnetPython = Join-Path $ProjectRoot "graspnet-baseline\.venv\Scripts\python.exe"
$VlmPython = Join-Path $ProjectRoot ".venv_vlm\Scripts\python.exe"
$ModelCache = Join-Path $ProjectRoot ".hf_cache\hub\models--Qwen--Qwen2-VL-2B-Instruct"

if (-not (Test-Path -LiteralPath $GraspnetPython)) {
    throw "Missing GraspNet/PyBullet Python: $GraspnetPython"
}
if (-not (Test-Path -LiteralPath $VlmPython)) {
    throw "Missing local Qwen Python: $VlmPython"
}

$env:HF_HOME = Join-Path $ProjectRoot ".hf_cache"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_ETAG_TIMEOUT = "1"

$OfflineArg = "--vlm-offline"
if (-not (Test-Path -LiteralPath $ModelCache)) {
    Write-Host "Qwen/Qwen2-VL-2B-Instruct is not cached yet. First run will download it."
    $OfflineArg = "--no-vlm-offline"
}

& $GraspnetPython ".\scripts\08_vlm_panda_textbox_app.py" `
    --vlm-backend qwen-local `
    --vlm-model "Qwen/Qwen2-VL-2B-Instruct" `
    --vlm-subprocess `
    --vlm-python $VlmPython `
    --vlm-device-map cpu `
    --no-fast-semantic `
    $OfflineArg `
    --camera-width 640 `
    --camera-height 480 `
    --vlm-max-pixels 25088 `
    --vlm-max-new-tokens 64 `
    --speed-scale 1.0
