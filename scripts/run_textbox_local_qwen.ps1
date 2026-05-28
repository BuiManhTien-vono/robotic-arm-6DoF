$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$GraspnetPython = Join-Path $ProjectRoot "graspnet-baseline\.venv\Scripts\python.exe"
$VlmPython = Join-Path $ProjectRoot ".venv_vlm\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $GraspnetPython)) {
    throw "Missing GraspNet/PyBullet Python: $GraspnetPython"
}
if (-not (Test-Path -LiteralPath $VlmPython)) {
    throw "Missing local Qwen Python: $VlmPython"
}

& $GraspnetPython ".\scripts\08_vlm_panda_textbox_app.py" `
    --vlm-backend qwen-local `
    --vlm-subprocess `
    --vlm-python $VlmPython `
    --vlm-device-map cpu `
    --fast-semantic `
    --vlm-offline `
    --vlm-max-pixels 50176 `
    --vlm-max-new-tokens 80 `
    --speed-scale 1.0
