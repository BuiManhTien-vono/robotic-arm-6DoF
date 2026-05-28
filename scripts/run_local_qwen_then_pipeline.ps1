$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$env:HF_HOME = Join-Path $ProjectRoot ".hf_cache"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:QWEN_VL_MAX_PIXELS = "50176"
$env:QWEN_VL_MAX_NEW_TOKENS = "80"
$env:QWEN_VL_DEVICE_MAP = "cpu"

$VlmPython = Join-Path $ProjectRoot ".venv_vlm\Scripts\python.exe"
$GraspnetPython = Join-Path $ProjectRoot "graspnet-baseline\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VlmPython)) {
    throw "Missing VLM Python: $VlmPython"
}
if (-not (Test-Path -LiteralPath $GraspnetPython)) {
    throw "Missing GraspNet Python: $GraspnetPython"
}

$Command = if ($args.Count -gt 0) { $args -join " " } else { "red" }

Write-Host "=== STEP 1: RUN LOCAL QWEN2.5-VL MODEL ==="
Write-Host "Command: $Command"
Write-Host "This step loads Qwen/Qwen2.5-VL-3B-Instruct locally. On CPU it can take several minutes."

& $VlmPython ".\scripts\01_test_vlm.py" `
    --image ".\graspnet-baseline\doc\example_data\color.png" `
    --command $Command `
    --output-json "data\outputs\vlm_result.json" `
    --output-image "data\outputs\vlm_bbox.png"

Write-Host "=== STEP 2: RUN GRASPNET + PYBULLET FROM QWEN OUTPUT ==="

& $GraspnetPython ".\scripts\05_run_full_pipeline.py" `
    --use-existing-vlm `
    --vlm-result "data\outputs\vlm_result.json" `
    --output-dir "data\outputs\full_pipeline_local_qwen"

Write-Host "=== DONE ==="
Write-Host "VLM JSON: data\outputs\vlm_result.json"
Write-Host "VLM bbox image: data\outputs\vlm_bbox.png"
Write-Host "Full pipeline result: data\outputs\full_pipeline_local_qwen\full_pipeline_result.json"
