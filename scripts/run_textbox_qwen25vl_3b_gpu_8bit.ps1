$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$GraspnetPython = Join-Path $ProjectRoot "graspnet-baseline\.venv\Scripts\python.exe"
$VlmPython = Join-Path $ProjectRoot ".venv_vlm_gpu\Scripts\python.exe"
$ModelCache = Join-Path $ProjectRoot ".hf_cache\hub\models--Qwen--Qwen2.5-VL-3B-Instruct"
$RequiredCacheFiles = @(
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt"
)

if (-not (Test-Path -LiteralPath $GraspnetPython)) {
    throw "Missing GraspNet/PyBullet Python: $GraspnetPython"
}
if (-not (Test-Path -LiteralPath $VlmPython)) {
    throw "Missing GPU VLM Python: $VlmPython. Run .\scripts\setup_vlm_gpu_4bit.ps1 first."
}

$env:HF_HOME = Join-Path $ProjectRoot ".hf_cache"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_ETAG_TIMEOUT = "1"
Remove-Item Env:\QWEN_VL_4BIT -ErrorAction SilentlyContinue
$env:QWEN_VL_8BIT = "1"
$env:QWEN_VL_TORCH_DTYPE = "float16"
$env:QWEN_VL_ATTENTION_IMPL = "eager"
$env:QWEN_VL_MAX_MEMORY_GPU = "3.2GiB"
$env:QWEN_VL_MAX_MEMORY_CPU = "6GiB"

$OfflineArg = "--vlm-offline"
$SnapshotDir = $null
if (Test-Path -LiteralPath (Join-Path $ModelCache "snapshots")) {
    $SnapshotDir = Get-ChildItem -LiteralPath (Join-Path $ModelCache "snapshots") -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}
$MissingCacheFiles = @()
if ($null -eq $SnapshotDir) {
    $MissingCacheFiles = $RequiredCacheFiles
} else {
    foreach ($FileName in $RequiredCacheFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $SnapshotDir.FullName $FileName))) {
            $MissingCacheFiles += $FileName
        }
    }
}
if ($MissingCacheFiles.Count -gt 0) {
    Write-Host "Qwen/Qwen2.5-VL-3B-Instruct cache is incomplete. Missing: $($MissingCacheFiles -join ', ')"
    Write-Host "This run will use internet to finish the cache. Later runs can use offline mode."
    $OfflineArg = "--no-vlm-offline"
}

& $GraspnetPython ".\scripts\08_vlm_panda_textbox_app.py" `
    --vlm-backend qwen-local `
    --vlm-model "Qwen/Qwen2.5-VL-3B-Instruct" `
    --vlm-subprocess `
    --vlm-keepalive `
    --vlm-preload `
    --vlm-stop-before-graspnet `
    --vlm-python $VlmPython `
    --vlm-device-map auto `
    --no-fast-semantic `
    $OfflineArg `
    --camera-width 640 `
    --camera-height 480 `
    --vlm-max-pixels 25088 `
    --vlm-max-new-tokens 96 `
    --speed-scale 1.0
