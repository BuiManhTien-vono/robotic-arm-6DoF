$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot "graspnet-baseline\.venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing GraspNet/PyBullet Python: $Python"
}
if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing .env file. Add GEMINI_API_KEY and GEMINI_MODEL first."
}

$ApiKeyConfigured = $false
foreach ($Line in Get-Content -LiteralPath $EnvFile) {
    if ($Line -match "^\s*GEMINI_API_KEY\s*=\s*(.+?)\s*$") {
        $Value = $Matches[1].Trim().Trim('"').Trim("'")
        if ($Value -and $Value -ne "your_api_key_here") {
            $ApiKeyConfigured = $true
        }
    }
}
if (-not $ApiKeyConfigured) {
    throw "GEMINI_API_KEY is missing or still uses the placeholder in .env."
}

& $Python ".\scripts\08_vlm_panda_textbox_app.py" `
    --vlm-backend gemini `
    --no-vlm-subprocess `
    --no-vlm-keepalive `
    --no-vlm-preload `
    --no-fast-semantic `
    --camera-width 640 `
    --camera-height 480 `
    --speed-scale 1.0
