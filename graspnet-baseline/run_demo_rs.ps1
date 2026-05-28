param(
    [switch]$NoVis,
    [int]$NumPoint = 20000
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Repo ".venv\Scripts\python.exe"

if (!(Test-Path $Python)) {
    $Python = "python"
}

$Checkpoint = Join-Path $Repo "checkpoints\checkpoint-rs.tar"
$DataDir = Join-Path $Repo "doc\example_data"
$OutputDir = Join-Path $Repo "outputs"
$OutputGrasps = Join-Path $OutputDir "demo_grasps.npy"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $Python (Join-Path $Repo "check_env.py")
$DemoArgs = @(
    (Join-Path $Repo "demo.py"),
    "--checkpoint_path", $Checkpoint,
    "--data_dir", $DataDir,
    "--num_point", $NumPoint,
    "--save_grasps_path", $OutputGrasps
)
if ($NoVis) {
    $DemoArgs += "--no_vis"
}

& $Python @DemoArgs
