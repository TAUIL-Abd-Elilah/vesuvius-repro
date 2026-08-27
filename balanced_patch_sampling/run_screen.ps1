param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('baseline', 'cap025', 'cap075', 'cap080', 'noband')]
    [string]$Arm,
    [Parameter(Mandatory = $true)][string]$VillaSpiralDir,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$DatasetDir,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$CacheDir = (Join-Path $OutputRoot 'cache')
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $PSCommandPath
$config = Join-Path $here "config_screen_${Arm}_seed17.json"
$runRoot = Join-Path $OutputRoot "$Arm-seed17"
foreach ($path in @($VillaSpiralDir, $PythonExe, $DatasetDir, $config)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
if (Test-Path -LiteralPath $runRoot) { throw "Refusing to reuse output path: $runRoot" }
New-Item -ItemType Directory -Force -Path $runRoot, $CacheDir | Out-Null

$env:AGENTS_AGENT_MODE = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:WANDB_MODE = 'disabled'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = $VillaSpiralDir
$env:FIT_SPIRAL_CONFIG_OVERRIDES = Get-Content -LiteralPath $config -Raw
$env:FIT_SPIRAL_OUT_DIR = $runRoot
$env:FIT_SPIRAL_CACHE_DIR = $CacheDir
$env:FIT_SPIRAL_RUN_TAG = "$Arm-seed17"
$env:FIT_SPIRAL_SKIP_SAVE_MESH = '1'
$env:FIT_SPIRAL_SKIP_SAVE_OVERLAY = '1'

Push-Location -LiteralPath $VillaSpiralDir
try {
    & $PythonExe fit_spiral.py --dataset $DatasetDir --cache $CacheDir
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
