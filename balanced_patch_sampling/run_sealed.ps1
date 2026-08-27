param(
    [Parameter(Mandatory = $true)][ValidateSet('baseline', 'cap075')][string]$Arm,
    [Parameter(Mandatory = $true)][ValidateSet(17, 23, 101)][int]$Seed,
    [Parameter(Mandatory = $true)][string]$VillaWorktree,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$FitDatasetDir,
    [Parameter(Mandatory = $true)][string]$SplitManifest,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$CacheDir = (Join-Path $OutputRoot 'cache')
)

# Portable sealed fit runner. Scoring is deliberately a separate explicit step:
# run_sealed_spiralcheck.py accepts the two completed checkpoint paths plus the
# held-out directory and refuses mismatched provenance.
$ErrorActionPreference = 'Stop'
$expectedCommit = '17dad916c79266f6a19f76abc507bb8b95c63a9b'
$evidence = Split-Path -Parent $PSCommandPath
$spiral = Join-Path $VillaWorktree 'spiral-fitting'
$config = Join-Path $evidence "config_sealed_${Arm}_seed$Seed.json"
$runRoot = Join-Path $OutputRoot "sealed-$Arm-seed$Seed"
foreach ($path in @($VillaWorktree, $spiral, $PythonExe, $FitDatasetDir, $SplitManifest, $config)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
if (Test-Path -LiteralPath $runRoot) { throw "Refusing to reuse output path: $runRoot" }
$commit = (& git -C $VillaWorktree rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $commit -ne $expectedCommit) { throw "Villa checkout is $commit; expected $expectedCommit" }
& git -C $VillaWorktree diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'Villa worktree has tracked modifications' }

New-Item -ItemType Directory -Force -Path $runRoot, $CacheDir | Out-Null
$preflight = [ordered]@{
    schema = 'sealed-fit-preflight-v1'; arm = $Arm; optimizer_seed = $Seed
    villa_commit = $commit; dataset = (Resolve-Path $FitDatasetDir).Path
    split_manifest = (Resolve-Path $SplitManifest).Path
    split_manifest_sha256 = (Get-FileHash -LiteralPath $SplitManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    config = (Resolve-Path $config).Path
    config_sha256 = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()
}
$preflight | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runRoot 'fit-preflight.json') -Encoding utf8

$env:AGENTS_AGENT_MODE = '1'; $env:WANDB_MODE = 'disabled'; $env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = $spiral; $env:FIT_SPIRAL_CONFIG_OVERRIDES = Get-Content -LiteralPath $config -Raw
$env:FIT_SPIRAL_OUT_DIR = $runRoot; $env:FIT_SPIRAL_CACHE_DIR = $CacheDir
$env:FIT_SPIRAL_RUN_TAG = "sealed-$Arm-seed$Seed"; $env:FIT_SPIRAL_SKIP_SAVE_MESH = '1'
$env:FIT_SPIRAL_SKIP_SAVE_OVERLAY = '1'
Push-Location -LiteralPath $spiral
try { & $PythonExe fit_spiral.py --dataset $FitDatasetDir --cache $CacheDir; exit $LASTEXITCODE }
finally { Pop-Location }
