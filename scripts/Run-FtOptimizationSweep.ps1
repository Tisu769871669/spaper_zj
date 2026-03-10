param(
    [string]$CondaEnv = "spaper",
    [string]$Dataset = "unsw-nb15",
    [int[]]$Seeds = @(42, 3407, 8888, 123),
    [int[]]$GpuIds = @(0, 1)
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

$LogDir = Join-Path $RootDir "outputs\logs\dual_gpu_runs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SweepLog = Join-Path $LogDir "ft_optimization_sweep_$Timestamp.log"

function Write-SweepLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $SweepLog -Append
}

$configs = @(
    @{
        Name = "ft_unsw_bs1024_lr7e4_aw060"
        Epochs = 15
        BatchSize = 1024
        Epsilon = 0.02
        Alpha = 0.005
        Steps = 2
        AdvWeight = 0.60
        LearningRate = 0.0007
        WeightDecay = 0.0001
        Dropout = 0.15
        DToken = 64
    },
    @{
        Name = "ft_unsw_bs1024_lr7e4_aw065"
        Epochs = 15
        BatchSize = 1024
        Epsilon = 0.02
        Alpha = 0.005
        Steps = 2
        AdvWeight = 0.65
        LearningRate = 0.0007
        WeightDecay = 0.0001
        Dropout = 0.15
        DToken = 64
    },
    @{
        Name = "ft_unsw_bs1024_lr5e4_eps015"
        Epochs = 15
        BatchSize = 1024
        Epsilon = 0.015
        Alpha = 0.004
        Steps = 2
        AdvWeight = 0.65
        LearningRate = 0.0005
        WeightDecay = 0.0001
        Dropout = 0.15
        DToken = 64
    }
)

foreach ($cfg in $configs) {
    Write-SweepLog "=== Starting config $($cfg.Name) on $Dataset ==="
    $cmd = @(
        ".\scripts\Run-DualGpuOptimization.ps1",
        "-CondaEnv", $CondaEnv,
        "-Model", "ft",
        "-Dataset", $Dataset,
        "-Epochs", $cfg.Epochs,
        "-BatchSize", $cfg.BatchSize,
        "-Epsilon", $cfg.Epsilon,
        "-Alpha", $cfg.Alpha,
        "-Steps", $cfg.Steps,
        "-AdvWeight", $cfg.AdvWeight,
        "-LearningRate", $cfg.LearningRate,
        "-WeightDecay", $cfg.WeightDecay,
        "-Dropout", $cfg.Dropout,
        "-DToken", $cfg.DToken,
        "-Seeds", ($Seeds -join ","),
        "-GpuIds", ($GpuIds -join ",")
    )

    $argString = $cmd -join " "
    Write-SweepLog "RUN: powershell -ExecutionPolicy Bypass -File $argString"
    powershell -ExecutionPolicy Bypass -File .\scripts\Run-DualGpuOptimization.ps1 `
        -CondaEnv $CondaEnv `
        -Model ft `
        -Dataset $Dataset `
        -Epochs $cfg.Epochs `
        -BatchSize $cfg.BatchSize `
        -Epsilon $cfg.Epsilon `
        -Alpha $cfg.Alpha `
        -Steps $cfg.Steps `
        -AdvWeight $cfg.AdvWeight `
        -LearningRate $cfg.LearningRate `
        -WeightDecay $cfg.WeightDecay `
        -Dropout $cfg.Dropout `
        -DToken $cfg.DToken `
        -Seeds $Seeds `
        -GpuIds $GpuIds

    if ($LASTEXITCODE -ne 0) {
        throw "Sweep failed for config $($cfg.Name)"
    }

    $datasetSlug = $Dataset.Replace("-", "_")
    $summaryPath = Join-Path $RootDir "outputs\results\main_results_summary_${datasetSlug}.csv"
    $detailedPath = Join-Path $RootDir "outputs\results\main_results_detailed_${datasetSlug}.csv"
    if (Test-Path $summaryPath) {
        Copy-Item $summaryPath (Join-Path $RootDir "outputs\results\main_results_summary_${datasetSlug}_$($cfg.Name).csv") -Force
    }
    if (Test-Path $detailedPath) {
        Copy-Item $detailedPath (Join-Path $RootDir "outputs\results\main_results_detailed_${datasetSlug}_$($cfg.Name).csv") -Force
    }

    Write-SweepLog "=== Finished config $($cfg.Name) ==="
}

Write-SweepLog "FT optimization sweep completed."
