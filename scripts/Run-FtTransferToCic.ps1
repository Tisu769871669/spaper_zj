param(
    [string]$CondaEnv = "spaper",
    [int[]]$Seeds = @(42, 3407, 8888, 123),
    [int[]]$GpuIds = @(0, 1)
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "Running FT transfer configuration on CIC-IDS2017-random..."
Write-Host "Best source config: batch=1024, epsilon=0.015, alpha=0.004, steps=2, adv_weight=0.65, lr=5e-4, wd=1e-4, dropout=0.15, d_token=64"

& .\scripts\Run-DualGpuOptimization.ps1 `
    -CondaEnv $CondaEnv `
    -Model ft `
    -Dataset "cic-ids2017-random" `
    -Epochs 15 `
    -BatchSize 1024 `
    -Epsilon 0.015 `
    -Alpha 0.004 `
    -Steps 2 `
    -AdvWeight 0.65 `
    -LearningRate 0.0005 `
    -WeightDecay 0.0001 `
    -Dropout 0.15 `
    -DToken 64 `
    -Seeds $Seeds `
    -GpuIds $GpuIds

if ($LASTEXITCODE -ne 0) {
    throw "FT transfer run on CIC-IDS2017-random failed."
}
