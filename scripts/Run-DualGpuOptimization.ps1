param(
    [string]$CondaEnv = "spaper",
    [ValidateSet("ft", "mlp", "both")]
    [string]$Model = "ft",
    [string]$Dataset = "unsw-nb15",
    [int[]]$Seeds = @(42, 3407, 8888, 123),
    [int]$Epochs = 15,
    [int]$BatchSize = 512,
    [double]$Epsilon = 0.02,
    [double]$Alpha = 0.005,
    [int]$Steps = 2,
    [double]$AdvWeight = 0.6,
    [int[]]$GpuIds = @(0, 1)
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

$LogDir = Join-Path $RootDir "outputs\logs\dual_gpu_runs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $LogDir "dual_gpu_optimization_$Timestamp.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $RunLog -Append
}

function Resolve-CondaPython {
    param([string]$EnvName)

    if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -eq $EnvName) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        $condaBase = (& conda info --base).Trim()
        $candidate = Join-Path $condaBase "envs\$EnvName\python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Cannot resolve python.exe for conda env '$EnvName'."
}

function New-SeedBatches {
    param([int[]]$AllSeeds, [int]$NumSlots)
    $batches = @()
    for ($i = 0; $i -lt $AllSeeds.Count; $i += $NumSlots) {
        $end = [Math]::Min($i + $NumSlots - 1, $AllSeeds.Count - 1)
        $batches += ,@($AllSeeds[$i..$end])
    }
    return $batches
}

function Start-TrainingProcess {
    param(
        [string]$PythonExe,
        [string]$ScriptPath,
        [int]$Seed,
        [int]$GpuId
    )

    $argList = @(
        $ScriptPath,
        "--dataset", $Dataset,
        "--seed", "$Seed",
        "--epochs", "$Epochs",
        "--batch_size", "$BatchSize",
        "--epsilon", "$Epsilon",
        "--alpha", "$Alpha",
        "--steps", "$Steps",
        "--adv_weight", "$AdvWeight"
    )

    $escapedArgs = $argList | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    $psi.Arguments = ($escapedArgs -join " ")
    $psi.WorkingDirectory = $RootDir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $psi.EnvironmentVariables["CUDA_VISIBLE_DEVICES"] = "$GpuId"
    $psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1"

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    [void]$process.Start()

    return [PSCustomObject]@{
        Process = $process
        Seed = $Seed
        GpuId = $GpuId
        ScriptPath = $ScriptPath
    }
}

function Wait-TrainingProcesses {
    param([object[]]$Handles)

    foreach ($handle in $Handles) {
        $proc = $handle.Process
        $proc.WaitForExit()
        $stdout = $proc.StandardOutput.ReadToEnd()
        $stderr = $proc.StandardError.ReadToEnd()

        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            $stdout.TrimEnd("`r", "`n") | Tee-Object -FilePath $RunLog -Append
        }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            $stderr.TrimEnd("`r", "`n") | Tee-Object -FilePath $RunLog -Append
        }
        if ($proc.ExitCode -ne 0) {
            throw "Training failed for seed=$($handle.Seed), gpu=$($handle.GpuId), script=$($handle.ScriptPath)"
        }
        Write-RunLog "Finished seed=$($handle.Seed) on GPU $($handle.GpuId): $($handle.ScriptPath)"
    }
}

function Invoke-Eval {
    param([string]$PythonExe)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    $psi.Arguments = "scripts/evaluate_main_results.py --dataset $Dataset"
    $psi.WorkingDirectory = $RootDir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    if ($stdout) { $stdout.TrimEnd("`r", "`n") | Tee-Object -FilePath $RunLog -Append }
    if ($stderr) { $stderr.TrimEnd("`r", "`n") | Tee-Object -FilePath $RunLog -Append }
    if ($process.ExitCode -ne 0) {
        throw "Evaluation failed for dataset=$Dataset"
    }
}

$PythonExe = Resolve-CondaPython -EnvName $CondaEnv
Write-RunLog "Repository: $RootDir"
Write-RunLog "Model: $Model"
Write-RunLog "Dataset: $Dataset"
Write-RunLog ("Seeds: {0}" -f ($Seeds -join ", "))
Write-RunLog ("GPU ids: {0}" -f ($GpuIds -join ", "))

$modelScripts = switch ($Model) {
    "ft" { @("src/baselines/bilevel_fttransformer_ids.py") }
    "mlp" { @("src/baselines/bilevel_supervised_ids.py") }
    "both" { @("src/baselines/bilevel_supervised_ids.py", "src/baselines/bilevel_fttransformer_ids.py") }
}

foreach ($script in $modelScripts) {
    Write-RunLog "=== Running $script on $Dataset ==="
    $batches = New-SeedBatches -AllSeeds $Seeds -NumSlots $GpuIds.Count
    foreach ($batch in $batches) {
        $handles = @()
        for ($i = 0; $i -lt $batch.Count; $i++) {
            $seed = $batch[$i]
            $gpuId = $GpuIds[$i]
            Write-RunLog "Launching seed=$seed on GPU ${gpuId}: $script"
            $handles += Start-TrainingProcess -PythonExe $PythonExe -ScriptPath $script -Seed $seed -GpuId $gpuId
        }
        Wait-TrainingProcesses -Handles $handles
    }
}

Invoke-Eval -PythonExe $PythonExe
Write-RunLog "Dual-GPU optimization run completed."
