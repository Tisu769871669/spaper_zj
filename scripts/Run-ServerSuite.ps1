param(
    [string]$CondaEnv = "spaper",
    [ValidateSet("core", "all", "supplement")]
    [string]$Suite = "core",
    [int[]]$Seeds = @(42, 3407, 8888, 123),
    [switch]$DisableFtCic,
    [switch]$DisableCiciot,
    [switch]$DisableLegacyAnalysis
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

$LogDir = Join-Path $RootDir "outputs\logs\server_runs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $LogDir "run_server_suite_$Timestamp.log"

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

function Invoke-Python {
    param(
        [string]$PythonExe,
        [string[]]$Args
    )

    Write-RunLog ("RUN: {0} {1}" -f $PythonExe, ($Args -join " "))

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    foreach ($arg in $Args) {
        [void]$psi.ArgumentList.Add($arg)
    }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    [void]$process.Start()

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    if (-not [string]::IsNullOrWhiteSpace($stdout)) {
        $stdout.TrimEnd("`r", "`n") | Tee-Object -FilePath $RunLog -Append
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        $stderr.TrimEnd("`r", "`n") | Tee-Object -FilePath $RunLog -Append
    }
    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $PythonExe $($Args -join ' ')"
    }
}

function Ensure-Dirs {
    @(
        "outputs\results",
        "outputs\models",
        "outputs\figures",
        "outputs\logs",
        "outputs\checkpoints",
        "runs"
    ) | ForEach-Object {
        New-Item -ItemType Directory -Force -Path (Join-Path $RootDir $_) | Out-Null
    }
}

function Record-Env {
    param([string]$PythonExe)
    Write-RunLog "Repository: $RootDir"
    Write-RunLog "Suite: $Suite"
    Write-RunLog ("Seeds: {0}" -f ($Seeds -join ", "))
    Invoke-Python -PythonExe $PythonExe -Args @("--version")
    Invoke-Python -PythonExe $PythonExe -Args @("-c", "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')")
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        Write-RunLog "RUN: nvidia-smi.exe"
        & nvidia-smi.exe 2>&1 | Tee-Object -FilePath $RunLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi failed"
        }
    }
}

function Run-TreeModels {
    param([string]$PythonExe, [string]$Dataset)
    foreach ($Seed in $Seeds) {
        Invoke-Python $PythonExe @("src/baselines/hgbt_ids.py", "--dataset", $Dataset, "--seed", "$Seed")
        Invoke-Python $PythonExe @("src/baselines/xgboost_ids.py", "--dataset", $Dataset, "--seed", "$Seed")
        Invoke-Python $PythonExe @("src/baselines/lightgbm_ids.py", "--dataset", $Dataset, "--seed", "$Seed")
    }
}

function Run-LstmModels {
    param([string]$PythonExe, [string]$Dataset, [int]$Epochs)
    foreach ($Seed in $Seeds) {
        Invoke-Python $PythonExe @("src/baselines/lstm_ids.py", "--dataset", $Dataset, "--seed", "$Seed", "--epochs", "$Epochs")
    }
}

function Run-BiatMlp {
    param(
        [string]$PythonExe,
        [string]$Dataset,
        [int]$Epochs,
        [double]$Epsilon,
        [double]$Alpha,
        [int]$Steps,
        [double]$AdvWeight,
        [int]$BatchSize = 512
    )
    foreach ($Seed in $Seeds) {
        Invoke-Python $PythonExe @(
            "src/baselines/bilevel_supervised_ids.py",
            "--dataset", $Dataset,
            "--seed", "$Seed",
            "--epochs", "$Epochs",
            "--batch_size", "$BatchSize",
            "--epsilon", "$Epsilon",
            "--alpha", "$Alpha",
            "--steps", "$Steps",
            "--adv_weight", "$AdvWeight"
        )
    }
}

function Run-BiatFt {
    param(
        [string]$PythonExe,
        [string]$Dataset,
        [int]$Epochs,
        [double]$Epsilon,
        [double]$Alpha,
        [int]$Steps,
        [double]$AdvWeight,
        [int]$BatchSize = 512
    )
    foreach ($Seed in $Seeds) {
        Invoke-Python $PythonExe @(
            "src/baselines/bilevel_fttransformer_ids.py",
            "--dataset", $Dataset,
            "--seed", "$Seed",
            "--epochs", "$Epochs",
            "--batch_size", "$BatchSize",
            "--epsilon", "$Epsilon",
            "--alpha", "$Alpha",
            "--steps", "$Steps",
            "--adv_weight", "$AdvWeight"
        )
    }
}

function Run-CoreSuite {
    param([string]$PythonExe)
    Write-RunLog "=== Core suite: UNSW-NB15 ==="
    Run-TreeModels -PythonExe $PythonExe -Dataset "unsw-nb15"
    Run-LstmModels -PythonExe $PythonExe -Dataset "unsw-nb15" -Epochs 20
    Run-BiatMlp -PythonExe $PythonExe -Dataset "unsw-nb15" -Epochs 8 -Epsilon 0.02 -Alpha 0.005 -Steps 2 -AdvWeight 0.6
    Run-BiatFt -PythonExe $PythonExe -Dataset "unsw-nb15" -Epochs 8 -Epsilon 0.02 -Alpha 0.005 -Steps 2 -AdvWeight 0.6
    Invoke-Python $PythonExe @("scripts/evaluate_main_results.py", "--dataset", "unsw-nb15")

    Write-RunLog "=== Core suite: CIC-IDS2017-random ==="
    Run-TreeModels -PythonExe $PythonExe -Dataset "cic-ids2017-random"
    Run-LstmModels -PythonExe $PythonExe -Dataset "cic-ids2017-random" -Epochs 20
    Run-BiatMlp -PythonExe $PythonExe -Dataset "cic-ids2017-random" -Epochs 8 -Epsilon 0.02 -Alpha 0.005 -Steps 2 -AdvWeight 0.6
    if (-not $DisableFtCic) {
        Run-BiatFt -PythonExe $PythonExe -Dataset "cic-ids2017-random" -Epochs 8 -Epsilon 0.02 -Alpha 0.005 -Steps 2 -AdvWeight 0.6
    }
    Invoke-Python $PythonExe @("scripts/evaluate_main_results.py", "--dataset", "cic-ids2017-random")
}

function Run-SupplementarySuite {
    param([string]$PythonExe)
    Write-RunLog "=== Supplementary suite: CICIoT2023-grouped ==="
    Invoke-Python $PythonExe @("src/baselines/hgbt_ids.py", "--dataset", "ciciot2023-grouped", "--seed", "42")
    Invoke-Python $PythonExe @("src/baselines/xgboost_ids.py", "--dataset", "ciciot2023-grouped", "--seed", "42")
    Invoke-Python $PythonExe @("src/baselines/lightgbm_ids.py", "--dataset", "ciciot2023-grouped", "--seed", "42")
    Invoke-Python $PythonExe @(
        "src/baselines/bilevel_supervised_ids.py",
        "--dataset", "ciciot2023-grouped",
        "--seed", "42",
        "--epochs", "5",
        "--batch_size", "512",
        "--epsilon", "0.01",
        "--alpha", "0.002",
        "--steps", "2",
        "--adv_weight", "0.6"
    )
    Invoke-Python $PythonExe @(
        "src/baselines/bilevel_fttransformer_ids.py",
        "--dataset", "ciciot2023-grouped",
        "--seed", "42",
        "--epochs", "5",
        "--batch_size", "512",
        "--epsilon", "0.01",
        "--alpha", "0.002",
        "--steps", "2",
        "--adv_weight", "0.6"
    )
    Invoke-Python $PythonExe @("scripts/evaluate_main_results.py", "--dataset", "ciciot2023-grouped")
}

function Run-LegacyAnalysis {
    param([string]$PythonExe)
    Write-RunLog "=== Legacy support analysis: NSL-KDD ==="
    Invoke-Python $PythonExe @("scripts/evaluate_main_results.py", "--dataset", "nsl-kdd")
    Invoke-Python $PythonExe @("scripts/evaluate_ablation.py", "--dataset", "nsl-kdd")
    Invoke-Python $PythonExe @("scripts/evaluate_adversarial_robustness.py", "--dataset", "nsl-kdd")
}

function Render-Figures {
    param([string]$PythonExe)
    Write-RunLog "=== Rendering figures ==="
    Invoke-Python $PythonExe @("scripts/plot_all_figures.py")
}

$PythonExe = Resolve-CondaPython -EnvName $CondaEnv
Ensure-Dirs
Record-Env -PythonExe $PythonExe

switch ($Suite) {
    "core" {
        Run-CoreSuite -PythonExe $PythonExe
    }
    "all" {
        Run-CoreSuite -PythonExe $PythonExe
        if (-not $DisableCiciot) {
            Run-SupplementarySuite -PythonExe $PythonExe
        }
        if (-not $DisableLegacyAnalysis) {
            Run-LegacyAnalysis -PythonExe $PythonExe
        }
    }
    "supplement" {
        Run-SupplementarySuite -PythonExe $PythonExe
    }
}

Render-Figures -PythonExe $PythonExe
Write-RunLog "All requested experiments completed."
