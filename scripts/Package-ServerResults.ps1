param(
    [string]$ArchiveName = ""
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

$PackageDir = Join-Path $RootDir "outputs\packages"
New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$HostTag = $env:COMPUTERNAME
if ([string]::IsNullOrWhiteSpace($ArchiveName)) {
    $ArchiveName = "spaper_results_${HostTag}_${Stamp}.zip"
}
$ArchivePath = Join-Path $PackageDir $ArchiveName
$ManifestPath = Join-Path $PackageDir "package_manifest_${Stamp}.txt"

$GitCommit = "unknown"
try {
    $GitCommit = (git rev-parse HEAD).Trim()
} catch {
}

@(
    "timestamp=$Stamp"
    "hostname=$HostTag"
    "root=$RootDir"
    "git_commit=$GitCommit"
    "contents="
    "  outputs\results"
    "  outputs\models"
    "  outputs\figures"
    "  outputs\logs\server_runs"
    "  docs\optimization_decision_log.md"
) | Set-Content -Path $ManifestPath -Encoding UTF8

$Items = @(
    "outputs\results",
    "outputs\models",
    "outputs\figures",
    "outputs\logs\server_runs",
    "docs\optimization_decision_log.md",
    "docs\server_conda_setup.md",
    $ManifestPath
) | ForEach-Object { Join-Path $RootDir $_ }

if (Test-Path $ArchivePath) {
    Remove-Item $ArchivePath -Force
}

Compress-Archive -Path $Items -DestinationPath $ArchivePath -CompressionLevel Optimal
Write-Host "Packaged results: $ArchivePath"
