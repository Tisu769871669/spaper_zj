param(
    [Parameter(Mandatory = $true)]
    [string]$ServerUser,
    [Parameter(Mandatory = $true)]
    [string]$ServerHost,
    [string]$RemoteDir = "~/spaper_zj",
    [switch]$SkipData
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not (Get-Command scp.exe -ErrorAction SilentlyContinue)) {
    throw "scp.exe not found. Please enable OpenSSH client on Windows."
}

$Remote = "${ServerUser}@${ServerHost}:${RemoteDir}"

Write-Host "Sync tracked project files to $Remote"
git archive --format=zip --output ".\project_sync.zip" HEAD
scp.exe ".\project_sync.zip" "${Remote}/project_sync.zip"

if (-not $SkipData) {
    if (-not (Test-Path ".\data")) {
        throw "Local data directory not found: $RootDir\data"
    }
    if (Test-Path ".\data_sync.zip") {
        Remove-Item ".\data_sync.zip" -Force
    }
    Compress-Archive -Path ".\data\*" -DestinationPath ".\data_sync.zip" -CompressionLevel Optimal
    scp.exe ".\data_sync.zip" "${Remote}/data_sync.zip"
}

Write-Host "Uploaded archives. Please extract them on the server:"
Write-Host "  tar -xf project_sync.zip"
if (-not $SkipData) {
    Write-Host "  tar -xf data_sync.zip"
}
