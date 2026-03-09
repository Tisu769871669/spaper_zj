param(
    [Parameter(Mandatory = $true)]
    [string]$ServerUser,
    [Parameter(Mandatory = $true)]
    [string]$ServerHost,
    [string]$RemoteDir = "~/spaper_zj",
    [string]$RemoteArchive = "outputs/packages/spaper_results_*.zip",
    [string]$LocalDir = ""
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not (Get-Command scp.exe -ErrorAction SilentlyContinue)) {
    throw "scp.exe not found. Please enable OpenSSH client on Windows."
}

if ([string]::IsNullOrWhiteSpace($LocalDir)) {
    $LocalDir = Join-Path $RootDir "outputs\server_fetch"
}
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

$Remote = "${ServerUser}@${ServerHost}:${RemoteDir}/${RemoteArchive}"
scp.exe $Remote $LocalDir
Write-Host "Fetched archives to $LocalDir"
