$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TargetDir = Join-Path $RepoRoot "data\\CSE_CIC_IDS2018"
$BucketListUrl = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/?list-type=2"
$BucketBaseUrl = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null

Write-Host "Fetching bucket manifest..."
$manifest = Invoke-WebRequest -Uri $BucketListUrl -UseBasicParsing

$entries = @()
$keyMatches = [regex]::Matches($manifest.Content, "<Contents><Key>(.*?)</Key>.*?<Size>(\d+)</Size>")
foreach ($match in $keyMatches) {
    $key = $match.Groups[1].Value
    $size = [int64]$match.Groups[2].Value
    if ($key -like "Processed Traffic Data for ML Algorithms/*TrafficForML_CICFlowMeter.csv") {
        $entries += [PSCustomObject]@{
            Key  = $key
            Size = $size
        }
    }
}

if (-not $entries -or $entries.Count -eq 0) {
    throw "No CIC-IDS2018 ML CSV files found in bucket listing."
}

Write-Host ("Found {0} CSV files." -f $entries.Count)

foreach ($entry in $entries) {
    $key = $entry.Key
    $expectedSize = $entry.Size
    $fileName = Split-Path $key -Leaf
    $targetPath = Join-Path $TargetDir $fileName
    if (Test-Path $targetPath) {
        $existingSize = (Get-Item $targetPath).Length
        if ($existingSize -eq $expectedSize) {
            Write-Host "Skipping existing $fileName"
            continue
        }
        Write-Host "Removing partial $fileName ($existingSize / $expectedSize bytes)"
        Remove-Item $targetPath -Force
    }

    $escapedKey = [System.Uri]::EscapeDataString($key).Replace("%2F", "/")
    $url = $BucketBaseUrl + $escapedKey
    Write-Host "Downloading $fileName ..."
    & curl.exe -L --fail --retry 5 --retry-delay 2 -o $targetPath $url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $fileName with exit code $LASTEXITCODE"
    }
    $downloadedSize = (Get-Item $targetPath).Length
    if ($downloadedSize -ne $expectedSize) {
        throw "Size mismatch for ${fileName}: got $downloadedSize, expected $expectedSize"
    }
}

Write-Host "Download completed:"
Get-ChildItem $TargetDir -Filter *.csv | Select-Object Name, Length
