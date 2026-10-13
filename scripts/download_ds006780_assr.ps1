param(
    [ValidateSet("TD", "ALL")]
    [string]$Cohort = "TD",
    [string]$DatasetRoot = "data\external\ds006780"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedDataset = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $DatasetRoot))
$ExpectedParent = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data\external"))
if (-not $ResolvedDataset.StartsWith($ExpectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "DatasetRoot must remain within $ExpectedParent"
}
if (-not (Test-Path -LiteralPath (Join-Path $ResolvedDataset "dataset_description.json"))) {
    throw "Not a ds006780 clone: $ResolvedDataset"
}

$Participants = Import-Csv -Delimiter "`t" -LiteralPath (Join-Path $ResolvedDataset "participants.tsv")
$Allowed = if ($Cohort -eq "ALL") {
    @($Participants.participant_id | ForEach-Object { $_ -replace '^sub-', '' })
} else {
    @($Participants | Where-Object group -eq "TD" | ForEach-Object { $_.participant_id -replace '^sub-', '' })
}

$Files = @(Get-ChildItem -LiteralPath $ResolvedDataset -Recurse -File |
    Where-Object Name -like "*task-ASSR*_eeg.bdf" |
    Where-Object {
        $Subject = [regex]::Match($_.FullName, 'sub-([^\\]+)').Groups[1].Value
        $Subject -in $Allowed
    } |
    Sort-Object FullName)
if (-not $Files) { throw "No ASSR BDF pointers matched cohort $Cohort" }

$Completed = 0L
$ExpectedTotal = 0L
$Records = foreach ($File in $Files) {
    $FirstLine = Get-Content -LiteralPath $File.FullName -TotalCount 1
    $Pointer = [regex]::Match($FirstLine, 'SHA256E-s(\d+)--([0-9a-f]{64})\.bdf')
    if ($Pointer.Success) {
        $ExpectedBytes = [int64]$Pointer.Groups[1].Value
        $ExpectedHash = $Pointer.Groups[2].Value
    } else {
        $ExpectedBytes = $File.Length
        $ExpectedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant()
    }
    $ExpectedTotal += $ExpectedBytes
    [pscustomobject]@{ File = $File; ExpectedBytes = $ExpectedBytes; ExpectedHash = $ExpectedHash }
}

Write-Output ("Downloading {0} ASSR recordings for {1} ({2:N2} GiB)" -f $Records.Count, $Cohort, ($ExpectedTotal / 1GB))
foreach ($Record in $Records) {
    $File = $Record.File
    if ($File.Length -eq $Record.ExpectedBytes) {
        $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant()
        if ($Hash -eq $Record.ExpectedHash) {
            $Completed += $Record.ExpectedBytes
            Write-Output ("SKIP {0} verified" -f $File.Name)
            continue
        }
    }
    $Relative = $File.FullName.Substring($ResolvedDataset.Length).TrimStart('\').Replace('\', '/')
    $Url = "https://s3.amazonaws.com/openneuro.org/ds006780/$Relative"
    $Temporary = $File.FullName + ".download"
    Write-Output ("GET  {0} ({1:N1} MiB; total {2:P1})" -f $File.Name, ($Record.ExpectedBytes / 1MB), ($Completed / $ExpectedTotal))
    & curl.exe --location --fail --retry 5 --retry-delay 2 --continue-at - --output $Temporary $Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $Relative" }
    $Downloaded = Get-Item -LiteralPath $Temporary
    if ($Downloaded.Length -ne $Record.ExpectedBytes) {
        throw "Size mismatch for ${Relative}: $($Downloaded.Length), expected $($Record.ExpectedBytes)"
    }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Temporary).Hash.ToLowerInvariant()
    if ($Hash -ne $Record.ExpectedHash) { throw "SHA-256 mismatch for $Relative" }
    Move-Item -LiteralPath $Temporary -Destination $File.FullName -Force
    $Completed += $Record.ExpectedBytes
}

Write-Output ("COMPLETE {0}/{1} verified ASSR recordings ({2:N2} GiB)" -f $Records.Count, $Records.Count, ($Completed / 1GB))
