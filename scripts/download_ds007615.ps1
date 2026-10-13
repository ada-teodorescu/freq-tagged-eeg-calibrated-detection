param(
    [string]$Destination = "data/external/ds007615",
    [int]$FirstParticipant = 1,
    [int]$LastParticipant = 69
)

$ErrorActionPreference = "Stop"
$datasetRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))
$workspaceRoot = [System.IO.Path]::GetFullPath((Get-Location).Path)
if (-not $datasetRoot.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must remain inside the workspace: $datasetRoot"
}
New-Item -ItemType Directory -Force -Path $datasetRoot | Out-Null
$hasLocalInventory = @(Get-ChildItem -Path $datasetRoot -Filter "*.vhdr" -File -Recurse -ErrorAction SilentlyContinue).Count -gt 0

$baseUrl = "https://s3.amazonaws.com/openneuro.org/ds007615"
$tasks = @(
    "task-ldaep",
    "task-rest_acq-ec",
    "task-rest_acq-eo"
)
$suffixes = @(
    "channels.tsv",
    "eeg.eeg",
    "eeg.json",
    "eeg.vhdr",
    "eeg.vmrk",
    "events.json",
    "events.tsv"
)

for ($number = $FirstParticipant; $number -le $LastParticipant; $number++) {
    $participant = "sub-{0:D2}" -f $number
    $participantFolder = Join-Path $datasetRoot "$participant/eeg"
    New-Item -ItemType Directory -Force -Path $participantFolder | Out-Null
    $arguments = @("--parallel", "--parallel-max", "6", "--fail", "--location",
                   "--continue-at", "-", "--retry", "3", "--retry-delay", "2", "--silent", "--show-error")
    foreach ($task in $tasks) {
        foreach ($suffix in $suffixes) {
            $filename = "${participant}_${task}_${suffix}"
            $destinationPath = Join-Path $participantFolder $filename
            # A DataLad/OpenNeuro clone already contains the authoritative task inventory.
            # Skip combinations absent from that inventory instead of emitting expected 404s.
            if ($hasLocalInventory -and -not (Test-Path -LiteralPath $destinationPath)) {
                continue
            }
            # DataLad represents annexed content as a short relative-path pointer on
            # Windows. curl --continue-at would mistake that pointer for a partial
            # download and preserve it as the beginning of the payload.
            if ((Test-Path -LiteralPath $destinationPath) -and
                (Get-Item -LiteralPath $destinationPath).Length -le 512) {
                $prefix = Get-Content -LiteralPath $destinationPath -Raw -ErrorAction SilentlyContinue
                if ($prefix.StartsWith("../../.git/annex/objects/")) {
                    Remove-Item -LiteralPath $destinationPath -Force
                }
            }
            $arguments += @("--url", "$baseUrl/$participant/eeg/$filename",
                            "--output", $destinationPath)
        }
    }
    Write-Host "Downloading $participant ($number/$LastParticipant)"
    & curl.exe @arguments
    # LDAEP is intentionally absent for 15 participants; remove only zero-byte failed placeholders.
    Get-ChildItem -LiteralPath $participantFolder -File | Where-Object Length -eq 0 | Remove-Item -Force
}

Write-Host "ds007615 download pass complete: $datasetRoot"
