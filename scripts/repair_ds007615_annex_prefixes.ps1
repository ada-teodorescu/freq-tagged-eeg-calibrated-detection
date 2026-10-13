param(
    [string]$Dataset = "data/external/ds007615"
)

$ErrorActionPreference = "Stop"
$datasetRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Dataset))
$workspaceRoot = [System.IO.Path]::GetFullPath((Get-Location).Path)
if (-not $datasetRoot.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Dataset must remain inside the workspace: $datasetRoot"
}
if (-not (Test-Path -LiteralPath $datasetRoot)) { throw "Dataset not found: $datasetRoot" }
$datasetPrefix = $datasetRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

$temporaryRoot = Join-Path $workspaceRoot (".repair-ds007615-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
$baseUrl = "https://s3.amazonaws.com/openneuro.org/ds007615"
$pattern = '^\.\./\.\./\.git/annex/objects/[^/]+/[^/]+/(?<key>SHA256E-s(?<size>\d+)--(?<hash>[0-9a-f]+)\.[A-Za-z0-9]+)/\k<key>'
$records = @()

try {
    foreach ($file in Get-ChildItem -Path $datasetRoot -File -Recurse) {
        $stream = [System.IO.File]::OpenRead($file.FullName)
        try {
            $buffer = New-Object byte[] 512
            $read = $stream.Read($buffer, 0, $buffer.Length)
        } finally { $stream.Dispose() }
        $head = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $read)
        $match = [regex]::Match($head, $pattern)
        if (-not $match.Success) { continue }
        $expectedSize = [int64]$match.Groups["size"].Value
        if ($file.Length -ne $expectedSize) {
            throw "Refusing prefix repair because size differs from annex key: $($file.FullName)"
        }
        $prefixLength = [System.Text.Encoding]::ASCII.GetByteCount($match.Value)
        if (-not $file.FullName.StartsWith($datasetPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "File escaped the dataset root: $($file.FullName)"
        }
        $relative = $file.FullName.Substring($datasetPrefix.Length).Replace("\", "/")
        $tempPath = Join-Path $temporaryRoot (([guid]::NewGuid().ToString("N")) + ".prefix")
        $records += [pscustomobject]@{
            File = $file.FullName
            Relative = $relative
            Url = "$baseUrl/$relative"
            Temp = $tempPath
            PrefixLength = $prefixLength
            ExpectedHash = $match.Groups["hash"].Value.ToLowerInvariant()
            OriginalPrefix = $buffer[0..($prefixLength - 1)]
        }
    }

    Write-Host "Found $($records.Count) pointer-prefixed files. Downloading only their first 256 bytes."
    foreach ($participantGroup in $records | Group-Object { ($_.Relative -split '/')[0] }) {
        $arguments = @("--parallel", "--parallel-max", "12", "--fail", "--location",
                       "--retry", "3", "--retry-delay", "2", "--silent", "--show-error", "--range", "0-255")
        foreach ($record in $participantGroup.Group) {
            $arguments += @("--url", $record.Url, "--output", $record.Temp)
        }
        & curl.exe @arguments
        if ($LASTEXITCODE -ne 0) { throw "Prefix download failed for $($participantGroup.Name)" }
        Write-Host "Fetched repair prefixes for $($participantGroup.Name)"
    }

    $repaired = 0
    foreach ($record in $records) {
        $replacement = [System.IO.File]::ReadAllBytes($record.Temp)
        if ($replacement.Length -lt $record.PrefixLength) {
            throw "Short prefix response for $($record.Relative)"
        }
        $stream = [System.IO.File]::Open($record.File, [System.IO.FileMode]::Open,
                                        [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        try {
            $stream.Position = 0
            $stream.Write($replacement, 0, $record.PrefixLength)
            $stream.Flush($true)
        } finally { $stream.Dispose() }
        $actualHash = (Get-FileHash -LiteralPath $record.File -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $record.ExpectedHash) {
            $stream = [System.IO.File]::Open($record.File, [System.IO.FileMode]::Open,
                                            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try { $stream.Write($record.OriginalPrefix, 0, $record.OriginalPrefix.Length) }
            finally { $stream.Dispose() }
            throw "Hash verification failed; original corrupt prefix restored: $($record.Relative)"
        }
        $repaired++
        if (($repaired % 25) -eq 0 -or $repaired -eq $records.Count) {
            Write-Host "Hash-verified $repaired/$($records.Count) repaired files"
        }
    }
    Write-Host "Repair complete: $repaired files now match their DataLad SHA-256 keys."
} finally {
    if ((Test-Path -LiteralPath $temporaryRoot) -and
        $temporaryRoot.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
