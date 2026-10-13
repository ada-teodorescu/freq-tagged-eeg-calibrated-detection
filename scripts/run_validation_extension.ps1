param(
    [string]$Stage = "doctor",
    [ValidateRange(1, 4)]
    [int]$Workers = 4,
    [switch]$Force
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Command = Join-Path $ProjectRoot ".venv\Scripts\cucu-validation-extension.exe"
if (-not (Test-Path -LiteralPath $Command)) {
    throw "Missing $Command. Install the project environment before running the extension."
}

Push-Location $ProjectRoot
try {
    switch ($Stage) {
        "doctor" { & $Command doctor }
        "list" { & $Command list }
        "status" { & $Command status }
        "export" {
            if ($Force) { & $Command export --force } else { & $Command export }
        }
        "run-all" {
            if ($Force) {
                throw "For selective reruns, call the CLI with --force-stage; blanket forcing is intentionally unsupported."
            }
            & $Command run-all --workers $Workers
        }
        default {
            if ($Force) { & $Command run $Stage --workers $Workers --force }
            else { & $Command run $Stage --workers $Workers }
        }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
