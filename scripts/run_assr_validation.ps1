param(
    [string]$Stage = "doctor",
    [ValidateRange(1, 4)]
    [int]$Workers = 4,
    [switch]$Force
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Command = Join-Path $ProjectRoot ".venv\Scripts\cucu-assr-validation.exe"
if (-not (Test-Path -LiteralPath $Command)) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing project environment. Run 'uv sync --extra raw --extra dev' first."
    }
    $Command = $Python
    $ModuleArguments = @("-m", "cucu_assr_validation.cli")
} else {
    $ModuleArguments = @()
}

Push-Location $ProjectRoot
try {
    switch ($Stage) {
        "doctor" { & $Command @ModuleArguments doctor }
        "list" { & $Command @ModuleArguments list }
        "status" { & $Command @ModuleArguments status }
        "export" {
            if ($Force) { & $Command @ModuleArguments export --force } else { & $Command @ModuleArguments export }
        }
        "run-all" {
            if ($Force) { throw "Use the CLI --force-stage option for selective invalidation." }
            & $Command @ModuleArguments run-all --workers $Workers
        }
        default {
            if ($Force) { & $Command @ModuleArguments run $Stage --workers $Workers --force }
            else { & $Command @ModuleArguments run $Stage --workers $Workers }
        }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
