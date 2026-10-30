$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot 'runtime\environments\.venv-official\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Organized official environment not found.' }
& $pythonExe -X utf8 (Join-Path $PSScriptRoot 'code\run.py') @args
exit $LASTEXITCODE
