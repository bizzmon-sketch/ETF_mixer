$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .\.venv)) {
    py -3.11 -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements-lock.txt
python -m pip install flask-cors

Write-Host "bootstrap complete"
