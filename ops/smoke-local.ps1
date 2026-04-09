$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")
. .\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = "$PWD\backend"

Write-Host "== import test =="
python -c "from app import app; print(app.url_map)"

Write-Host "== health check =="
$proc = Start-Process python -ArgumentList "-m","flask","--app","app","run" -WorkingDirectory $PWD -PassThru
Start-Sleep -Seconds 3
try {
    Invoke-WebRequest "http://127.0.0.1:5000/api/health" -UseBasicParsing | Select-Object StatusCode, Content
    Invoke-WebRequest "http://127.0.0.1:5000/api/scatter" -UseBasicParsing | Select-Object StatusCode
}
finally {
    Stop-Process -Id $proc.Id -Force
}
