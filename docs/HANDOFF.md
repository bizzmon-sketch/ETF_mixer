\# ETF\_dashboard HANDOFF



\## Current working branch

\- local working branch: `foundation/sprint-0`

\- deploy target can be switched by `ops/deploy-aws.ps1 -Branch <branch>`



\## Local project path

\- `C:\\Users\\bizzm\\projects\\ETF\_dashboard`



\## Python / venv

\- Python: 3.11

\- venv path: `.venv`



\## Local bootstrap

&#x20;   powershell -ExecutionPolicy Bypass -File .\\ops\\bootstrap-local.ps1



\## Local smoke test

&#x20;   powershell -ExecutionPolicy Bypass -File .\\ops\\smoke-local.ps1



\## Local run (manual)

&#x20;   .\\.venv\\Scripts\\Activate.ps1

&#x20;   $env:PYTHONPATH = "$PWD\\backend"

&#x20;   python -m flask --app app run --debug



\## SSH alias

\- alias: `etf-mixer-lightsail`



\## AWS server

\- host alias: `etf-mixer-lightsail`

\- repo path: `\~/ETF\_mixer`

\- service: `etf\_mixer.service`



\## Deploy

&#x20;   powershell -ExecutionPolicy Bypass -File .\\ops\\deploy-aws.ps1 -Branch foundation/sprint-0



\## Server health check

&#x20;   ssh etf-mixer-lightsail "cd \~/ETF\_mixer \&\& git branch --show-current \&\& curl -s http://127.0.0.1:5000/api/health"



\## Current status

\- local smoke test: passed

\- ssh alias connection: passed

\- aws deploy script: passed

\- server branch switched to `foundation/sprint-0`: confirmed



\## Notes

\- `deploy-aws.ps1` default branch is currently `step3/custom-portfolio`

\- when deploying another branch, pass `-Branch <branch>`

\- current repo contains `requirements-lock.txt` but not `requirements.txt`

\- `flask-cors` is required for local app import

