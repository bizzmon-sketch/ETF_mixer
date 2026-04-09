\# ETF\_dashboard RUNBOOK



\## 1. Local bootstrap

&#x20;   powershell -ExecutionPolicy Bypass -File .\\ops\\bootstrap-local.ps1



\## 2. Local smoke test

&#x20;   powershell -ExecutionPolicy Bypass -File .\\ops\\smoke-local.ps1



\## 3. Local manual run

&#x20;   .\\.venv\\Scripts\\Activate.ps1

&#x20;   $env:PYTHONPATH = "$PWD\\backend"

&#x20;   python -m flask --app app run --debug



\## 4. Deploy selected branch to AWS

&#x20;   powershell -ExecutionPolicy Bypass -File .\\ops\\deploy-aws.ps1 -Branch foundation/sprint-0



\## 5. Check server branch and health

&#x20;   ssh etf-mixer-lightsail "cd \~/ETF\_mixer \&\& git branch --show-current \&\& curl -s http://127.0.0.1:5000/api/health"



\## 6. Check service status

&#x20;   ssh etf-mixer-lightsail "sudo systemctl status etf\_mixer.service --no-pager -n 30"



\## 7. Restart service only

&#x20;   ssh etf-mixer-lightsail "sudo systemctl restart etf\_mixer.service \&\& systemctl is-active etf\_mixer.service"



\## 8. View recent logs

&#x20;   ssh etf-mixer-lightsail "journalctl -u etf\_mixer.service -n 100 --no-pager"



\## 9. Roll back server branch

&#x20;   ssh etf-mixer-lightsail "cd \~/ETF\_mixer \&\& git checkout step3/custom-portfolio \&\& git pull --ff-only origin step3/custom-portfolio \&\& sudo systemctl restart etf\_mixer.service \&\& curl -s http://127.0.0.1:5000/api/health"



\## 10. Notes

\- Git is the source of truth.

\- AWS is pull-only.

\- Do not edit code directly on the server.

\- Default deploy script branch may differ from current working branch, so pass `-Branch` explicitly when needed.

