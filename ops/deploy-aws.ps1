param(
    [string]$HostAlias = "etf-mixer-lightsail",
    [string]$Branch = "step3/custom-portfolio",
    [string]$RepoPath = "~/ETF_mixer",
    [string]$ServiceName = "etf_mixer.service"
)

$cmd = @"
cd $RepoPath &&
git fetch --all --prune &&
git checkout $Branch &&
git pull --ff-only origin $Branch &&
rm -f backend/data/cache/scatter.json &&
rm -f backend/data/cache/portfolios_*.json &&
sudo systemctl restart $ServiceName &&
sleep 2 &&
systemctl is-active $ServiceName &&
curl -s http://127.0.0.1:5000/api/health
"@

ssh $HostAlias $cmd
