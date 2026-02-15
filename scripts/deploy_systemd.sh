#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/ETF_mixer"
SYSTEMD_DIR="/etc/systemd/system"
SERVICE_NAME="etf_mixer_prices.service"
TIMER_NAME="etf_mixer_prices.timer"
LOG_FILE="/tmp/etf_mixer_prices.log"

echo "[1/5] Copying unit files to ${SYSTEMD_DIR}"
sudo cp "${REPO_ROOT}/backend/systemd/${SERVICE_NAME}" "${SYSTEMD_DIR}/${SERVICE_NAME}"
sudo cp "${REPO_ROOT}/backend/systemd/${TIMER_NAME}" "${SYSTEMD_DIR}/${TIMER_NAME}"

echo "[2/5] Reloading systemd daemon"
sudo systemctl daemon-reload

echo "[3/5] Enabling and starting timer"
sudo systemctl enable --now "${TIMER_NAME}"

echo "[4/5] Running service once immediately"
sudo systemctl start "${SERVICE_NAME}"

echo "[5/5] Status and recent log"
sudo systemctl status "${TIMER_NAME}" --no-pager
sudo systemctl status "${SERVICE_NAME}" --no-pager
echo "--- tail ${LOG_FILE} ---"
tail -n 50 "${LOG_FILE}" || true
