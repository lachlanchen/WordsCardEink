#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="wordscard"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:-$(whoami)}"
APP_ARGS=${APP_ARGS:-"--use_csv"}

SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "${SERVICE_PATH}" > /dev/null <<SERVICE
[Unit]
Description=WordsCardEink service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${ROOT_DIR}
Environment=GPIOZERO_PIN_FACTORY=lgpio
Environment=APP_ARGS=${APP_ARGS}
ExecStart=/bin/bash -lc 'cd ${ROOT_DIR} && source wordscard/bin/activate && python app.py ${APP_ARGS}'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

cat <<MSG
Installed ${SERVICE_NAME}.service
Start now with: sudo systemctl start ${SERVICE_NAME}
Check status: sudo systemctl status ${SERVICE_NAME} -n 50
Logs: journalctl -u ${SERVICE_NAME} -n 100 --no-pager
MSG
