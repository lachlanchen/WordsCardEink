#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="wordscard"
VENV_NAME="wordscard"
APP_ARGS=${APP_ARGS:-""}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}"
tmux send-keys -t "${SESSION_NAME}" "cd ${ROOT_DIR} && source ${VENV_NAME}/bin/activate && GPIOZERO_PIN_FACTORY=lgpio python app.py ${APP_ARGS}" C-m

echo "tmux session '${SESSION_NAME}' started"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
