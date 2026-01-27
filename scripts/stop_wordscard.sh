#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="wordscard"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
  echo "tmux session '${SESSION_NAME}' stopped"
else
  echo "tmux session '${SESSION_NAME}' not running"
fi
