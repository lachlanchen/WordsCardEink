#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="wordscard"
VENV_NAME="wordscard"
APP_ARGS=${APP_ARGS:-"--use_csv"}

# Basic OS deps (Pi)
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev \
  libjpeg-dev zlib1g-dev libfreetype6-dev \
  python3-spidev python3-rpi.gpio \
  python3-lgpio \
  liblgpio-dev liblgpio1 \
  swig \
  tmux

# OpenCC package name differs across Debian/RPi releases
if ! sudo apt-get install -y libopencc1 libopencc-dev; then
  sudo apt-get install -y libopencc2 libopencc-dev || true
fi

# Create venv if missing
if [ ! -d "${VENV_NAME}" ]; then
  python3 -m venv "${VENV_NAME}"
fi

# Activate venv
# shellcheck disable=SC1090
source "${VENV_NAME}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Python deps (runtime)
python -m pip install \
  openai tornado Pillow numpy nltk opencc pykakasi arabic-reshaper python-bidi pytz json5 pandas \
  spidev RPi.GPIO gpiozero lgpio

# Optional: download NLTK words corpus (used by some word fetchers)
python -m nltk.downloader words || true

# Install Waveshare driver package (uses setup.py in repo)
python setup.py install

# Start tmux session running the app
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists"
else
  tmux new-session -d -s "${SESSION_NAME}" "bash -lc 'cd $(pwd) && source ${VENV_NAME}/bin/activate && GPIOZERO_PIN_FACTORY=lgpio python app.py ${APP_ARGS}'"
  echo "tmux session '${SESSION_NAME}' started"
fi

echo "Attach with: tmux attach -t ${SESSION_NAME}"
