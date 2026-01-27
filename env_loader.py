import os
from pathlib import Path


def load_env(env_path=None):
    """Load .env values and override any existing environment variables."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value
