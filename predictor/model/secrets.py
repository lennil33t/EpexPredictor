import os
from pathlib import Path


def load_secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    secret_file = Path(".secret")
    if not secret_file.exists():
        return default

    for line in secret_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name:
            parsed = raw_value.strip().strip('"').strip("'")
            return parsed or default

    return default
