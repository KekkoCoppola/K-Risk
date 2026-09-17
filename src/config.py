from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "config.yaml"


def load_config(path=CONFIG_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative):
    return ROOT / relative


CONFIG = load_config()
