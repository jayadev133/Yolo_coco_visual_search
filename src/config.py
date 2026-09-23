from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path=DEFAULT_CONFIG_PATH):
    config_path = _project_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    config_path = _project_path(config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
