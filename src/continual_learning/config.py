import copy
from pathlib import Path
import yaml

DEFAULT_CONFIG = {
    "model": {
        "name": "Qwen/Qwen2.5-1.5B",
        "modified_layers_start": 21,
        "modified_layers_end": 28,
        "device": "auto",
    },
    "ttt": {
        "learning_rate": 1e-5,
        "mini_batch_size": 32,
        "gradient_steps": 1,
        "max_tokens_per_document": 4096,
    },
    "tfidf_gate": {
        "threshold": 0.3,
        "calibration_samples": 2000,
    },
    "checkpoint": {
        "dir": "checkpoints",
    },
    "alpha": {
        "initial": 1.0,
        "decay_rate": 0.01,
        "min_value": 0.3,
    },
}

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(config_path: Path = None) -> dict:
    """Load config from YAML file, merging with defaults."""
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    config = copy.deepcopy(DEFAULT_CONFIG)

    if Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return config


def save_config(config: dict, config_path: Path = None) -> None:
    """Save config to YAML file."""
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
