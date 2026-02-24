import pytest
from pathlib import Path
from continual_learning.config import load_config, save_config, DEFAULT_CONFIG


class TestConfig:
    def test_load_defaults(self, tmp_path):
        config = load_config(config_path=tmp_path / "nonexistent.yaml")
        assert config["model"]["name"] == DEFAULT_CONFIG["model"]["name"]
        assert config["ttt"]["learning_rate"] == DEFAULT_CONFIG["ttt"]["learning_rate"]

    def test_save_and_load_roundtrip(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        config = load_config(config_path=config_path)
        config["ttt"]["learning_rate"] = 0.001
        save_config(config, config_path=config_path)
        loaded = load_config(config_path=config_path)
        assert loaded["ttt"]["learning_rate"] == 0.001

    def test_user_config_merges_with_defaults(self, tmp_path):
        config_path = tmp_path / "partial.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump({"ttt": {"learning_rate": 0.01}}, f)
        config = load_config(config_path=config_path)
        assert config["ttt"]["learning_rate"] == 0.01
        assert config["model"]["name"] == DEFAULT_CONFIG["model"]["name"]
