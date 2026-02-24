import pytest
from continual_learning.cli.main import build_menu_choices, format_status_panel


class TestBuildMenuChoices:
    def test_returns_questionary_choices(self):
        choices = build_menu_choices()
        assert len(choices) > 0

    def test_includes_core_options(self):
        choices = build_menu_choices()
        values = [c.value for c in choices if hasattr(c, "value")]
        assert "chat" in values
        assert "learn_doc" in values
        assert "benchmark" in values
        assert "exit" in values


class TestFormatStatusPanel:
    def test_returns_string(self):
        status = format_status_panel(
            model_name="Qwen2.5-1.5B",
            status="Ready",
            learned_count=3,
            vram_used="8.2GB",
            vram_total="24GB",
        )
        assert "Qwen2.5-1.5B" in status
        assert "Ready" in status
        assert "3" in status
