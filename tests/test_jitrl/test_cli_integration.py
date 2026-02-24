import pytest
from continual_learning.cli.main import build_menu_choices, MENU_HANDLERS


class TestJitRLMenuEntries:
    def test_menu_has_jitrl_mvp(self):
        choices = build_menu_choices()
        values = [c.value for c in choices if hasattr(c, "value")]
        assert "jitrl_mvp" in values

    def test_menu_has_jitrl_full(self):
        choices = build_menu_choices()
        values = [c.value for c in choices if hasattr(c, "value")]
        assert "jitrl_full" in values

    def test_menu_has_compare_engines(self):
        choices = build_menu_choices()
        values = [c.value for c in choices if hasattr(c, "value")]
        assert "compare_engines" in values

    def test_handlers_registered(self):
        assert "jitrl_mvp" in MENU_HANDLERS
        assert "jitrl_full" in MENU_HANDLERS
        assert "compare_engines" in MENU_HANDLERS
