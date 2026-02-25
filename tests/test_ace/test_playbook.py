import json
import os
import pytest
from continual_learning.ace.playbook import Playbook


class TestPlaybook:
    def test_new_playbook_is_empty(self):
        pb = Playbook()
        assert pb.strategies == []
        assert pb.stats["total_loops"] == 0
        assert pb.stats["documents_processed"] == 0

    def test_add_strategy(self):
        pb = Playbook()
        pb.add_strategy("Look for dates in temporal questions", source="loop_1")
        assert len(pb.strategies) == 1
        assert pb.strategies[0]["rule"] == "Look for dates in temporal questions"
        assert pb.strategies[0]["source"] == "loop_1"
        assert "id" in pb.strategies[0]

    def test_update_strategy(self):
        pb = Playbook()
        pb.add_strategy("Old rule", source="loop_1")
        sid = pb.strategies[0]["id"]
        pb.update_strategy(sid, "Updated rule")
        assert pb.strategies[0]["rule"] == "Updated rule"

    def test_remove_strategy(self):
        pb = Playbook()
        pb.add_strategy("Rule one", source="loop_1")
        pb.add_strategy("Rule two", source="loop_1")
        sid = pb.strategies[0]["id"]
        pb.remove_strategy(sid)
        assert len(pb.strategies) == 1
        assert pb.strategies[0]["rule"] == "Rule two"

    def test_max_strategies_enforced(self):
        pb = Playbook(max_strategies=3)
        for i in range(5):
            pb.add_strategy(f"Rule {i}", source="loop")
        assert len(pb.strategies) == 3
        assert pb.strategies[0]["rule"] == "Rule 2"

    def test_render_returns_string(self):
        pb = Playbook()
        pb.add_strategy("Use exact quotes", source="loop_1")
        rendered = pb.render()
        assert "Use exact quotes" in rendered
        assert isinstance(rendered, str)

    def test_save_and_load(self, tmp_path):
        pb = Playbook()
        pb.add_strategy("Test rule", source="loop_1")
        pb.stats["total_loops"] = 3
        path = tmp_path / "test_playbook.json"
        pb.save(str(path))
        assert path.exists()
        loaded = Playbook.load(str(path))
        assert len(loaded.strategies) == 1
        assert loaded.strategies[0]["rule"] == "Test rule"
        assert loaded.stats["total_loops"] == 3

    def test_save_creates_directory(self, tmp_path):
        pb = Playbook()
        path = tmp_path / "subdir" / "playbook.json"
        pb.save(str(path))
        assert path.exists()

    def test_to_dict_and_from_dict(self):
        pb = Playbook()
        pb.add_strategy("A rule", source="test")
        pb.stats["total_loops"] = 5
        d = pb.to_dict()
        pb2 = Playbook.from_dict(d)
        assert pb2.strategies == pb.strategies
        assert pb2.stats == pb.stats
