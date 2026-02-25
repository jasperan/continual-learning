import json
from unittest.mock import MagicMock
from continual_learning.ace.curator import Curator
from continual_learning.ace.playbook import Playbook


class TestCurator:
    def setup_method(self):
        self.client = MagicMock()
        self.curator = Curator(client=self.client)

    def test_curate_adds_new_strategies(self):
        pb = Playbook()
        feedback = {
            "assessment": "incorrect",
            "suggested_strategies": ["Quote exact text for definitions"],
        }
        patch = {"add": ["Quote exact text for definitions"], "remove": [], "update": []}
        self.client.generate.return_value = json.dumps(patch)
        self.curator.curate(playbook=pb, feedback=feedback, loop_num=1)
        assert len(pb.strategies) == 1
        assert "Quote exact text" in pb.strategies[0]["rule"]

    def test_curate_removes_strategies(self):
        pb = Playbook()
        sid = pb.add_strategy("Bad rule", source="loop_0")
        feedback = {"assessment": "incorrect", "suggested_strategies": []}
        patch = {"add": [], "remove": [sid], "update": []}
        self.client.generate.return_value = json.dumps(patch)
        self.curator.curate(playbook=pb, feedback=feedback, loop_num=1)
        assert len(pb.strategies) == 0

    def test_curate_updates_strategies(self):
        pb = Playbook()
        sid = pb.add_strategy("Old rule", source="loop_0")
        feedback = {"assessment": "partially_correct", "suggested_strategies": []}
        patch = {"add": [], "remove": [], "update": [{"id": sid, "rule": "New rule"}]}
        self.client.generate.return_value = json.dumps(patch)
        self.curator.curate(playbook=pb, feedback=feedback, loop_num=1)
        assert pb.strategies[0]["rule"] == "New rule"

    def test_curate_handles_malformed_json(self):
        pb = Playbook()
        feedback = {
            "assessment": "incorrect",
            "suggested_strategies": ["Fallback rule"],
        }
        self.client.generate.return_value = "not json"
        self.curator.curate(playbook=pb, feedback=feedback, loop_num=1)
        assert len(pb.strategies) == 1
        assert "Fallback rule" in pb.strategies[0]["rule"]

    def test_curate_increments_loop_stats(self):
        pb = Playbook()
        feedback = {"assessment": "correct", "suggested_strategies": []}
        self.client.generate.return_value = json.dumps({"add": [], "remove": [], "update": []})
        self.curator.curate(playbook=pb, feedback=feedback, loop_num=1)
        assert pb.stats["total_loops"] == 1
