import json
from unittest.mock import MagicMock
from continual_learning.ace.engine import ACEEngine


class TestACEEngine:
    def setup_method(self):
        self.client = MagicMock()

        def mock_generate(prompt, system=None, temperature=0.7):
            if system is None:
                return "Oracle was founded in 1977"
            elif "curation" in (system or "").lower():
                # Check curation before reflection because the curator's
                # system prompt also contains the word "reflection".
                return json.dumps({
                    "add": ["Always look for founding dates"],
                    "remove": [],
                    "update": [],
                })
            elif "reflection" in (system or "").lower():
                return json.dumps({
                    "assessment": "correct",
                    "strengths": ["Found the date"],
                    "weaknesses": [],
                    "suggested_strategies": ["Always look for founding dates"],
                })
            return "fallback"

        self.client.generate.side_effect = mock_generate

    def _patch_engine(self, engine):
        """Replace all internal clients with the mock."""
        engine._client = self.client
        engine._generator.client = self.client
        engine._reflector.client = self.client
        engine._curator.client = self.client

    def test_init(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=3)
        assert engine.num_loops == 3

    def test_learn_stores_document(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self._patch_engine(engine)
        metrics = engine.learn("Oracle was founded in 1977 by Larry Ellison.")
        assert metrics["tokens_processed"] > 0
        assert metrics["method"] == "ace"
        assert engine.num_documents == 1

    def test_learn_with_qa_runs_loops(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=2)
        self._patch_engine(engine)
        metrics = engine.learn(
            "Oracle was founded in 1977.",
            qa_pairs=[{"question": "When was Oracle founded?", "answer": "1977"}],
        )
        assert metrics["loops_completed"] == 2
        assert len(engine._playbook.strategies) > 0

    def test_learn_callback_invoked(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self._patch_engine(engine)
        calls = []
        engine.learn(
            "Oracle was founded in 1977.",
            qa_pairs=[{"question": "When?", "answer": "1977"}],
            callback=lambda loop, assessment: calls.append((loop, assessment)),
        )
        assert len(calls) == 1
        assert calls[0][0] == 1

    def test_generate_uses_playbook(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self._patch_engine(engine)
        engine.learn("Oracle was founded in 1977.")
        result = engine.generate("When was Oracle founded?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clear_resets_state(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self._patch_engine(engine)
        engine.learn("Some text")
        engine.clear()
        assert engine.num_documents == 0
        assert engine._playbook.strategies == []

    def test_save_and_load_playbook(self, tmp_path):
        engine = ACEEngine(
            ollama_model="qwen2.5:7b", num_loops=1, playbook_dir=str(tmp_path)
        )
        self._patch_engine(engine)
        engine.learn(
            "Oracle was founded in 1977.",
            qa_pairs=[{"question": "When?", "answer": "1977"}],
        )
        engine.save_playbook("test_pb")
        assert (tmp_path / "test_pb.json").exists()

        engine2 = ACEEngine(
            ollama_model="qwen2.5:7b", num_loops=1, playbook_dir=str(tmp_path)
        )
        engine2.load_playbook("test_pb")
        assert len(engine2._playbook.strategies) > 0

    def test_num_strategies_in_metrics(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self._patch_engine(engine)
        metrics = engine.learn(
            "Oracle was founded in 1977.",
            qa_pairs=[{"question": "When?", "answer": "1977"}],
        )
        assert "num_strategies" in metrics
        assert metrics["num_strategies"] > 0

    def test_multiple_documents(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self._patch_engine(engine)
        engine.learn("First document.")
        engine.learn("Second document.")
        assert engine.num_documents == 2

    def test_learn_no_qa_returns_zero_loops(self):
        engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=3)
        self._patch_engine(engine)
        metrics = engine.learn("Just a document, no QA pairs.")
        assert metrics["loops_completed"] == 0
        assert metrics["num_strategies"] == 0
