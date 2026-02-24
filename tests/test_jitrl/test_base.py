import pytest
from continual_learning.jitrl.base import BaseJitRLEngine


class TestBaseJitRLEngine:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseJitRLEngine(model=None, tokenizer=None)

    def test_interface_has_learn(self):
        assert hasattr(BaseJitRLEngine, "learn")

    def test_interface_has_generate(self):
        assert hasattr(BaseJitRLEngine, "generate")

    def test_interface_has_clear(self):
        assert hasattr(BaseJitRLEngine, "clear")
