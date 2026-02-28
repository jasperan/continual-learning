from continual_learning.doc2lora.checkpoint_utils import (
    get_checkpoint_path,
    is_checkpoint_cached,
)


class TestCheckpointUtils:
    def test_get_checkpoint_path_returns_path(self):
        path = get_checkpoint_path(
            repo_id="SakanaAI/doc-to-lora",
            filename="gemma_demo/checkpoint-80000/pytorch_model.bin",
            cache_dir="/tmp/test_cache",
        )
        assert isinstance(path, str)
        assert "pytorch_model.bin" in path

    def test_is_checkpoint_cached_false_for_missing(self, tmp_path):
        assert not is_checkpoint_cached(
            repo_id="SakanaAI/doc-to-lora",
            filename="gemma_demo/checkpoint-80000/pytorch_model.bin",
            cache_dir=str(tmp_path),
        )

    def test_is_checkpoint_cached_true_when_exists(self, tmp_path):
        ckpt_dir = tmp_path / "SakanaAI" / "doc-to-lora"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "pytorch_model.bin").write_text("fake")
        assert is_checkpoint_cached(
            repo_id="SakanaAI/doc-to-lora",
            filename="pytorch_model.bin",
            cache_dir=str(tmp_path),
        )
