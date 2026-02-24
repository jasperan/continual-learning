import os
import json
import torch
import pytest
from pathlib import Path
from continual_learning.checkpointing.manager import CheckpointManager
from continual_learning.model.dual_mlp import DualMLP


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def dual_mlp():
    return DualMLP(
        hidden_size=64,
        intermediate_size=128,
        alpha_initial=1.0,
        tfidf_threshold=0.3,
    )


@pytest.fixture
def manager(tmp_checkpoint_dir):
    return CheckpointManager(checkpoint_dir=str(tmp_checkpoint_dir))


class TestCheckpointManager:
    def test_save_creates_directory(self, manager, dual_mlp, tmp_checkpoint_dir):
        manager.save("test_save", dual_mlps=[dual_mlp], metadata={"doc": "test"})
        assert (tmp_checkpoint_dir / "test_save").exists()

    def test_save_creates_files(self, manager, dual_mlp, tmp_checkpoint_dir):
        manager.save("test_files", dual_mlps=[dual_mlp], metadata={"doc": "test"})
        cp_dir = tmp_checkpoint_dir / "test_files"
        assert (cp_dir / "trainable_mlps.pt").exists()
        assert (cp_dir / "tfidf_stats.pt").exists()
        assert (cp_dir / "metadata.json").exists()

    def test_save_updates_latest_symlink(self, manager, dual_mlp, tmp_checkpoint_dir):
        manager.save("v1", dual_mlps=[dual_mlp], metadata={})
        latest = tmp_checkpoint_dir / "latest"
        assert latest.exists()
        assert os.path.realpath(latest) == str(tmp_checkpoint_dir / "v1")

    def test_load_restores_weights(self, manager, dual_mlp, tmp_checkpoint_dir):
        for param in dual_mlp.trainable_mlp.parameters():
            param.data.fill_(0.42)
        manager.save("restore_test", dual_mlps=[dual_mlp], metadata={})

        new_dual = DualMLP(hidden_size=64, intermediate_size=128)
        # New DualMLP has small random init, not the saved 0.42 values
        assert not torch.allclose(list(new_dual.trainable_mlp.parameters())[0], torch.tensor(0.42))

        manager.load("restore_test", dual_mlps=[new_dual])
        for param in new_dual.trainable_mlp.parameters():
            assert torch.allclose(param.data, torch.tensor(0.42))

    def test_list_checkpoints(self, manager, dual_mlp):
        manager.save("cp_a", dual_mlps=[dual_mlp], metadata={})
        manager.save("cp_b", dual_mlps=[dual_mlp], metadata={})
        names = manager.list()
        assert "cp_a" in names
        assert "cp_b" in names
        assert "latest" not in names

    def test_load_metadata(self, manager, dual_mlp):
        meta = {"documents": ["doc1.txt"], "timestamp": "2026-02-24"}
        manager.save("meta_test", dual_mlps=[dual_mlp], metadata=meta)
        loaded_meta = manager.load_metadata("meta_test")
        assert loaded_meta["documents"] == ["doc1.txt"]
        assert loaded_meta["timestamp"] == "2026-02-24"

    def test_load_nonexistent_raises(self, manager, dual_mlp):
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent", dual_mlps=[dual_mlp])
