import json
import os
import torch
from pathlib import Path

from continual_learning.model.dual_mlp import DualMLP


class CheckpointManager:
    """Manages saving and loading of trainable MLP weights and TF-IDF stats.

    Only saves the trainable MLP weights and TF-IDF gate statistics,
    not the full model. This keeps checkpoints small (~50-100MB vs 3GB).
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(self, name: str, dual_mlps: list[DualMLP], metadata: dict) -> Path:
        """Save trainable MLP states and TF-IDF gate stats.

        Args:
            name: Checkpoint name (used as subdirectory name).
            dual_mlps: List of DualMLP modules whose trainable weights to save.
            metadata: Arbitrary metadata dict to save alongside the checkpoint.

        Returns:
            Path to the created checkpoint directory.
        """
        cp_dir = self.checkpoint_dir / name
        cp_dir.mkdir(parents=True, exist_ok=True)

        # Save trainable MLP state dicts
        trainable_states = []
        for dual in dual_mlps:
            trainable_states.append(dual.trainable_mlp.state_dict())
        torch.save(trainable_states, cp_dir / "trainable_mlps.pt")

        # Save TF-IDF gate stats
        gate_states = []
        for dual in dual_mlps:
            gate_states.append(dual.gate.state_dict_custom())
        torch.save(gate_states, cp_dir / "tfidf_stats.pt")

        # Save alpha values into metadata
        alphas = [dual.alpha for dual in dual_mlps]
        metadata["alphas"] = alphas

        with open(cp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update "latest" symlink
        latest = self.checkpoint_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(cp_dir.resolve())

        return cp_dir

    def load(self, name: str, dual_mlps: list[DualMLP]) -> dict:
        """Load trainable MLP states and TF-IDF gate stats.

        Args:
            name: Checkpoint name to load.
            dual_mlps: List of DualMLP modules to restore weights into.

        Returns:
            The metadata dict from the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
        """
        cp_dir = self.checkpoint_dir / name
        if not cp_dir.exists():
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {cp_dir}")

        # Restore trainable MLP weights
        trainable_states = torch.load(cp_dir / "trainable_mlps.pt", weights_only=True)
        for dual, state in zip(dual_mlps, trainable_states):
            dual.trainable_mlp.load_state_dict(state)

        # Restore TF-IDF gate stats
        gate_states = torch.load(cp_dir / "tfidf_stats.pt", weights_only=False)
        for dual, state in zip(dual_mlps, gate_states):
            dual.gate.load_state_dict_custom(state)

        # Restore alpha values
        metadata = self.load_metadata(name)
        if "alphas" in metadata:
            for dual, alpha in zip(dual_mlps, metadata["alphas"]):
                dual.alpha = alpha

        return metadata

    def load_metadata(self, name: str) -> dict:
        """Load only the metadata from a checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            The metadata dict.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
        """
        cp_dir = self.checkpoint_dir / name
        if not cp_dir.exists():
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {cp_dir}")
        with open(cp_dir / "metadata.json") as f:
            return json.load(f)

    def list(self) -> list[str]:
        """List all checkpoint names (excluding the 'latest' symlink).

        Returns:
            Sorted list of checkpoint directory names.
        """
        if not self.checkpoint_dir.exists():
            return []
        return sorted([
            d.name for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and not d.is_symlink()
        ])
