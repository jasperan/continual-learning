import os
from pathlib import Path


def get_checkpoint_path(
    repo_id: str,
    filename: str,
    cache_dir: str = None,
) -> str:
    """Get the local path where a checkpoint would be cached.

    Args:
        repo_id: HuggingFace repo ID (e.g., "SakanaAI/doc-to-lora").
        filename: File path within the repo.
        cache_dir: Local cache directory. Defaults to ~/.cache/continual_learning.

    Returns:
        Local file path string.
    """
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "continual_learning")
    return os.path.join(cache_dir, repo_id, os.path.basename(filename))


def is_checkpoint_cached(
    repo_id: str,
    filename: str,
    cache_dir: str = None,
) -> bool:
    """Check if a checkpoint is already cached locally."""
    path = get_checkpoint_path(repo_id, filename, cache_dir)
    return os.path.exists(path)


def download_checkpoint(
    repo_id: str,
    filename: str,
    cache_dir: str = None,
) -> str:
    """Download a checkpoint from HuggingFace Hub if not cached.

    Returns:
        Local file path to the checkpoint.
    """
    path = get_checkpoint_path(repo_id, filename, cache_dir)

    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(path),
    )
    return downloaded
