"""Model Hub and Pretrained Checkpoint Registry for Oxideon.

Provides hub interfaces to download, cache, and instantiate
community models and weight checkpoints directly from repository URLs or local directories.
"""

import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
import urllib.request

from .serialization import load as _ox_load


_HUB_DIR = os.environ.get("OXIDEON_HOME", os.path.expanduser("~/.cache/oxideon/hub"))
_HASH_REGEX = re.compile(r"-([a-f0-9]{8,64})\.")


def get_dir() -> str:
    """Returns the directory used to cache checkpoints and downloaded repositories."""
    os.makedirs(_HUB_DIR, exist_ok=True)
    return _HUB_DIR


def set_dir(d: str) -> None:
    """Sets the directory used to cache checkpoints and downloaded repositories."""
    global _HUB_DIR
    if not d or not d.strip():
        raise ValueError("Hub directory cannot be empty")
    _HUB_DIR = d
    os.makedirs(_HUB_DIR, exist_ok=True)


def download_url_to_file(url: str, dst: str, hash_prefix: Optional[str] = None, progress: bool = True) -> None:
    """Downloads an object at the specified URL to a local destination file."""
    dst_path = Path(dst).expanduser().resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = str(dst_path) + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Oxideon-Hub/1.0"})
    sha256 = hashlib.sha256()

    with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as f:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            f.write(chunk)
            sha256.update(chunk)

    if hash_prefix is not None:
        digest = sha256.hexdigest()
        if not digest.startswith(hash_prefix):
            os.remove(tmp_path)
            raise RuntimeError(f"Hash prefix mismatch for {url}: expected {hash_prefix}, got {digest}")

    if os.path.exists(str(dst_path)):
        os.remove(str(dst_path))
    os.rename(tmp_path, str(dst_path))


def load_state_dict_from_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
) -> Any:
    """Downloads and caches a state dict or checkpoint from a remote URL."""
    if model_dir is None:
        model_dir = os.path.join(get_dir(), "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    if file_name is None:
        file_name = os.path.basename(url.split("?")[0])
    cached_file = os.path.join(model_dir, file_name)

    hash_prefix = None
    if check_hash:
        match = _HASH_REGEX.search(file_name)
        if match:
            hash_prefix = match.group(1)

    if not os.path.exists(cached_file):
        download_url_to_file(url, cached_file, hash_prefix=hash_prefix, progress=progress)
    elif hash_prefix is not None:
        sha256 = hashlib.sha256()
        with open(cached_file, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        digest = sha256.hexdigest()
        if not digest.startswith(hash_prefix):
            raise RuntimeError(f"Cached file hash mismatch for {cached_file}: expected {hash_prefix}, got {digest}")

    return _ox_load(cached_file, device=map_location)


def list(repo_or_dir: str, force_reload: bool = False, trust_repo: Optional[bool] = None) -> List[str]:
    """Lists entrypoint model names available in the specified directory or repository."""
    p = Path(repo_or_dir).expanduser().resolve()
    if p.is_dir():
        hubconf = p / "hubconf.py"
        if hubconf.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("hubconf", str(hubconf))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return [name for name in dir(mod) if not name.startswith("_") and callable(getattr(mod, name))]
    return []


def help(repo_or_dir: str, model: str, force_reload: bool = False, trust_repo: Optional[bool] = None) -> str:
    """Returns docstring for the specified model in repo_or_dir."""
    p = Path(repo_or_dir).expanduser().resolve()
    hubconf = p / "hubconf.py"
    if hubconf.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("hubconf", str(hubconf))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            entry = getattr(mod, model, None)
            if entry is not None:
                return entry.__doc__ or ""
    return ""


def load(repo_or_dir: str, model: str, *args, **kwargs) -> Any:
    """Loads a model from a local directory or github repository."""
    kwargs.pop("force_reload", None)
    kwargs.pop("trust_repo", None)
    kwargs.pop("verbose", None)
    kwargs.pop("source", None)

    p = Path(repo_or_dir).expanduser().resolve()
    if p.is_dir():
        hubconf = p / "hubconf.py"
        if hubconf.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("hubconf", str(hubconf))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                entry = getattr(mod, model)
                return entry(*args, **kwargs)

    raise ValueError(f"Could not load model '{model}' from '{repo_or_dir}'")


__all__ = [
    "download_url_to_file",
    "load_state_dict_from_url",
    "get_dir",
    "set_dir",
    "list",
    "help",
    "load",
]
