"""Small, non-secret provenance records for numerical experiments."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_COMMIT = "77da6cdc67775aa45af6c4148068f00a75e54c41"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def manifest(command, inputs=(), seed=None, classification="numerical_validation"):
    packages = {}
    for name in ("numpy", "matplotlib", "scipy", "torch", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "command": command,
        "seed": seed,
        "paper": "https://arxiv.org/abs/2307.08529v2",
        "upstream_commit": UPSTREAM_COMMIT,
        "working_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                          for p in sorted((ROOT / "repro").rglob("*.py"))},
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(), "packages": packages},
        "inputs": [{"name": Path(p).name, "bytes": Path(p).stat().st_size,
                    "sha256": sha256(p)} for p in inputs],
    }
