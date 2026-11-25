from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(*parts):
    return int(digest(parts)[:16], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def provenance(config):
    root = Path(__file__).parent.parent
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    versions = {}
    for name in ("chess", "numpy", "torch", "transformers", "peft", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": versions, "config": config,
            "config_sha256": digest(config),
            "source_sha256": digest({str(p.relative_to(root)): p.read_text() for p in files})}
