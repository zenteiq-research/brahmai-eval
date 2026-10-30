# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run provenance: software versions, hardware, git state, and the resolved run configuration."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

from .data import ROOT

_PKGS = ("vllm", "lm_eval", "torch", "transformers", "datasets", "sacrebleu", "huggingface_hub")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def package_versions() -> dict[str, str | None]:
    out = {}
    for p in _PKGS:
        try:
            out[p] = __import__(p).__version__
        except Exception:
            out[p] = None
    return out


def gpus() -> list[dict]:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout
        return [dict(zip(("index", "name", "memory_total", "driver"), [x.strip() for x in l.split(",")])) for l in q.splitlines() if l.strip()]
    except Exception:
        return []


def git_state() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip())
        return {"commit": sha or None, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def version() -> str:
    """The installed brahmai-eval version (the package's own `__version__` when it is not installed)."""
    try:
        from importlib.metadata import version as _v
        return _v("brahmai-eval")
    except Exception:
        from . import __version__
        return __version__


def environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": package_versions(), "gpus": gpus(),
            "brahmai_eval": {"version": version(), **git_state()}, "captured": now()}


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)
