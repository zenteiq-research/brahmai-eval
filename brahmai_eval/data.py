# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Data directory, dataset preparation and task-YAML rendering.

Every benchmark comes from a public source.  Hugging Face datasets are fetched by `datasets` into the normal HF cache.
Sources that are not on the Hub (e.g. a GitHub release) are downloaded by `brahmai_eval.datasets.<name>.prepare()` into
DATA_DIR, and task YAMLs refer to that directory with the literal placeholder `__DATA_DIR__`, which `render_tasks`
replaces before the harness registers the tasks.
"""
from __future__ import annotations

import glob
import importlib
import os
import pkgutil
import shutil

PLACEHOLDER = "__DATA_DIR__"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_SRC = os.path.join(ROOT, "tasks")


def require_tasks() -> str:
    """The `tasks/` tree ships next to the package in a clone (`pip install -e .`); a plain wheel install lacks it."""
    if not glob.glob(os.path.join(TASKS_SRC, "**", "*.yaml"), recursive=True):
        raise SystemExit(f"task definitions not found at {TASKS_SRC}: brahmai-eval must be installed from a clone of the "
                         "repository with `pip install -e .` (the tasks/ directory is read from the checkout)")
    return TASKS_SRC


def home() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("BRAHMAI_EVAL_HOME") or "~/.cache/brahmai-eval"))


def data_dir() -> str:
    d = os.environ.get("BRAHMAI_EVAL_DATA") or os.path.join(home(), "data")
    os.makedirs(d, exist_ok=True)
    return d


def dataset_modules() -> list[str]:
    from . import datasets as pkg
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__))


def nltk_dir() -> str:
    d = os.environ.get("NLTK_DATA") or os.path.join(home(), "nltk_data")
    os.environ["NLTK_DATA"] = d
    return d


def prepare_nltk() -> None:
    """IFEval counts sentences with NLTK's punkt models; fetch them once into NLTK_DATA."""
    import nltk
    d = nltk_dir()
    os.makedirs(d, exist_ok=True)
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}", paths=[d])
        except LookupError:
            nltk.download(res, download_dir=d, quiet=True)


def prepare(names: list[str] | None = None) -> None:
    """Download every non-Hub dataset (or only `names`) into the data directory.  Idempotent."""
    d = data_dir()
    if not names:
        prepare_nltk()
    for name in dataset_modules():
        if names and name not in names:
            continue
        mod = importlib.import_module(f"brahmai_eval.datasets.{name}")
        print(f"[prepare] {name} -> {d}", flush=True)
        mod.prepare(d)


def render_tasks(dst: str) -> str:
    """Copy tasks/ to `dst` (one directory per run, so concurrent runs never race) with the data placeholder resolved."""
    d = data_dir()
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(require_tasks(), dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for p in glob.glob(os.path.join(dst, "**", "*.yaml"), recursive=True):
        s = open(p, encoding="utf-8").read()
        if PLACEHOLDER in s:
            open(p, "w", encoding="utf-8").write(s.replace(PLACEHOLDER, d))
    return dst


def include_paths(rendered: str) -> list[str]:
    """Every task family directory under the rendered tasks tree; the harness scans each one recursively, and a
    `!function utils.x` in a YAML resolves against that YAML's own directory."""
    return sorted(p for p in glob.glob(os.path.join(rendered, "*")) if os.path.isdir(p) and glob.glob(os.path.join(p, "**", "*.yaml"), recursive=True))
