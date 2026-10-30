# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Berkeley Function Calling Leaderboard (BFCL v3) data and official scorer for the `bfcl_*` tool-use tasks.

Source: https://github.com/ShishirPatil/gorilla (berkeley-function-call-leaderboard) at the pinned commit below,
Apache-2.0.  `prepare(data_dir)` downloads the GitHub archive of that commit and materialises

    <data_dir>/bfcl/data/BFCL_v3_<category>.json            question files (one JSON object per line)
    <data_dir>/bfcl/data/possible_answer/BFCL_v3_<cat>.json  ground truth for the AST and multi-turn categories
    <data_dir>/bfcl/data/multi_turn_func_doc/<api>.json     function docs of the eight stateful multi-turn APIs
    <data_dir>/bfcl/bfcl_scorer/                            the official AST checker, multi-turn checker and executor

The scorer is the upstream code with only its import lines rewritten (the upstream package imports every API
provider's SDK at module scope, which this toolkit does not depend on).  Every rewrite is listed in PATCHES and
recorded in SOURCE.json.  The Java/JavaScript categories need tree-sitter parsers and are not run; their parser
imports are replaced by stubs that raise if called.

Import the scorer with `load_scorer(data_dir)`, which returns the `bfcl_scorer` package.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

REPO_URL = "https://github.com/ShishirPatil/gorilla"
SHA = "1dedb829739049f4c783904a7da96d59fdcd71bd"
ARCHIVE_URL = f"{REPO_URL}/archive/{SHA}.tar.gz"
SOURCE = f"{REPO_URL}@{SHA}"
LICENSE = "Apache-2.0"
SUBDIR = "berkeley-function-call-leaderboard/"
PACKAGE = "bfcl_scorer"

# (category, kind, rows).  kind: ast = scored by the AST checker against possible_answer; irrelevance = correct
# when no function is called; relevance = correct when some function is called.  Python categories only.
SINGLE_TURN = [
    ("simple", "ast", 400),
    ("multiple", "ast", 200),
    ("parallel", "ast", 200),
    ("parallel_multiple", "ast", 200),
    ("irrelevance", "irrelevance", 240),
    ("live_simple", "ast", 258),
    ("live_multiple", "ast", 1053),
    ("live_parallel", "ast", 16),
    ("live_parallel_multiple", "ast", 24),
    ("live_irrelevance", "irrelevance", 882),
    ("live_relevance", "relevance", 18),
]
MULTI_TURN = [("base", 200), ("miss_func", 200), ("miss_param", 200), ("long_context", 200)]

FUNC_DOC_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json", "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json", "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json",
}
FUNC_DOC_ROWS = {"gorilla_file_system.json": 18, "math_api.json": 17, "message_api.json": 10, "posting_api.json": 14,
                 "ticket_api.json": 9, "trading_bot.json": 22, "travel_booking.json": 17, "vehicle_control.json": 22}

# repository-relative path -> expected row count
DATA_ROWS: dict[str, int] = {}
for _c, _k, _n in SINGLE_TURN:
    DATA_ROWS[f"data/BFCL_v3_{_c}.json"] = _n
    if _k == "ast":
        DATA_ROWS[f"data/possible_answer/BFCL_v3_{_c}.json"] = _n
for _c, _n in MULTI_TURN:
    DATA_ROWS[f"data/BFCL_v3_multi_turn_{_c}.json"] = _n
    DATA_ROWS[f"data/possible_answer/BFCL_v3_multi_turn_{_c}.json"] = _n
for _f, _n in FUNC_DOC_ROWS.items():
    DATA_ROWS[f"data/multi_turn_func_doc/{_f}"] = _n

# upstream file -> path inside the vendored package
SCORER_FILES = {
    "bfcl/constants/type_mappings.py": "type_mappings.py",
    "bfcl/constants/default_prompts.py": "default_prompts.py",
    "bfcl/model_handler/model_style.py": "model_style.py",
    "bfcl/model_handler/utils.py": "model_utils.py",
    "bfcl/eval_checker/ast_eval/ast_checker.py": "ast_checker.py",
    "bfcl/eval_checker/ast_eval/type_convertor/java_type_converter.py": "type_convertor/java_type_converter.py",
    "bfcl/eval_checker/ast_eval/type_convertor/js_type_converter.py": "type_convertor/js_type_converter.py",
    "bfcl/eval_checker/multi_turn_eval/multi_turn_checker.py": "multi_turn_checker.py",
    "bfcl/eval_checker/multi_turn_eval/multi_turn_utils.py": "multi_turn_utils.py",
}
FUNC_SOURCE_DIR = "bfcl/eval_checker/multi_turn_eval/func_source_code/"
UTILS_FILE = "bfcl/utils.py"           # two helpers are extracted from it (see FORMAT_HELPERS)
FORMAT_HELPERS = ["is_function_calling_format_output", "is_empty_output"]

# (regex, replacement template, reason); `{dots}` becomes the relative-import prefix for the file's depth.
PATCHES = [
    (r"from bfcl\.constants\.eval_config import UNDERSCORE_TO_DOT",
     "UNDERSCORE_TO_DOT: list = []  # no provider name regex applies: function names are compared as written",
     "ast_checker: the list of API models whose provider forbids '.' in names is irrelevant for a local model"),
    (r"from bfcl\.constants\.(\w+) import", r"from {dots}\1 import", "package-relative import"),
    (r"from bfcl\.model_handler\.model_style import", r"from {dots}model_style import", "package-relative import"),
    (r"from bfcl\.eval_checker\.ast_eval\.type_convertor\.(\w+) import", r"from {dots}type_convertor.\1 import",
     "package-relative import"),
    (r"from bfcl\.eval_checker\.multi_turn_eval\.multi_turn_utils import", r"from {dots}multi_turn_utils import",
     "package-relative import"),
    (r"from bfcl\.eval_checker\.multi_turn_eval\.func_source_code\.(\w+) import", r"from .\1 import",
     "package-relative import (sibling module inside func_source_code)"),
    (r'"bfcl\.eval_checker\.multi_turn_eval\.func_source_code\.', f'"{PACKAGE}.func_source_code.',
     "executor imports the API classes by absolute module name"),
    (r"from bfcl\.model_handler\.parser\.java_parser import parse_java_function_call\n"
     r"from bfcl\.model_handler\.parser\.js_parser import parse_javascript_function_call\n",
     "def parse_java_function_call(*a, **k):\n"
     "    raise NotImplementedError('the Java category needs tree_sitter + tree_sitter_java; only Python categories are run')\n\n\n"
     "def parse_javascript_function_call(*a, **k):\n"
     "    raise NotImplementedError('the JavaScript category needs tree_sitter + tree_sitter_javascript; only Python categories are run')\n",
     "tree-sitter parsers are not a dependency"),
    (r"from tenacity import \((.*?)\)\n",
     "try:\n    from tenacity import (\\1)\nexcept ImportError:  # only the API retry helpers use it\n"
     "    retry = retry_if_exception_message = retry_if_exception_type = wait_random_exponential = None\n",
     "tenacity is optional"),
]


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "brahmai-eval"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)


def _wanted(rel: str) -> bool:
    return rel in DATA_ROWS or rel in SCORER_FILES or rel == UTILS_FILE or rel == "LICENSE" or (
        rel.startswith(FUNC_SOURCE_DIR) and rel.endswith(".py"))


def _read_archive(tar_path: str) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if rel == "LICENSE":
                key = "LICENSE"
            elif rel.startswith(SUBDIR):
                key = rel[len(SUBDIR):]
            else:
                continue
            if _wanted(key):
                fh = tar.extractfile(member)
                if fh is not None:
                    found[key] = fh.read()
    return found


def _read_tree(root: str) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    base = os.path.join(root, SUBDIR.rstrip("/"))
    for rel in list(DATA_ROWS) + list(SCORER_FILES) + [UTILS_FILE]:
        with open(os.path.join(base, rel), "rb") as fh:
            found[rel] = fh.read()
    src = os.path.join(base, FUNC_SOURCE_DIR)
    for f in os.listdir(src):
        if f.endswith(".py"):
            with open(os.path.join(src, f), "rb") as fh:
                found[FUNC_SOURCE_DIR + f] = fh.read()
    with open(os.path.join(root, "LICENSE"), "rb") as fh:
        found["LICENSE"] = fh.read()
    return found


def _rows(blob: bytes) -> int:
    return sum(1 for l in blob.decode("utf-8").splitlines() if l.strip())


def _patch(rel_dst: str, text: str) -> tuple[str, list[str]]:
    dots = "." * (rel_dst.count("/") + 1)
    applied = []
    for pat, rep, reason in PATCHES:
        text, n = re.subn(pat, rep.replace("{dots}", dots), text, flags=re.S)
        if n:
            applied.append(f"{rel_dst}: {reason} (x{n})")
    leftover = [l for l in text.splitlines() if re.match(r"\s*(from|import)\s+bfcl\b", l)]
    if leftover:
        raise RuntimeError(f"{rel_dst} still imports the upstream package after patching: {leftover}")
    return text, applied


def _extract_helpers(utils_src: str) -> str:
    """Copy the two format helpers verbatim out of bfcl/utils.py (the module itself imports the provider SDKs)."""
    tree = ast.parse(utils_src)
    out = ['"""Format helpers copied verbatim from bfcl/utils.py of the pinned commit."""', ""]
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FORMAT_HELPERS:
            out.append(ast.get_source_segment(utils_src, node) + "\n")
            found.add(node.name)
    missing = set(FORMAT_HELPERS) - found
    if missing:
        raise RuntimeError(f"bfcl/utils.py lacks {sorted(missing)}")
    return "\n".join(out)


def prepare(data_dir: str, source_dir: str | None = None) -> None:
    """Materialise `<data_dir>/bfcl/`.  Idempotent: returns at once when `.complete` exists.  `source_dir` may point
    at a checkout of the repository at SHA to skip the download."""
    out_root = os.path.join(data_dir, "bfcl")
    marker = os.path.join(out_root, ".complete")
    if os.path.exists(marker):
        return
    os.makedirs(out_root, exist_ok=True)
    if source_dir:
        files = _read_tree(source_dir)
    else:
        tmp = tempfile.mkdtemp(prefix=".download-", dir=out_root)
        try:
            tar_path = os.path.join(tmp, f"gorilla-{SHA}.tar.gz")
            _download(ARCHIVE_URL, tar_path)
            files = _read_archive(tar_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    missing = [k for k in list(DATA_ROWS) + list(SCORER_FILES) + [UTILS_FILE, "LICENSE"] if k not in files]
    if missing:
        raise RuntimeError(f"BFCL archive @{SHA[:8]} is missing {len(missing)} file(s), e.g. {missing[:3]}")
    for rel, expected in DATA_ROWS.items():
        got = _rows(files[rel])
        if got != expected:
            raise RuntimeError(f"BFCL {rel}@{SHA[:8]}: expected {expected} rows, found {got}")

    for rel in DATA_ROWS:
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(files[rel])

    pkg = os.path.join(out_root, PACKAGE)
    applied: list[str] = []
    for rel_src, rel_dst in SCORER_FILES.items():
        text, a = _patch(rel_dst, files[rel_src].decode("utf-8"))
        applied += a
        dst = os.path.join(pkg, rel_dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(text)
    src_files = sorted(k for k in files if k.startswith(FUNC_SOURCE_DIR))
    if len(src_files) < len(FUNC_DOC_FILES):
        raise RuntimeError(f"BFCL archive has only {len(src_files)} API source files under {FUNC_SOURCE_DIR}")
    for rel_src in src_files:
        rel_dst = "func_source_code/" + rel_src[len(FUNC_SOURCE_DIR):]
        text, a = _patch(rel_dst, files[rel_src].decode("utf-8"))
        applied += a
        dst = os.path.join(pkg, rel_dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(pkg, "format_helpers.py"), "w", encoding="utf-8") as fh:
        fh.write(_extract_helpers(files[UTILS_FILE].decode("utf-8")))
    for d in (pkg, os.path.join(pkg, "type_convertor"), os.path.join(pkg, "func_source_code")):
        p = os.path.join(d, "__init__.py")
        if not os.path.exists(p):
            with open(p, "w") as fh:
                fh.write(f"# BFCL scorer vendored from {SOURCE}\n")

    with open(os.path.join(out_root, "LICENSE"), "wb") as fh:
        fh.write(files["LICENSE"])
    with open(os.path.join(out_root, "SOURCE.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": SOURCE, "archive": ARCHIVE_URL, "license": LICENSE, "rows": DATA_ROWS, "patches": applied},
                  fh, indent=1)
    with open(marker, "w") as fh:
        fh.write(SOURCE + "\n")


def load_scorer(data_dir: str):
    """Import the vendored scorer package from `<data_dir>/bfcl/` (prepared on demand)."""
    prepare(data_dir)
    root = os.path.join(data_dir, "bfcl")
    if PACKAGE in sys.modules:
        mod = sys.modules[PACKAGE]
        if os.path.dirname(os.path.abspath(mod.__file__)) != os.path.abspath(os.path.join(root, PACKAGE)):
            raise RuntimeError(f"{PACKAGE} is already imported from {mod.__file__}, not from {root}")
        return mod
    if root not in sys.path:
        sys.path.insert(0, root)
    mod = importlib.import_module(PACKAGE)
    for sub in ("type_mappings", "default_prompts", "model_style", "model_utils", "ast_checker", "format_helpers",
                "multi_turn_utils", "multi_turn_checker"):
        importlib.import_module(f"{PACKAGE}.{sub}")
    return mod


def data_path(data_dir: str, *parts: str) -> str:
    return os.path.join(data_dir, "bfcl", "data", *parts)


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
