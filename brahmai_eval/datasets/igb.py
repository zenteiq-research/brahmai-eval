# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""IndicGenBench data preparation for the `igb_*` tasks.

Source: https://github.com/google-research-datasets/indic-gen-bench at the pinned commit
below (the Hugging Face mirror of the same data is gated, the GitHub release is not).
`prepare(data_dir)` downloads the GitHub archive of that commit, reads the test and dev
files of the four families (XQuAD-IN, XorQA-IN, Flores-IN, CrossSum-IN) for the 13
languages the tasks cover, and writes one `test.json` / `dev.json` pair per task under
`<data_dir>/igb/<task>/` (same `{"examples": [...]}` layout as the release files).

Test sets: XQuAD-IN and XorQA-IN are used in full; Flores-IN is a seeded random sample of
500 of the 1,012 test sentences per direction; CrossSum-IN a seeded random sample of up to
250 test articles per language.  Few-shot pools: a seeded random sample of 64 dev rows per
task (8 for CrossSum-IN, which is run 0-shot).  One `random.Random(SEED)` instance is
shared by every draw, in the fixed task order of `TASKS`, so the whole procedure is a
single deterministic function of the pinned source.

Licenses (per the repository's LICENSE file): XQuAD-IN and Flores-IN CC-BY-SA-4.0,
XorQA-IN MIT, CrossSum-IN CC-BY-NC-SA-4.0.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tarfile
import tempfile
import urllib.request

REPO_URL = "https://github.com/google-research-datasets/indic-gen-bench"
SHA = "c96a10d90ed9b38cc2108cac6f515a1b8bfdc230"
ARCHIVE_URL = f"{REPO_URL}/archive/{SHA}.tar.gz"
SOURCE = f"{REPO_URL}@{SHA}"

SEED = 1234
FEWSHOT_POOL = 64
CROSSSUM_POOL = 8
CAPS = {"xquad": None, "xorqa": None, "flores": 500, "crosssum": 250}

LANGS = ["hi", "bn", "ta", "te", "ml", "kn", "gu", "mr", "pa", "or", "as", "ur"]
EN = "en"

LICENSES = {
    "xquad": "CC-BY-SA-4.0",
    "flores": "CC-BY-SA-4.0",
    "xorqa": "MIT",
    "crosssum": "CC-BY-NC-SA-4.0",
}


def _tasks() -> list[tuple[str, str, str, str]]:
    """(task, family, test file, dev file) in the fixed order the sampler draws in."""
    out = []
    for lg in LANGS + [EN]:
        out.append((f"xquad_{lg}", "xquad", f"xquad_in/xquad_{lg}_test.json", f"xquad_in/xquad_{lg}_dev.json"))
    for lg in LANGS + [EN]:
        out.append((f"xorqa_{lg}", "xorqa", f"xorqa_in/xorqa_{lg}_test.json", f"xorqa_in/xorqa_{lg}_dev.json"))
    for lg in LANGS:
        for a, b in ((EN, lg), (lg, EN)):
            out.append((f"flores_{a}_{b}", "flores", f"flores_in/flores_{a}_{b}_test.json",
                        f"flores_in/flores_{a}_{b}_dev.json"))
    for lg in LANGS + [EN]:
        out.append((f"crosssum_{lg}", "crosssum", f"crosssum_in/crosssum_english-{lg}_test.json",
                    f"crosssum_in/crosssum_english-{lg}_dev.json"))
    return out


TASKS = _tasks()

# Row counts of the release files at SHA, keyed by the path inside the repository.
# Checked after download; a mismatch means the source changed and the tasks must not run.
_XORQA_TEST = {"hi": 539, "bn": 539, "ta": 538, "te": 539, "ml": 537, "kn": 536, "gu": 535,
               "mr": 538, "pa": 532, "or": 537, "as": 539, "ur": 539, "en": 539}
_XORQA_DEV = {"hi": 499, "bn": 500, "ta": 500, "te": 500, "ml": 500, "kn": 500, "gu": 499,
              "mr": 498, "pa": 500, "or": 499, "as": 500, "ur": 500, "en": 500}
_CROSSSUM_TEST = {"hi": 481, "bn": 155, "ta": 315, "te": 212, "ml": 500, "kn": 500, "gu": 104,
                  "mr": 118, "pa": 136, "or": 500, "as": 500, "ur": 632, "en": 500}
_CROSSSUM_DEV = {"hi": 463, "bn": 153, "ta": 315, "te": 212, "ml": 100, "kn": 100, "gu": 90,
                 "mr": 118, "pa": 130, "or": 100, "as": 100, "ur": 632, "en": 100}
SOURCE_ROWS: dict[str, int] = {}
for _lg in LANGS + [EN]:
    SOURCE_ROWS[f"xquad_in/xquad_{_lg}_test.json"] = 1190
    SOURCE_ROWS[f"xquad_in/xquad_{_lg}_dev.json"] = 110 if _lg == "te" else 111
    SOURCE_ROWS[f"xorqa_in/xorqa_{_lg}_test.json"] = _XORQA_TEST[_lg]
    SOURCE_ROWS[f"xorqa_in/xorqa_{_lg}_dev.json"] = _XORQA_DEV[_lg]
    SOURCE_ROWS[f"crosssum_in/crosssum_english-{_lg}_test.json"] = _CROSSSUM_TEST[_lg]
    SOURCE_ROWS[f"crosssum_in/crosssum_english-{_lg}_dev.json"] = _CROSSSUM_DEV[_lg]
for _lg in LANGS:
    for _a, _b in ((EN, _lg), (_lg, EN)):
        SOURCE_ROWS[f"flores_in/flores_{_a}_{_b}_test.json"] = 1012
        SOURCE_ROWS[f"flores_in/flores_{_a}_{_b}_dev.json"] = 997

# Row counts of the per-task files prepare() writes: (test, dev) per task.
TASK_ROWS: dict[str, tuple[int, int]] = {}
for _task, _fam, _t, _d in TASKS:
    _cap = CAPS[_fam]
    _pool = CROSSSUM_POOL if _fam == "crosssum" else FEWSHOT_POOL
    TASK_ROWS[_task] = (min(SOURCE_ROWS[_t], _cap) if _cap else SOURCE_ROWS[_t], min(SOURCE_ROWS[_d], _pool))

NEEDED_FILES = sorted(SOURCE_ROWS)
LICENSE_FILES = ["LICENSE", "LICENSE-CC-BY-SA4.0", "LICENSE-MIT", "LICENSE-CC-BY-NC-SA4.0"]


def _sample(rows: list, cap: int | None, rng: random.Random) -> list:
    """Seeded random subset of `cap` rows in original order; the whole list (no draw) if it fits."""
    if cap is None or cap >= len(rows):
        return rows
    idx = sorted(rng.sample(range(len(rows)), cap))
    return [rows[i] for i in idx]


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "brahmai-eval"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)


def _read_archive(tar_path: str, wanted: set[str]) -> dict[str, bytes]:
    """Read the wanted repository-relative files straight out of the archive (nothing is extracted to disk)."""
    found: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/", 1)
            if len(parts) != 2 or not member.isfile():
                continue
            rel = parts[1]
            if rel in wanted:
                fh = tar.extractfile(member)
                if fh is not None:
                    found[rel] = fh.read()
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"IndicGenBench archive is missing {len(missing)} file(s), e.g. {sorted(missing)[:3]}")
    return found


def _read_tree(root: str, wanted: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for rel in wanted:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise RuntimeError(f"IndicGenBench source tree lacks {rel}")
        with open(p, "rb") as fh:
            found[rel] = fh.read()
    return found


def _examples(blob: bytes) -> list:
    return json.loads(blob.decode("utf-8"))["examples"]


def prepare(data_dir: str, source_dir: str | None = None) -> None:
    """Materialise `<data_dir>/igb/<task>/{test,dev}.json` for every task in TASKS.

    Idempotent: returns at once when `<data_dir>/igb/.complete` exists.  `source_dir` may point at
    a checkout of the repository at SHA to skip the download.
    """
    out_root = os.path.join(data_dir, "igb")
    marker = os.path.join(out_root, ".complete")
    if os.path.exists(marker):
        return
    os.makedirs(out_root, exist_ok=True)
    wanted = set(NEEDED_FILES) | set(LICENSE_FILES)

    if source_dir:
        files = _read_tree(source_dir, wanted)
    else:
        tmp = tempfile.mkdtemp(prefix=".download-", dir=out_root)
        try:
            tar_path = os.path.join(tmp, f"indic-gen-bench-{SHA}.tar.gz")
            _download(ARCHIVE_URL, tar_path)
            files = _read_archive(tar_path, wanted)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    data = {rel: _examples(files[rel]) for rel in NEEDED_FILES}
    for rel, expected in SOURCE_ROWS.items():
        got = len(data[rel])
        if got != expected:
            raise RuntimeError(f"IndicGenBench {rel}@{SHA[:8]}: expected {expected} rows, found {got}")

    rng = random.Random(SEED)
    for task, fam, test_rel, dev_rel in TASKS:
        if fam == "flores":
            test = _sample(data[test_rel], CAPS[fam], rng)
            dev = _sample(data[dev_rel], FEWSHOT_POOL, rng)
        elif fam == "crosssum":
            test = _sample(data[test_rel], CAPS[fam], rng)
            dev = _sample(data[dev_rel], CROSSSUM_POOL, rng)
        else:  # xquad, xorqa: the dev pool is drawn before the (uncapped) test set
            dev = _sample(data[dev_rel], FEWSHOT_POOL, rng)
            test = _sample(data[test_rel], CAPS[fam], rng)
        exp_test, exp_dev = TASK_ROWS[task]
        if (len(test), len(dev)) != (exp_test, exp_dev):
            raise RuntimeError(f"igb_{task}: expected {exp_test}/{exp_dev} test/dev rows, got {len(test)}/{len(dev)}")
        tdir = os.path.join(out_root, task)
        os.makedirs(tdir, exist_ok=True)
        for split, rows in (("test", test), ("dev", dev)):
            with open(os.path.join(tdir, f"{split}.json"), "w", encoding="utf-8") as fh:
                json.dump({"examples": rows}, fh, ensure_ascii=False)

    for name in LICENSE_FILES:
        with open(os.path.join(out_root, name), "wb") as fh:
            fh.write(files[name])
    with open(os.path.join(out_root, "SOURCE.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": SOURCE, "archive": ARCHIVE_URL, "seed": SEED, "licenses": LICENSES,
                   "tasks": {f"igb_{t}": {"test": TASK_ROWS[t][0], "dev": TASK_ROWS[t][1]} for t, *_ in TASKS}},
                  fh, indent=1)
    with open(marker, "w") as fh:
        fh.write(SOURCE + "\n")


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
