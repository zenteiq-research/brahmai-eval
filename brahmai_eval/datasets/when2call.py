# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""When2Call test set (multiple-choice split) for the `when2call` tool-use task.

Source: Hugging Face dataset nvidia/When2Call at the pinned revision below, CC-BY-4.0.  `prepare(data_dir)`
downloads `test/when2call_test_mcq.jsonl` (3,652 items) into `<data_dir>/when2call/test_mcq.jsonl` together with
the dataset card.  Each item carries a user question, the declared tools (BFCL-style function docs as JSON strings),
four candidate replies (tool_call, request_for_info, cannot_answer, direct) and the correct behaviour.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

REPO = "nvidia/When2Call"
SHA = "0582f7749df63a96fdc3070932e83e72396ace53"
SOURCE = f"{REPO}@{SHA}"
LICENSE = "CC-BY-4.0"
FILE = "test/when2call_test_mcq.jsonl"
ROWS = 3652
FIELDS = ("uuid", "question", "correct_answer", "answers", "tools")
CLASSES = ["tool_call", "request_for_info", "cannot_answer"]
CANDIDATES = CLASSES + ["direct"]


def _validate(rows: list[dict]) -> None:
    if len(rows) != ROWS:
        raise RuntimeError(f"When2Call {FILE}@{SHA[:8]}: expected {ROWS} rows, found {len(rows)}")
    for r in rows:
        missing = [f for f in FIELDS if f not in r]
        if missing:
            raise RuntimeError(f"When2Call row {r.get('uuid')} lacks {missing}")
        if r["correct_answer"] not in CLASSES or any(c not in r["answers"] for c in CANDIDATES):
            raise RuntimeError(f"When2Call row {r['uuid']}: unexpected answer layout")


def prepare(data_dir: str) -> None:
    """Materialise `<data_dir>/when2call/test_mcq.jsonl`.  Idempotent: returns at once when `.complete` exists."""
    out_root = os.path.join(data_dir, "when2call")
    marker = os.path.join(out_root, ".complete")
    if os.path.exists(marker):
        return
    os.makedirs(out_root, exist_ok=True)
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(REPO, FILE, repo_type="dataset", revision=SHA)
    card = hf_hub_download(REPO, "README.md", repo_type="dataset", revision=SHA)
    with open(src, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    _validate(rows)
    shutil.copyfile(src, os.path.join(out_root, "test_mcq.jsonl"))
    shutil.copyfile(card, os.path.join(out_root, "README.md"))
    with open(os.path.join(out_root, "SOURCE.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": SOURCE, "file": FILE, "license": LICENSE, "rows": ROWS}, fh, indent=1)
    with open(marker, "w") as fh:
        fh.write(SOURCE + "\n")


def load(data_dir: str) -> list[dict]:
    prepare(data_dir)
    with open(os.path.join(data_dir, "when2call", "test_mcq.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    _validate(rows)
    return rows


if __name__ == "__main__":
    prepare(sys.argv[1])
