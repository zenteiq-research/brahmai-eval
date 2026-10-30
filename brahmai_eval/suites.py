# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Suites: named groups of benchmark tasks.

A task row is {label, task, num_fewshot, chat, gen_kwargs, cost_secs, runner, unsafe_code, limit}.
  * runner "lmeval": a task name known to lm-evaluation-harness (stock, or one of the YAMLs under tasks/);
  * runner "tooluse": a function-calling benchmark implemented in brahmai_eval.tooluse (same vLLM engine).
`chat` means the prompt goes through the model's chat template; `cost_secs` is a rough relative cost used only to
balance work across GPUs.
"""
from __future__ import annotations

import glob
import json
import os

from .data import TASKS_SRC, require_tasks

# Stock lm-evaluation-harness tasks (downloaded from the Hub by the harness itself).
_STOCK = [
    # label, task, nshot, chat, gen_kwargs, cost
    ("gsm8k", "gsm8k", 5, True, None, 307),
    ("ifeval", "ifeval", 0, True, None, 560),
    ("gsm_plus", "gsm_plus", 5, True, None, 2500),
    ("aime25", "aime25", 0, True, {"max_gen_toks": 2048}, 300),
    ("boolq", "boolq", 0, False, None, 150),
    ("global_mmlu_en", "global_mmlu_full_en", 5, False, None, 500),
    ("hellaswag", "hellaswag", 10, False, None, 954),
    ("mmlu", "mmlu", 5, False, None, 496),
    ("arc_challenge", "arc_challenge", 25, False, None, 139),
    ("arc_easy", "arc_easy", 5, False, None, 59),
    ("piqa", "piqa", 5, False, None, 34),
    ("winogrande", "winogrande", 5, False, None, 17),
    ("sciq", "sciq", 5, False, None, 62),
    ("commonsense_qa", "commonsense_qa", 5, False, None, 23),
    ("truthfulqa_mc2", "truthfulqa_mc2", 0, False, None, 32),
    ("coqa", "coqa", 0, False, None, 93),
    ("triviaqa", "triviaqa", 5, False, None, 2327),
    ("drop", "drop", 0, True, None, 1439),
    ("mgsm_direct_bn", "mgsm_direct_bn", 5, True, None, 106),
    ("mgsm_direct_te", "mgsm_direct_te", 5, True, None, 107),
] + [(f"mmlu_pro_{c}", f"mmlu_pro_{c}", 5, True, None, n) for c, n in [
    ("biology", 1075), ("business", 1183), ("chemistry", 1698), ("computer_science", 615), ("economics", 1266),
    ("engineering", 1453), ("health", 1227), ("history", 571), ("law", 1651), ("math", 2026), ("other", 1386),
    ("philosophy", 748), ("physics", 1948), ("psychology", 1197)]]

_STOCK_SUITE = {
    "english": ["hellaswag", "mmlu", "arc_challenge", "arc_easy", "piqa", "winogrande", "sciq", "commonsense_qa",
                "truthfulqa_mc2", "coqa", "triviaqa", "drop", "boolq", "global_mmlu_en", "ifeval"],
    "math": ["gsm8k", "gsm_plus", "aime25", "mgsm_direct_bn", "mgsm_direct_te"],
    "mmlu_pro": [f"mmlu_pro_{c}" for c in ["biology", "business", "chemistry", "computer_science", "economics", "engineering",
                                           "health", "history", "law", "math", "other", "philosophy", "physics", "psychology"]],
}

# task families shipped under tasks/<family>/TASKS.json -> suite name
_FAMILY_SUITE = {"english": "english", "code": "code", "math": "math", "indic": "indic", "indic_ext": "indic_ext", "igb": "igb"}


def _row(label, task, nshot, chat, gen_kwargs, cost, runner="lmeval", unsafe=False, limit=None) -> dict:
    return {"label": label, "task": task, "num_fewshot": int(nshot), "chat": bool(chat), "gen_kwargs": gen_kwargs,
            "cost_secs": int(cost), "runner": runner, "unsafe_code": bool(unsafe), "limit": limit}


def _family_rows(family: str) -> list[dict]:
    p = os.path.join(TASKS_SRC, family, "TASKS.json")
    if not os.path.exists(p):
        return []
    rows = []
    for r in json.load(open(p)):
        gk = r.get("gen_kwargs")
        if isinstance(gk, str) and gk:
            gk = dict(kv.split("=", 1) for kv in gk.split(",")) if "=" in gk else json.loads(gk)
            gk = {k: (json.loads(v) if isinstance(v, str) else v) for k, v in gk.items()}
        rows.append(_row(r["label"], r["task"], r.get("num_fewshot", 0), r.get("chat", False), gk or None, r.get("cost_secs", 60),
                         unsafe=r.get("unsafe_code", False) or family == "code", limit=r.get("limit")))
    return rows


def all_suites() -> dict[str, list[dict]]:
    stock = {r[0]: _row(*r) for r in _STOCK}
    suites: dict[str, list[dict]] = {name: [stock[l] for l in labels] for name, labels in _STOCK_SUITE.items()}
    for family, suite in _FAMILY_SUITE.items():
        suites.setdefault(suite, []).extend(_family_rows(family))
    try:
        from .tooluse import TASKS as TOOL_TASKS
    except ImportError:
        TOOL_TASKS = {}
    if TOOL_TASKS:
        suites["tool_use"] = [_row(t, t, 0, True, None, spec["cost_secs"], runner="tooluse") for t, spec in TOOL_TASKS.items()]
    suites = {k: v for k, v in suites.items() if v}
    suites["all"] = [r for k in sorted(suites) for r in suites[k]]
    seen = set()
    suites["all"] = [r for r in suites["all"] if not (r["label"] in seen or seen.add(r["label"]))]
    return suites


def resolve(spec: str) -> list[dict]:
    """'all' | 'code,math' | a task label | a mix of suite names and task labels, comma separated."""
    suites = all_suites()
    by_label = {r["label"]: r for r in suites["all"]}
    out, seen = [], set()
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        rows = suites.get(name) or ([by_label[name]] if name in by_label else None)
        if rows is None:
            raise SystemExit(f"unknown suite or task '{name}'. Suites: {', '.join(sorted(suites))}. See `brahmai-eval list`.")
        for r in rows:
            if r["label"] not in seen:
                seen.add(r["label"]); out.append(r)
    return out


def families_present() -> list[str]:
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(require_tasks(), "*", "TASKS.json")))
