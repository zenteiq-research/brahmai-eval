# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Merge the shard records of a run into results.json and a readable results.md."""
from __future__ import annotations

import collections
import glob
import json
import os

from . import manifest as M
from .suites import all_suites

# which metric is THE number of a task, by preference order
_PRIMARY = ("exact_match,strict-match", "exact_match,flexible-extract", "prompt_level_strict_acc,none", "pass@1,create_test", "pass@1,none",
            "acc_norm,none", "acc,none", "exact_match,none", "f1_inlang,none", "f1,none", "chrf,none", "mc2,none", "em,none", "overall,none",
            "accuracy,none", "macro_f1,none", "mcq_f1,none")


def primary_metric(rec: dict) -> tuple[str, float] | None:
    keys = [k for k in rec if "," in k and "stderr" not in k and isinstance(rec[k], (int, float))]
    for p in _PRIMARY:
        if p in keys:
            return p, float(rec[p])
    return (keys[0], float(rec[keys[0]])) if keys else None


def summarize(out_dir: str) -> dict:
    plan = json.load(open(os.path.join(out_dir, "plan.json")))
    recs: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "shards", "*", "results.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            if r["label"] not in recs or "error" not in r:
                recs[r["label"]] = r
    suites = all_suites()
    membership = collections.defaultdict(list)
    for name, rows in suites.items():
        if name == "all":
            continue
        for r in rows:
            membership[r["label"]].append(name)
    tasks, errors = {}, []
    for t in plan["tasks"]:
        r = recs.get(t["label"])
        if r is None:
            errors.append(t["label"]); tasks[t["label"]] = {"task": t["task"], "status": "not_run"}; continue
        if "error" in r:
            errors.append(t["label"]); tasks[t["label"]] = {"task": t["task"], "status": "error", "error": r["error"]}; continue
        pm = primary_metric(r)
        tasks[t["label"]] = {"task": t["task"], "status": "ok", "n": r.get("n"), "num_fewshot": t["num_fewshot"], "chat": t["chat"], "secs": r.get("secs"),
                             "primary": {"metric": pm[0], "value": pm[1]} if pm else None,
                             "metrics": {k: v for k, v in r.items() if "," in k and isinstance(v, (int, float))}}
    by_suite = collections.defaultdict(list)
    for label, t in tasks.items():
        if t["status"] == "ok" and t["primary"]:
            for s in membership.get(label, ["other"]):
                by_suite[s].append(t["primary"]["value"])
    suite_means = {s: {"mean": sum(v) / len(v), "n_tasks": len(v)} for s, v in sorted(by_suite.items())}
    res = {"model": plan["model"], "suite": plan["suite"], "limit": plan.get("limit"), "config_hash": plan.get("config_hash"), "summarized": M.now(),
           "suite_means": suite_means, "tasks": tasks, "errors": errors}
    M.write_json(os.path.join(out_dir, "results.json"), res)
    lines = [f"# {plan['model']}", "", f"suite: `{plan['suite']}`" + (f" · limit {plan['limit']} (smoke test)" if plan.get("limit") else ""),
             f"engine: vLLM · max_model_len {plan['backend']['max_model_len']} · seed {plan['backend']['seed']} · config {plan.get('config_hash')}", "",
             "| suite | mean of primary metrics | tasks |", "|---|---:|---:|"]
    lines += [f"| {s} | {v['mean']:.4f} | {v['n_tasks']} |" for s, v in suite_means.items()]
    lines += ["", "| task | n-shot | chat | metric | value | items | secs |", "|---|---:|:---:|---|---:|---:|---:|"]
    for label, t in tasks.items():
        if t["status"] == "ok":
            pm = t["primary"] or {"metric": "-", "value": float("nan")}
            lines.append(f"| {label} | {t['num_fewshot']} | {'y' if t['chat'] else ''} | {pm['metric']} | {pm['value']:.4f} | {t['n']} | {t['secs']} |")
        else:
            lines.append(f"| {label} | | | **{t['status']}** | {str(t.get('error', ''))[:80]} | | |")
    open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return res


def print_summary(out_dir: str) -> None:
    print(open(os.path.join(out_dir, "results.md"), encoding="utf-8").read())
