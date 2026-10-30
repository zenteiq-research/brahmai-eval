# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v3 single-turn categories (Python only), scored by the official AST checker.

Protocol per item: the category's function docs are declared either through the chat template (`tools=`, format
"json") or pasted into BFCL's system prompt (format "python"); the reply is greedy-decoded (512 tokens), parsed into
`[{name: {arg: val}}]` and checked against the possible answers.  Irrelevance items are correct when no function
is called, relevance items when one is.  Items whose prompt does not fit the context window are excluded and
counted in `context_overflow`; items upstream ships without a ground truth are excluded as well.

Metrics: accuracy per category; `nonlive_ast` (unweighted mean of simple/multiple/parallel/parallel_multiple),
`nonlive_overall` (the same plus irrelevance), `live_ast` and `live_overall` (item-weighted over the live
categories, as the leaderboard computes them), `overall` = mean(nonlive_overall, live_overall), `decode_rate`
(share of AST-category replies that parsed as a call at all; a collapse means a format problem, not a capability
result) and `context_overflow`.
"""
from __future__ import annotations

import time

from ..datasets.bfcl import SINGLE_TURN, data_path, load_scorer
from .common import (
    Engine,
    check_prompt_carries_tools,
    clip01,
    functions_to_tools,
    is_empty_calls,
    load_jsonl,
    prepare_functions,
    python_messages,
    safe_div,
    spread,
    write_record,
)

MAX_NEW = 512
NONLIVE_AST = ["simple", "multiple", "parallel", "parallel_multiple"]
LIVE_AST = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]


def build_prompt(engine: Engine, scorer, entry: dict) -> tuple[list, list, str, str]:
    """-> (messages, preprocessed functions, category, render flag)."""
    category = entry["id"].rsplit("_", 1)[0]
    functions = prepare_functions(scorer, entry["function"], category)
    if engine.format == "python":
        msgs = python_messages(scorer, entry["question"][0], functions, category)
        prompt, flag = engine.render(msgs)
    else:
        prompt, flag = engine.render(entry["question"][0], functions_to_tools(scorer, functions) or None)
    return prompt, functions, category, flag


def run(engine: Engine, data_dir: str, out_path: str, limit: int | None = None) -> tuple[int, dict]:
    scorer = load_scorer(data_dir)
    ast_checker = scorer.ast_checker.ast_checker
    stats: dict[str, dict] = {}
    gated = False
    with open(out_path, "w", encoding="utf-8") as fo:
        for name, kind, n_expected in SINGLE_TURN:
            t0 = time.time()
            entries = load_jsonl(data_path(data_dir, f"BFCL_v3_{name}.json"))
            if len(entries) != n_expected:
                raise RuntimeError(f"BFCL {name}: expected {n_expected} items, found {len(entries)}")
            if limit:
                entries = spread(entries, limit)
            gold: dict = {}
            n_missing_gold = 0
            if kind == "ast":
                gold = {g["id"]: g["ground_truth"] for g in load_jsonl(data_path(data_dir, "possible_answer", f"BFCL_v3_{name}.json"))}
                missing = [e["id"] for e in entries if e["id"] not in gold]
                if len(missing) > 0.01 * len(entries):
                    raise RuntimeError(f"BFCL {name}: {len(missing)}/{len(entries)} items have no ground truth, e.g. {missing[:3]}")
                n_missing_gold = len(missing)
                entries = [e for e in entries if e["id"] in gold]

            ids_list, meta, n_overflow = [], [], 0
            for e in entries:
                prompt, functions, category, flag = build_prompt(engine, scorer, e)
                if not gated:
                    check_prompt_carries_tools(prompt, functions, engine.format)
                    gated = True
                ids = engine.encode(prompt)
                if not engine.fits(ids, MAX_NEW):
                    n_overflow += 1
                    write_record(fo, {"task": "bfcl_single_turn", "category": name, "kind": kind, "id": e["id"], "prompt": prompt,
                                      "output": None, "parsed": None, "correct": None, "why": "context_overflow", "n_prompt_toks": len(ids)})
                    continue
                ids_list.append(ids)
                meta.append((e["id"], functions, category, prompt, flag))

            gens = engine.generate(ids_list, MAX_NEW)
            n_ok = n_decoded = n_trunc = 0
            for (eid, functions, category, prompt, flag), g in zip(meta, gens):
                n_trunc += g.finish == "length"
                decoded, derr = None, None
                try:
                    decoded = engine.parse_calls(scorer, g.text, functions)
                    n_decoded += 1
                except ValueError as ex:
                    derr = str(ex)
                if kind == "ast":
                    if decoded is None:
                        ok, why = False, f"decode_failed: {derr}"
                    else:
                        try:
                            r = ast_checker(functions, decoded, gold[eid], "Python", category, "local")
                            ok, why = bool(r["valid"]), "" if r["valid"] else str(r.get("error_type", ""))
                        except Exception as ex:  # noqa: BLE001 -- a checker exception is a wrong answer, recorded
                            ok, why = False, f"checker_raised: {type(ex).__name__}: {ex}"
                else:
                    has_call = decoded is not None and not is_empty_calls(scorer, decoded)
                    ok = (not has_call) if kind == "irrelevance" else has_call
                    why = "" if ok else ("called_a_function" if kind == "irrelevance" else "no_function_call")
                n_ok += ok
                write_record(fo, {"task": "bfcl_single_turn", "category": name, "kind": kind, "id": eid, "prompt": prompt, "output": g.raw,
                                  "text": g.text, "parsed": decoded, "parse_error": derr, "correct": ok, "why": why, "finish": g.finish,
                                  "n_gen_toks": g.n_tokens, "render": flag})
            n = len(meta)
            stats[name] = {"kind": kind, "n": n, "correct": n_ok, "decoded": n_decoded, "overflow": n_overflow, "missing_gold": n_missing_gold}
            acc, rate = safe_div(n_ok, n), safe_div(n_decoded, n)
            flag = "  *** decode rate collapsed: a format problem, not a capability result ***" if kind == "ast" and n and rate < 0.10 else ""
            print(f"[bfcl_single_turn/{engine.format}] {name:24s} {kind:12s} n={n:5d} acc={acc:.4f} decode={rate:.4f} "
                  f"trunc={n_trunc} overflow={n_overflow} {time.time() - t0:6.1f}s{flag}", flush=True)

    def acc(c: str) -> float:
        return safe_div(stats[c]["correct"], stats[c]["n"])

    def weighted(cats: list[str]) -> float:
        return safe_div(sum(stats[c]["correct"] for c in cats), sum(stats[c]["n"] for c in cats))

    def unweighted(cats: list[str]) -> float:
        return sum(acc(c) for c in cats) / len(cats)

    metrics = {c: acc(c) for c in stats}
    metrics["nonlive_ast"] = unweighted(NONLIVE_AST)
    metrics["nonlive_overall"] = unweighted(NONLIVE_AST + ["irrelevance"])
    metrics["live_ast"] = weighted(LIVE_AST)
    metrics["live_overall"] = weighted(LIVE_AST + ["live_irrelevance", "live_relevance"])
    metrics["overall"] = (metrics["nonlive_overall"] + metrics["live_overall"]) / 2
    ast_cats = [c for c, s in stats.items() if s["kind"] == "ast"]
    metrics["decode_rate"] = safe_div(sum(stats[c]["decoded"] for c in ast_cats), sum(stats[c]["n"] for c in ast_cats))
    total = sum(s["n"] + s["overflow"] for s in stats.values())
    metrics["context_overflow"] = safe_div(sum(s["overflow"] for s in stats.values()), total)
    return sum(s["n"] for s in stats.values()), {k: clip01(v) for k, v in metrics.items()}
