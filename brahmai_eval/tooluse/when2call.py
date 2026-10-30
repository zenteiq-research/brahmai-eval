# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""When2Call (nvidia/When2Call, test/mcq, 3,652 items): does the model know WHEN to call a tool?

Each item is a user request plus declared tools, with one correct behaviour out of {tool_call, request_for_info,
cannot_answer}; a fourth candidate, `direct` (answering without the tool), is always a distractor.  Tools are
declared the way the run's format declares them (chat template `tools=` for "json", BFCL's system prompt for
"python"); there is no other system text.

Two protocols, both through the chat template:
  mcq  the paper's protocol: the log-likelihood of each of the four candidate replies rendered as an assistant turn
       (the tool_call candidate rendered the way the template renders a call, or as a <tool_call> JSON block /
       `[name(arg=val)]` when the template has no rendering for calls), summed over the reply tokens including the
       end-of-turn token.  mcq_acc = argmax of the sum, mcq_acc_norm = argmax of the per-token mean; mcq_f1 is the
       macro-F1 over the three gold classes (predictions by sum); tool_hallucination = share of cannot_answer items
       where the tool_call candidate wins.
  gen  greedy generation (512 tokens): gen_call_recall = share of tool_call items whose reply contains a parseable
       call naming a declared tool; gen_false_call = share of non-tool_call items whose reply contains any call
       (also split by class); call_valid = parseable + declared name, among replies that call.

`overall` = mcq_f1.  Items whose prompt plus the longest candidate does not fit the context window are excluded
and counted in `context_overflow`.
"""
from __future__ import annotations

import collections
import json
import time

from ..datasets.when2call import CANDIDATES, CLASSES, load
from .common import (
    TOOL_CALL_RE,
    Engine,
    calls_to_strings,
    check_prompt_carries_tools,
    clip01,
    functions_to_tools,
    is_empty_calls,
    prepare_functions,
    python_messages,
    safe_div,
    spread,
    write_record,
)

MAX_NEW = 512


def macro_f1(pred: list[str], gold: list[str]) -> tuple[float, dict[str, float]]:
    out = {}
    for c in CLASSES:
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        fp = sum(p == c and g != c for p, g in zip(pred, gold))
        fn = sum(p != c and g == c for p, g in zip(pred, gold))
        pr, rc = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        out[c] = safe_div(2 * pr * rc, pr + rc)
    return sum(out.values()) / len(out), out


def candidate_messages(engine: Engine, cand: str, answer: str) -> list[tuple[dict, str]]:
    """The assistant-turn variants to try for a candidate reply, most template-native first."""
    if cand != "tool_call":
        return [({"role": "assistant", "content": answer}, "text")]
    call = json.loads(answer)
    name, args = call["name"], call.get("arguments") or {}
    if engine.format == "python":
        return [({"role": "assistant", "content": "[" + calls_to_strings([{name: args}])[0] + "]"}, "python")]
    return [({"role": "assistant", "content": "", "tool_calls": [{"id": "000000001", "type": "function", "function": {"name": name, "arguments": args}}]}, "structured"),
            ({"role": "assistant", "content": "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"}, "text")]


def render_candidate(engine: Engine, msgs: list, tools: list | None, prompt: str, cand: str, answer: str) -> tuple[str, str]:
    """-> (continuation text after the generation prompt, form used)."""
    for asst, form in candidate_messages(engine, cand, answer):
        try:
            full, _ = engine.render(msgs + [asst], tools, add_generation_prompt=False)
        except RuntimeError:
            continue
        tail = full[len(prompt):] if full.startswith(prompt) else None
        if tail and (form != "structured" or json.loads(answer)["name"] in tail):
            return tail, form
    asst, form = candidate_messages(engine, cand, answer)[-1]
    return asst["content"] + engine.eos, form + "+eos"


def run(engine: Engine, data_dir: str, out_path: str, limit: int | None = None) -> tuple[int, dict]:
    from ..datasets.bfcl import load_scorer
    scorer = load_scorer(data_dir)
    items = load(data_dir)
    if limit:
        items = spread(items, limit)
    t0 = time.time()
    rendered, overflow = [], []
    gated = False
    for it in items:
        functions = [json.loads(t) for t in it["tools"]]
        names = {f["name"] for f in functions}
        category = (it.get("source_id") or "live_simple").rsplit("_", 1)[0]
        if engine.format == "python":
            functions = prepare_functions(scorer, functions, category)
            msgs, tools = python_messages(scorer, [{"role": "user", "content": it["question"]}], functions, category), None
        else:
            msgs, tools = [{"role": "user", "content": it["question"]}], functions_to_tools(scorer, functions) or None
        prompt, flag = engine.render(msgs, tools)
        if not gated:
            check_prompt_carries_tools(prompt, functions, engine.format)
            gated = True
        p_ids = engine.encode(prompt)
        cands, forms = {}, {}
        for c in CANDIDATES:
            text, forms[c] = render_candidate(engine, msgs, tools, prompt, c, it["answers"][c])
            cands[c] = (text, engine.encode(text))
        need = len(p_ids) + max(len(ids) for _, ids in cands.values()) + 1
        if need > engine.max_len or not engine.fits(p_ids, MAX_NEW):
            overflow.append(it)
            continue
        rendered.append((it, prompt, p_ids, cands, forms, functions, names, flag))
    print(f"[when2call/{engine.format}] {len(rendered)} items rendered, {len(overflow)} exceed the context window  {time.time() - t0:.0f}s", flush=True)

    # mcq: log-likelihood of every (item, candidate)
    pairs, idx = [], []
    for i, (it, prompt, p_ids, cands, *_rest) in enumerate(rendered):
        for c in CANDIDATES:
            pairs.append((p_ids, cands[c][1]))
            idx.append((i, c))
    score: dict[int, dict] = collections.defaultdict(dict)
    for (i, c), (total, n) in zip(idx, engine.logprob_sums(pairs)):
        score[i][c] = (total, safe_div(total, n))
    gold = [r[0]["correct_answer"] for r in rendered]
    pred = [max(CANDIDATES, key=lambda k: score[i][k][0]) for i in range(len(rendered))]
    pred_norm = [max(CANDIDATES, key=lambda k: score[i][k][1]) for i in range(len(rendered))]
    print(f"[when2call/{engine.format}] mcq scored  {time.time() - t0:.0f}s", flush=True)

    # gen: greedy replies
    gens = engine.generate([r[2] for r in rendered], MAX_NEW)
    has_call, valid, parsed_all = [], [], []
    for (it, prompt, p_ids, cands, forms, functions, names, flag), g in zip(rendered, gens):
        parsed, derr = None, None
        try:
            parsed = engine.parse_calls(scorer, g.text, functions)
            if is_empty_calls(scorer, parsed):
                parsed, derr = None, "empty call list"
        except ValueError as ex:
            derr = str(ex)
        called = parsed is not None or (engine.format == "json" and bool(TOOL_CALL_RE.search(g.text)))
        ok = parsed is not None and any(name in names for c in parsed for name in c)
        has_call.append(called)
        valid.append(ok)
        parsed_all.append((parsed, derr))
    print(f"[when2call/{engine.format}] generation scored  {time.time() - t0:.0f}s", flush=True)

    n = len(rendered)
    tc = [i for i, g in enumerate(gold) if g == "tool_call"]
    ntc = [i for i, g in enumerate(gold) if g != "tool_call"]
    ca = [i for i, g in enumerate(gold) if g == "cannot_answer"]
    rfi = [i for i, g in enumerate(gold) if g == "request_for_info"]
    called = [i for i in range(n) if has_call[i]]
    mf1, per_class = macro_f1(pred, gold)
    metrics = {
        "mcq_acc": safe_div(sum(p == g for p, g in zip(pred, gold)), n),
        "mcq_acc_norm": safe_div(sum(p == g for p, g in zip(pred_norm, gold)), n),
        "mcq_f1": mf1,
        **{f"mcq_f1_{c}": v for c, v in per_class.items()},
        "tool_hallucination": safe_div(sum(pred[i] == "tool_call" for i in ca), len(ca)),
        "gen_call_recall": safe_div(sum(valid[i] for i in tc), len(tc)),
        "gen_false_call": safe_div(sum(has_call[i] for i in ntc), len(ntc)),
        "gen_false_call_cannot_answer": safe_div(sum(has_call[i] for i in ca), len(ca)),
        "gen_false_call_request_for_info": safe_div(sum(has_call[i] for i in rfi), len(rfi)),
        "call_valid": safe_div(sum(valid[i] for i in called), len(called)),
        "context_overflow": safe_div(len(overflow), len(items)),
    }
    metrics["overall"] = metrics["mcq_f1"]
    with open(out_path, "w", encoding="utf-8") as fo:
        for i, ((it, prompt, p_ids, cands, forms, functions, names, flag), g) in enumerate(zip(rendered, gens)):
            write_record(fo, {"task": "when2call", "id": it["uuid"], "gold": gold[i], "prompt": prompt, "candidates": {c: cands[c][0] for c in CANDIDATES},
                              "candidate_forms": forms, "scores": score[i], "mcq_pred": pred[i], "mcq_pred_norm": pred_norm[i],
                              "output": g.raw, "text": g.text, "parsed": parsed_all[i][0], "parse_error": parsed_all[i][1],
                              "has_call": has_call[i], "call_valid": valid[i], "finish": g.finish, "n_gen_toks": g.n_tokens, "render": flag})
        for it in overflow:
            write_record(fo, {"task": "when2call", "id": it["uuid"], "gold": it["correct_answer"], "output": None, "parsed": None, "why": "context_overflow"})
    conf = collections.Counter((g, p) for g, p in zip(gold, pred))
    print(f"[when2call/{engine.format}] n={n} mcq_acc={metrics['mcq_acc']:.4f} mcq_f1={mf1:.4f} halluc={metrics['tool_hallucination']:.4f} "
          f"call_recall={metrics['gen_call_recall']:.4f} false_call={metrics['gen_false_call']:.4f} "
          f"confusion={ {f'{g}->{p}': v for (g, p), v in sorted(conf.items())} }  {time.time() - t0:.0f}s", flush=True)
    return n, {k: clip01(v) for k, v in metrics.items()}
