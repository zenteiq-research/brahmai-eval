# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v3 multi-turn categories (base, miss_func, miss_param, long_context; 200 entries each), scored by the
official state/response checker after executing the model's calls against the benchmark's stateful API classes.

Inference mirrors upstream's multi-turn loop:
  * the function docs of the entry's API classes are declared as tools (format "json") or pasted into BFCL's
    system prompt (format "python");
  * per user turn, up to 20 steps: generate -> parse the calls -> no call (or an unparseable reply) ends the turn;
    otherwise every call is executed against the entry's API instances and each result is fed back (a `tool`
    message per call; a chat template that cannot render tool messages gets them as one user message holding the
    upstream `[{"role": "tool", "name": ..., "content": ...}]` list);
  * more than 20 steps in one turn force-terminates the entry (it fails);
  * miss_func: the held-out function docs are added at their turn with upstream's fixed user message;
  * a prompt that no longer fits the context window ends the entry (it fails; counted in `context_overflow`).
All entries advance in lock-step, one generation batch per round.

Upstream's executor and checker build a Python expression from each call and run it with `eval`, so a call written
by the model reaches them only if its name is a documented function of the entry's API classes (the held-out
miss_func functions included) and every argument name is a Python identifier; argument values are always rendered
as literals.  Any other call is refused: it is not executed, the model gets upstream's error text for an undefined
name, and the checker never sees it (upstream such a call would only have produced that error, so scores of normal
replies are unchanged).

Metrics: accuracy per category, `overall` (unweighted mean of the four), `context_overflow`.
"""
from __future__ import annotations

import copy
import json
import time

from ..datasets.bfcl import FUNC_DOC_FILES, MULTI_TURN, data_path, load_scorer
from .common import (
    TOOL_CALL_RE,
    Engine,
    calls_to_strings,
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

MAX_NEW = 1024
MODEL_NAME = "local_model"   # key of the executor's per-entry API instances (the checker suffixes its own)
FORMS_JSON = ("structured", "text", "user")
FORMS_PYTHON = ("text", "user")


def load_entries(scorer, data_dir: str, cat: str) -> list[dict]:
    """Upstream's test-case processing: attach the involved classes' function docs, pull the held-out ones aside."""
    entries = load_jsonl(data_path(data_dir, f"BFCL_v3_multi_turn_{cat}.json"))
    gold = {g["id"]: g["ground_truth"] for g in load_jsonl(data_path(data_dir, "possible_answer", f"BFCL_v3_multi_turn_{cat}.json"))}
    docs = {c: load_jsonl(data_path(data_dir, "multi_turn_func_doc", f)) for c, f in FUNC_DOC_FILES.items()}
    for e in entries:
        e["function"] = []
        for c in e["involved_classes"]:
            e["function"].extend(copy.deepcopy(docs[c]))
        e["_allowed"] = frozenset(f["name"] for f in e["function"])   # every documented function, held-out ones included
        if "missed_function" in e:
            for t, names in list(e["missed_function"].items()):
                e["missed_function"][t] = []
                for n in names:
                    for i, f in enumerate(e["function"]):
                        if f["name"] == n:
                            e["missed_function"][t].append(f)
                            e["function"].pop(i)
                            break
        e["_gold"] = gold[e["id"]]
    return entries


def flatten_calls(calls: list[dict]) -> list[tuple[str, dict]]:
    """`[{name: args}, ...]` -> `[(name, args), ...]` in reply order."""
    return [(name, args or {}) for c in calls for name, args in c.items()]


def safe_call(name: str, args: dict, allowed) -> bool:
    """Whether `name(args)` may be handed to upstream's `eval`-based executor: a documented function name and
    identifier argument names only (the values are rendered as literals by `calls_to_strings`)."""
    return name in allowed and all(isinstance(k, str) and k.isidentifier() for k in args)


def refused_result(name: str) -> str:
    """Upstream's execution error for a function that does not exist."""
    return f"Error during execution: name '{name}' is not defined"


def materialise(msgs: list, form: str) -> list:
    """Internal conversation -> the message list handed to the chat template, in one of the supported forms."""
    if form == "structured":
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]
    out: list = []
    pending: list = []

    def flush():
        if pending:
            out.append({"role": "user", "content": repr([{"role": "tool", "name": p["name"], "content": p["content"]} for p in pending])})
            pending.clear()

    for m in msgs:
        if m["role"] == "assistant":
            flush()
            out.append({"role": "assistant", "content": m.get("_text", m.get("content") or "")})
        elif m["role"] == "tool":
            if form == "text":
                out.append({"role": "tool", "name": m["name"], "content": m["content"]})
            else:
                pending.append(m)
        else:
            flush()
            out.append({k: v for k, v in m.items() if not k.startswith("_")})
    flush()
    return out


class Entry:
    def __init__(self, scorer, e: dict, cat: str, fmt: str):
        self.e, self.cat, self.fmt = e, cat, fmt
        self.category = f"multi_turn_{cat}"
        self.funcs = prepare_functions(scorer, e["function"], self.category) if fmt == "python" else list(e["function"])
        self.msgs: list = []
        self.turn, self.step, self.n_calls = -1, 0, 0
        self.result: list = []           # per turn: list of per-step call-string lists (the checker's input)
        self.steps: list = []            # per step: what the model produced (the samples record)
        self.done, self.fail = False, ""
        self.scorer = scorer
        self.next_turn()

    def next_turn(self) -> None:
        self.turn += 1
        if self.turn >= len(self.e["question"]):
            self.done = True
            return
        q = copy.deepcopy(self.e["question"][self.turn])
        hold = self.e.get("missed_function", {}).get(str(self.turn))
        if hold is not None:
            if q:
                raise RuntimeError(f"{self.e['id']}: a hold-out turn carries a user message")
            dp = self.scorer.default_prompts
            if self.fmt == "python":
                q = [{"role": "user", "content": dp.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=hold)}]
                self.funcs.extend(prepare_functions(self.scorer, hold, self.category))
            else:
                q = [{"role": "user", "content": dp.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
                self.funcs.extend(hold)
        if self.turn == 0 and self.fmt == "python":
            q = python_messages(self.scorer, q, self.funcs, self.category)
        self.msgs.extend(q)
        self.result.append([])
        self.step = 0


def render_entry(engine: Engine, scorer, x: Entry, forms: tuple, preferred: str) -> tuple[str, str]:
    """Render the entry's conversation, trying the preferred message form first; returns (prompt, form used)."""
    tools = (functions_to_tools(scorer, x.funcs) or None) if engine.format == "json" else None
    order = [preferred] + [f for f in forms if f != preferred]
    err: Exception | None = None
    for form in order:
        try:
            prompt, _ = engine.render(materialise(x.msgs, form), tools)
            return prompt, form
        except RuntimeError as e:
            err = e
    raise RuntimeError(f"{x.e['id']}: no message form renders through the chat template: {err}") from err


def run(engine: Engine, data_dir: str, out_path: str, limit: int | None = None) -> tuple[int, dict]:
    scorer = load_scorer(data_dir)
    mt_utils, mt_checker = scorer.multi_turn_utils, scorer.multi_turn_checker.multi_turn_checker
    max_steps = scorer.default_prompts.MAXIMUM_STEP_LIMIT

    def reset_instances() -> None:  # the executor keeps API instances in its module globals, keyed by entry id
        for k in [k for k in vars(mt_utils) if k.endswith("_instance")]:
            delattr(mt_utils, k)

    reset_instances()
    t0 = time.time()
    ents: list[Entry] = []
    for cat, n_expected in MULTI_TURN:
        es = load_entries(scorer, data_dir, cat)
        if len(es) != n_expected:
            raise RuntimeError(f"BFCL multi_turn_{cat}: expected {n_expected} entries, found {len(es)}")
        if limit:
            es = spread(es, limit)
        ents += [Entry(scorer, e, cat, engine.format) for e in es]
    forms = FORMS_JSON if engine.format == "json" else FORMS_PYTHON
    form = forms[0]
    gated = False
    rnd = 0
    while True:
        active = [x for x in ents if not x.done]
        if not active:
            break
        ids_list, keep = [], []
        for x in active:
            prompt, form = render_entry(engine, scorer, x, forms, form)
            if not gated:
                check_prompt_carries_tools(prompt, x.funcs, engine.format)
                gated = True
            ids = engine.encode(prompt)
            if not engine.fits(ids, MAX_NEW):
                x.done, x.fail = True, "context_overflow"
                x.steps.append({"turn": x.turn, "step": x.step, "n_prompt_toks": len(ids), "output": None, "why": "context_overflow"})
                continue
            ids_list.append(ids)
            keep.append((x, prompt, form))
        if not keep:
            continue
        gens = engine.generate(ids_list, MAX_NEW)
        for (x, prompt, used_form), g in zip(keep, gens):
            calls, derr = None, None
            try:
                calls = engine.parse_calls(scorer, g.text, x.funcs)
                if is_empty_calls(scorer, calls):
                    calls, derr = None, "empty call list"
            except ValueError as ex:
                derr = str(ex)
            step = {"turn": x.turn, "step": x.step, "form": used_form, "output": g.raw, "finish": g.finish, "n_gen_toks": g.n_tokens,
                    "prompt": prompt if (x.turn == 0 and x.step == 0) else None}
            if calls is None:                                   # no call, or unparseable: the turn ends (upstream: break)
                x.msgs.append({"role": "assistant", "content": g.text, "_text": g.text})
                step["calls"] = None
                step["why"] = derr
                x.steps.append(step)
                x.next_turn()
                continue
            flat = flatten_calls(calls)
            strs = [calls_to_strings([{name: args}])[0] for name, args in flat]
            safe = [safe_call(name, args, x.e["_allowed"]) for name, args in flat]
            exec_strs = [s for s, ok in zip(strs, safe) if ok]       # only these reach upstream's executor and checker
            x.result[-1].append(exec_strs)
            exec_results: list = []
            if exec_strs:
                exec_results, _ = mt_utils.execute_multi_turn_func_call(exec_strs, x.e["initial_config"], x.e["involved_classes"], MODEL_NAME,
                                                                        x.e["id"], long_context=("long_context" in x.cat), is_evaL_run=False)
            it = iter(exec_results)
            results = [str(next(it)) if ok else refused_result(name) for (name, _), ok in zip(flat, safe)]
            tool_calls = []
            for name, args in flat:
                x.n_calls += 1
                tool_calls.append({"id": f"{x.n_calls:09d}", "type": "function", "function": {"name": name, "arguments": args}})
            content = engine.strip(TOOL_CALL_RE.sub("", g.text)) if engine.format == "json" else g.text
            x.msgs.append({"role": "assistant", "content": content, "tool_calls": tool_calls, "_text": g.text})
            for tc, r in zip(tool_calls, results):
                x.msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": r})
            step["calls"] = strs
            step["results"] = results
            if not all(safe):
                step["refused"] = [s for s, ok in zip(strs, safe) if not ok]
            x.steps.append(step)
            x.step += 1
            if x.step > max_steps:
                x.done, x.fail = True, "force_terminated"
        rnd += 1
        print(f"[bfcl_multi_turn/{engine.format}] round {rnd}: active {len(keep)}, done {sum(x.done for x in ents)}/{len(ents)}  "
              f"{time.time() - t0:.0f}s", flush=True)

    per: dict[str, dict] = {c: {"n": 0, "correct": 0, "overflow": 0, "errors": {}} for c, _ in MULTI_TURN}
    with open(out_path, "w", encoding="utf-8") as fo:
        for x in ents:
            ok, err = False, x.fail
            if not x.fail and len(x.result) == len(x.e["_gold"]):
                entry = {k: v for k, v in x.e.items() if k not in ("function", "_gold")}
                try:
                    r = mt_checker(x.result, x.e["_gold"], entry, x.category, MODEL_NAME)
                    ok = bool(r.get("valid"))
                    err = "" if ok else (r.get("error_type") or "invalid")
                except Exception as ex:  # noqa: BLE001 -- recorded as a failed entry
                    err = f"checker_raised: {type(ex).__name__}: {ex}"
            elif not x.fail:
                err = "turn_count_mismatch"
            p = per[x.cat]
            p["n"] += 1
            p["correct"] += ok
            p["overflow"] += x.fail == "context_overflow"
            if err:
                p["errors"][err] = p["errors"].get(err, 0) + 1
            write_record(fo, {"task": "bfcl_multi_turn", "category": x.cat, "id": x.e["id"], "correct": ok, "why": err, "parsed": x.result,
                              "gold": x.e["_gold"], "steps": x.steps, "messages": None if ok else materialise(x.msgs, "structured")})
    reset_instances()
    for c, p in per.items():
        print(f"[bfcl_multi_turn/{engine.format}] {c:14s} n={p['n']:4d} acc={safe_div(p['correct'], p['n']):.4f} "
              f"errors={json.dumps(p['errors'])[:200]}", flush=True)
    metrics = {c: safe_div(p["correct"], p["n"]) for c, p in per.items()}
    metrics["overall"] = sum(metrics[c] for c, _ in MULTI_TURN) / len(MULTI_TURN)
    metrics["context_overflow"] = safe_div(sum(p["overflow"] for p in per.values()), sum(p["n"] for p in per.values()))
    return len(ents), {k: clip01(v) for k, v in metrics.items()}
