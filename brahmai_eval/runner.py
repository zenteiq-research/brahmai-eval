# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""One worker = one vLLM engine on one GPU (or one tensor-parallel group) running its share of the tasks.

Invoked by `brahmai_eval.sharding` as a subprocess:  python -m brahmai_eval.runner <plan.json> <shard index>
(vLLM starts its engine with multiprocessing 'spawn', so this module must only run under a __main__ guard).
Outputs, under <out_dir>/shards/<k>/:  results.jsonl (one record per task), samples/<task>.jsonl (every scored item),
log.txt (captured by the parent), DONE when every task of the shard was attempted.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

from . import manifest as M
from .data import include_paths


def preflight(lm, needs_chat: bool, max_new: int = 64) -> dict:
    """Prove the checkpoint can be prompted the way the suite will prompt it."""
    tok = lm.tokenizer
    info: dict[str, Any] = {"stop_token_ids": lm.stop_token_ids}
    if not lm.stop_token_ids:
        raise SystemExit("ABORT: the checkpoint declares no eos_token_id (config.json / generation_config.json / tokenizer); "
                         "generation would never stop.  Pass --stop-tokens <token> to declare the end-of-turn token.")
    if not needs_chat:
        return info
    if not getattr(tok, "chat_template", None):
        raise SystemExit("ABORT: the tokenizer has no chat template, but the selected tasks prompt through one.  Pass "
                         "--chat-template <file.jinja>, or run only raw suites (see `brahmai-eval list`).")
    prompt = tok.apply_chat_template([{"role": "user", "content": "What is 2+2? Answer briefly."}], tokenize=False, add_generation_prompt=True)
    text, n = lm.probe_generate(prompt, max_new)
    if n >= max_new:
        raise SystemExit(f"ABORT: a greedy probe reply ran the full {max_new} tokens without hitting a stop token "
                         f"(stop ids {lm.stop_token_ids}).  The chat template's end-of-turn token is probably not an eos id: "
                         f"pass --stop-tokens <token>.  Reply tail: {text[-160:]!r}")
    info.update({"probe_prompt_tail": prompt[-60:], "probe_reply": text[:200], "probe_tokens": n})
    print(f"[preflight] ok: reply stopped after {n} tokens; stop ids {lm.stop_token_ids}", flush=True)
    return info


def build_lm(plan: dict, shard: int = 0, level: int = 1):
    from .backend import EvalVLLM, tokenizer_parity_check
    from .sharding import LADDER
    b = plan["backend"]
    ctx = (plan.get("shard_max_model_len") or [b["max_model_len"]] * len(plan["shards"]))[shard]
    kw = {"max_length": ctx, "gpu_memory_utilization": b["gpu_memory_utilization"], "tensor_parallel_size": b["tensor_parallel_size"],
          "seed": b["seed"], "dtype": b["dtype"], "batch_size": "auto", "enable_prefix_caching": True, "trust_remote_code": b["trust_remote_code"]}
    if b.get("max_num_seqs"):
        kw["max_num_seqs"] = b["max_num_seqs"]
    kw.update(LADDER.get(level, {}))                      # conservative engine settings on a retried shard
    if level > 1:
        print(f"[shard {shard}] fallback level {level}: engine settings {LADDER[level]}", flush=True)
    lm = EvalVLLM(plan["model_dir"], add_bos_prefix=b["add_bos"], **kw)
    parity = tokenizer_parity_check(lm, plan["model_dir"])
    return lm, parity


def _metrics(res: dict) -> dict:
    return {k: v for k, v in res.items() if "," in k and "stderr" not in k}


FATAL = ("out of memory", "enginedeaderror", "engine core", "cuda error", "nccl", "killed", "engine is dead", "asyncenginedeaderror")


def _fatal(err: str) -> bool:
    """An engine-level failure: nothing further can be scored in this process, the parent retries the shard."""
    e = err.lower()
    return any(f in e for f in FATAL)


def run_shard(plan: dict, shard: int, level: int = 1) -> int:
    import lm_eval
    from lm_eval.tasks import TaskManager

    tasks = [t for t in plan["tasks"] if t["label"] in set(plan["shards"][shard])]
    sd = os.path.join(plan["out_dir"], "shards", str(shard))
    os.makedirs(os.path.join(sd, "samples"), exist_ok=True)
    if os.path.exists(os.path.join(sd, "DONE")):
        os.remove(os.path.join(sd, "DONE"))
    done_labels = set()
    rp = os.path.join(sd, "results.jsonl")
    if os.path.exists(rp):
        for l in open(rp):
            r = json.loads(l)
            if "error" not in r:
                done_labels.add(r["label"])
    todo = [t for t in tasks if t["label"] not in done_labels]
    print(f"[shard {shard}] {len(tasks)} task(s), {len(done_labels)} already scored: {[t['label'] for t in todo]}", flush=True)
    if not todo:
        open(os.path.join(sd, "DONE"), "w").write(M.now()); return 0
    limit = plan.get("limit")
    if limit:
        print(f"[shard {shard}] --limit {limit}: a smoke test, not a reportable number", flush=True)
    if any(t["unsafe_code"] for t in todo):
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    lm, parity = build_lm(plan, shard, level)
    pf = preflight(lm, needs_chat=any(t["chat"] for t in todo))
    M.write_json(os.path.join(sd, "engine.json"), {"engine": lm.engine_info(), "tokenizer_parity": parity, "preflight": pf})
    tm = TaskManager(include_path=include_paths(plan["tasks_dir"]))
    seed = plan["backend"]["seed"]
    out = open(rp, "a")
    errors = 0
    for t in todo:
        label, task = t["label"], t["task"]
        t0 = time.time(); lm.reset_diag()
        res: dict[str, Any] = {}
        try:
            if t["runner"] == "tooluse":
                from . import tooluse
                r = tooluse.run_task(task, lm, plan["model_dir"], os.path.join(sd, "samples"), limit=limit, tool_format=plan["backend"]["tool_format"])
                res = {"n": r["n"], **{f"{k},none": v for k, v in r["metrics"].items()}}
            else:
                kw: dict[str, Any] = {}
                if t.get("gen_kwargs"):
                    kw["gen_kwargs"] = ",".join(f"{k}={json.dumps(v)}" for k, v in t["gen_kwargs"].items())
                if plan.get("system_instruction") and t["chat"]:
                    kw["system_instruction"] = plan["system_instruction"]
                r = lm_eval.simple_evaluate(
                    model=lm, tasks=[task], num_fewshot=t["num_fewshot"], limit=limit or t.get("limit"), batch_size="auto",
                    apply_chat_template=t["chat"], fewshot_as_multiturn=(t["chat"] and t["num_fewshot"] > 0),
                    random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
                    log_samples=True, confirm_run_unsafe_code=bool(t["unsafe_code"]), task_manager=tm, verbosity="ERROR", **kw)
                rr = r["results"][task]
                n, docs = 0, set()
                with open(os.path.join(sd, "samples", f"{task}.jsonl"), "w", encoding="utf-8") as fh:
                    for sub, recs in sorted((r.get("samples") or {}).items()):
                        for sm in recs or []:
                            fh.write(json.dumps({"task": sub, **{k: v for k, v in sm.items() if not k.endswith("_hash")}}, ensure_ascii=False, default=str) + "\n")
                            n += 1; docs.add((sub, sm.get("doc_id")))
                if not n:
                    raise RuntimeError(f"{task} produced no sample records")
                n = len(docs)                      # scored items (a task with several filters logs one record per filter)
                bad = {k: v for k, v in _metrics(rr).items() if not isinstance(v, (int, float))}
                if bad:
                    raise RuntimeError(f"non-numeric metric values: {bad}")
                res = {"n": n, **_metrics(rr), **{k: v for k, v in rr.items() if "stderr" in k}}
        except Exception as e:  # noqa: BLE001 -- recorded per task, surfaced through the exit code
            traceback.print_exc()
            res = {"error": repr(e)[:500]}; errors += 1
        rec = {"label": label, "task": task, "runner": t["runner"], "num_fewshot": t["num_fewshot"], "chat": t["chat"], "limit": limit or t.get("limit"),
               "secs": round(time.time() - t0, 1), **lm.diag(), **res}
        print("RESULT " + json.dumps(rec, default=str), flush=True)
        out.write(json.dumps(rec, default=str) + "\n"); out.flush()
        if "error" in res and _fatal(res["error"]):
            out.close()
            print(f"[shard {shard}] engine failure on {label}; stopping so the shard is retried with safer settings", flush=True)
            return 3
    out.close()
    open(os.path.join(sd, "DONE"), "w").write(M.now())
    print(f"[shard {shard}] finished: {len(todo)} task(s), {errors} error(s)", flush=True)
    return 1 if errors else 0


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    plan = json.load(open(argv[0]))
    return run_shard(plan, int(argv[1]), int(argv[2]) if len(argv) > 2 else 1)


if __name__ == "__main__":
    sys.exit(main())
