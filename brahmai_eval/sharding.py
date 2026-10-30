# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Plan a run, split the tasks over the GPUs, launch one worker per GPU slot, retry failed shards, summarise."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from . import manifest as M
from .data import nltk_dir, prepare, render_tasks
from .model import apply_overrides, checksums, model_name, resolve_model
from .suites import resolve
from .summarize import summarize

TOOLUSE_MIN_CONTEXT = 16384
MAX_TRIES = 3
# Fallback ladder.  Try 1 runs the planned layout (possibly several engines per GPU).  A shard whose worker dies
# (engine failed to start, CUDA out of memory, a killed process) is retried on an EXCLUSIVE GPU with less concurrency,
# then once more with the most conservative engine settings.  Finished tasks are never repeated: the worker skips
# every task already scored in its results.jsonl.
LADDER = {1: {}, 2: {"max_num_seqs": 32, "gpu_memory_utilization": 0.85}, 3: {"max_num_seqs": 4, "gpu_memory_utilization": 0.85, "enforce_eager": True}}


def lpt_partition(tasks: list[dict], n: int) -> list[list[str]]:
    """Longest-processing-time-first: the heaviest task goes to the least loaded shard."""
    bins: list[list[str]] = [[] for _ in range(n)]
    load = [0] * n
    for t in sorted(tasks, key=lambda t: -t["cost_secs"]):
        i = min(range(n), key=lambda k: load[k])
        bins[i].append(t["label"]); load[i] += t["cost_secs"]
    return bins


def visible_gpus(spec: str | None) -> list[str]:
    if spec:
        return [g.strip() for g in spec.split(",") if g.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return [l.strip() for l in q.splitlines() if l.strip()]
    except Exception:
        return []


def run(a) -> int:
    t0 = time.time()
    gpus = visible_gpus(a.gpus)
    if not gpus:
        raise SystemExit("no GPU visible (nvidia-smi found none); pass --gpus 0,1,...")
    if not any(shutil.which(c) for c in ("cc", "gcc", "clang")):
        raise SystemExit("no C compiler on PATH: vLLM compiles its kernels with torch.compile/Triton and needs one (Debian/Ubuntu: "
                         "`apt install build-essential`; or run inside the Docker image, which has it).")
    tp = max(1, a.tensor_parallel)
    if len(gpus) % tp:
        raise SystemExit(f"{len(gpus)} GPU(s) cannot be split into tensor-parallel groups of {tp}")
    groups = [gpus[i:i + tp] for i in range(0, len(gpus), tp)]
    if a.serial:
        groups = groups[:1]; a.jobs_per_gpu = 1
    slots = [g for g in groups for _ in range(a.jobs_per_gpu)]
    mem = a.gpu_memory_utilization or round(0.88 / a.jobs_per_gpu, 2)

    tasks = resolve(a.suite)
    if a.no_code_execution:
        tasks = [t for t in tasks if not t["unsafe_code"]]
    if not tasks:
        raise SystemExit("no tasks selected")
    model_dir = resolve_model(a.model, a.revision)
    model_dir, overrides = apply_overrides(model_dir, a.chat_template, [s for s in (a.stop_tokens or "").split(",") if s])
    prepare()                                   # non-Hub datasets (idempotent)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = a.resume or os.path.join(a.out, model_name(a.model), stamp)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    tasks_dir = render_tasks(os.path.join(out_dir, "tasks"))     # the exact task definitions this run used
    n_shards = min(len(slots), len(tasks))
    plan_p = os.path.join(out_dir, "plan.json")
    if a.resume and os.path.exists(plan_p):
        plan = json.load(open(plan_p))
        n_shards = len(plan["shards"])
    else:
        shards = lpt_partition(tasks, n_shards)
        # the function-calling benchmarks carry long tool declarations and multi-turn histories: a shard that holds one
        # gets a window of at least 16384 tokens so no item is dropped as context_overflow
        need = {t["label"]: TOOLUSE_MIN_CONTEXT for t in tasks if t["runner"] == "tooluse"}
        shard_ctx = [max([a.max_model_len] + [need[l] for l in sh if l in need]) for sh in shards]
        plan = {"model": a.model, "model_dir": model_dir, "model_checksums": checksums(model_dir), "overrides": overrides, "suite": a.suite,
                "tasks": tasks, "shards": shards, "shard_max_model_len": shard_ctx, "tasks_dir": tasks_dir, "out_dir": out_dir, "limit": a.limit,
                "system_instruction": a.system_instruction,
                "backend": {"max_model_len": a.max_model_len, "gpu_memory_utilization": mem, "tensor_parallel_size": tp, "seed": a.seed,
                            "dtype": a.dtype, "trust_remote_code": a.trust_remote_code, "add_bos": a.add_bos, "max_num_seqs": a.max_num_seqs,
                            "tool_format": a.tool_format},
                "command": " ".join(sys.argv), "started": M.now(), "environment": M.environment()}
        plan["config_hash"] = M.config_hash({k: plan[k] for k in ("model_checksums", "overrides", "tasks", "backend", "limit", "system_instruction")})
        M.write_json(plan_p, plan)
    print(f"[run] {len(tasks)} task(s) in {n_shards} shard(s) on GPU slots {slots[:n_shards]} -> {out_dir}", flush=True)
    if any(t["unsafe_code"] for t in tasks):
        print("[run] the code suite executes model-generated programs on this machine (HumanEval/MBPP protocol); "
              "use --no-code-execution to skip it", flush=True)

    pending = list(range(n_shards))
    for k in pending:                      # a DONE marker from an earlier attempt must not hide an unfinished retry
        m = os.path.join(out_dir, "shards", str(k), "DONE")
        if os.path.exists(m):
            os.remove(m)
    tries = {k: 0 for k in pending}
    running: dict[int, tuple[subprocess.Popen, list[str], float, int]] = {}
    free_slots = [slots[k] for k in range(min(n_shards, len(slots)))]   # a resumed plan may hold more shards than this machine has slots
    failed: list[int] = []
    key = lambda g: ",".join(g)
    while pending or running:
        for k, (p, g, ts, lvl) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[k]; free_slots.append(g)
            done = os.path.exists(os.path.join(out_dir, "shards", str(k), "DONE"))
            print(f"[run] shard {k} on GPU {key(g)} exited rc={rc} after {(time.time() - ts) / 60:.1f} min" + ("" if done else " (incomplete)"), flush=True)
            if not done and tries[k] < MAX_TRIES:
                pending.append(k)
                print(f"[run] shard {k} will be retried at fallback level {tries[k] + 1} on an exclusive GPU ({LADDER[tries[k] + 1]})", flush=True)
            elif not done:
                failed.append(k)
        # vLLM sizes its KV cache from the memory free at start-up, so two engines sharing a GPU must start one after the
        # other (a GPU whose engine has not written engine.json yet is 'loading'); a retried shard needs the whole GPU.
        loading = {key(g) for k, (p, g, ts, lvl) in running.items() if not os.path.exists(os.path.join(out_dir, "shards", str(k), "engine.json"))}
        busy = {key(g) for k, (p, g, ts, lvl) in running.items()}
        exclusive = {key(g) for k, (p, g, ts, lvl) in running.items() if lvl > 1}
        for k in list(pending):
            lvl = tries[k] + 1
            cands = [g for g in free_slots if key(g) not in loading and key(g) not in exclusive and (lvl == 1 or key(g) not in busy)]
            if not cands:
                continue
            g = cands[0]; free_slots.remove(g); pending.remove(k); tries[k] = lvl
            loading.add(key(g)); busy.add(key(g))
            if lvl > 1:
                exclusive.add(key(g))
            sd = os.path.join(out_dir, "shards", str(k)); os.makedirs(sd, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=key(g), TOKENIZERS_PARALLELISM="false", NLTK_DATA=nltk_dir(),
                       VLLM_LOGGING_LEVEL=os.environ.get("VLLM_LOGGING_LEVEL", "WARNING"))
            log = open(os.path.join(sd, "log.txt"), "a")
            log.write(f"\n===== try {lvl} on GPU {key(g)} at {M.now()} =====\n"); log.flush()
            p = subprocess.Popen([sys.executable, "-m", "brahmai_eval.runner", plan_p, str(k), str(lvl)], env=env, stdout=log, stderr=subprocess.STDOUT)
            running[k] = (p, g, time.time(), lvl)
            print(f"[run] shard {k} ({len(plan['shards'][k])} tasks) -> GPU {key(g)} (try {lvl})", flush=True)
        time.sleep(5)
    res = summarize(out_dir)
    status = "finished" if not failed and not res["errors"] else "finished_with_errors"
    M.write_json(os.path.join(out_dir, "manifest.json"), {"status": status, "finished": M.now(), "duration_secs": round(time.time() - t0, 1),
                                                          "failed_shards": failed, "task_errors": res["errors"], "config_hash": plan["config_hash"],
                                                          "command": plan["command"], "environment": plan["environment"]})
    print(f"\n[run] {status} in {(time.time() - t0) / 60:.1f} min -> {out_dir}/results.md", flush=True)
    if res["errors"]:
        print(f"[run] task errors: {res['errors']}  (re-run with --resume {out_dir} after fixing)", flush=True)
    return 1 if (failed or res["errors"]) else 0
