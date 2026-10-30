# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""brahmai-eval: evaluate a Hugging Face checkpoint on open benchmarks with vLLM.

  brahmai-eval run --model <hf repo id | local dir> [--suite all] [--gpus 0,1,2,3] [--out results]
  brahmai-eval list                      suites and tasks
  brahmai-eval prepare                   download every dataset up front (optional; `run` does it on demand)
  brahmai-eval summarize <run dir>       rebuild results.json / results.md from the shard records
"""
from __future__ import annotations

import argparse
import os
import sys


def cmd_run(a) -> int:
    from .sharding import run
    return run(a)


def cmd_list(a) -> int:
    from .suites import all_suites
    suites = all_suites()
    for name, rows in suites.items():
        if name == "all":
            continue
        print(f"{name}  ({len(rows)} tasks, {'chat' if all(r['chat'] for r in rows) else 'raw' if not any(r['chat'] for r in rows) else 'mixed'})")
        if a.verbose:
            for r in rows:
                print(f"   {r['label']:34s} {r['task']:34s} {r['num_fewshot']}-shot {'chat' if r['chat'] else 'raw '} {r['runner']}")
    print(f"all  ({len(suites['all'])} tasks)")
    return 0


def cmd_prepare(a) -> int:
    from .data import home, include_paths, prepare, render_tasks
    prepare()
    rendered = render_tasks(os.path.join(home(), "tasks-rendered"))
    if a.hub:
        from lm_eval.tasks import TaskManager, get_task_dict

        from .suites import resolve
        tm = TaskManager(include_path=include_paths(rendered))
        names = sorted({t["task"] for t in resolve(a.suite) if t["runner"] == "lmeval"})
        print(f"[prepare] materialising {len(names)} harness task(s) from the Hub", flush=True)
        for n in names:
            d = get_task_dict([n], tm)
            for k, v in d.items():
                print(f"[prepare] {k}: {len(list(v.eval_docs)) if hasattr(v, 'eval_docs') else 'group'} docs", flush=True)
    return 0


def cmd_summarize(a) -> int:
    from .summarize import print_summary, summarize
    summarize(a.run_dir); print_summary(a.run_dir); return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="brahmai-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate a checkpoint")
    r.add_argument("--model", required=True, help="Hugging Face repo id (org/name) or a local checkpoint directory")
    r.add_argument("--revision", default=None, help="Hub revision for --model")
    r.add_argument("--suite", default="all", help="suite names and/or task labels, comma separated (default: all)")
    r.add_argument("--gpus", default=None, help="GPU indices, comma separated (default: every visible GPU)")
    r.add_argument("--jobs-per-gpu", type=int, default=1, help="engines per GPU; 2 is faster for models under ~4B parameters")
    r.add_argument("--tensor-parallel", type=int, default=1, help="GPUs per engine for large models")
    r.add_argument("--out", default="results", help="results root (default ./results)")
    r.add_argument("--limit", type=int, default=None, help="items per task: a smoke test, never a reportable number")
    r.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue a run directory (unfinished tasks only)")
    r.add_argument("--max-model-len", type=int, default=8192, help="vLLM context window (prompts longer than this minus the generation budget are left-truncated)")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--gpu-memory-utilization", type=float, default=None, help="default 0.88 / jobs-per-gpu")
    r.add_argument("--max-num-seqs", type=int, default=None)
    r.add_argument("--trust-remote-code", action="store_true")
    r.add_argument("--chat-template", default=None, metavar="FILE", help="Jinja chat template overriding the tokenizer's")
    r.add_argument("--stop-tokens", default=None, help="extra end-of-generation tokens, comma separated (e.g. '<|im_end|>')")
    r.add_argument("--add-bos", action="store_true", help="prefix the BOS token to every raw (non-chat) prompt")
    r.add_argument("--system-instruction", default=None, help="system prompt for chat tasks (default none)")
    r.add_argument("--tool-format", default="auto", choices=["auto", "json", "python"],
                   help="how the model writes function calls in the tool-use suite (auto = detect from the chat template)")
    r.add_argument("--no-code-execution", action="store_true", help="skip tasks that execute model-generated code")
    r.add_argument("--serial", action="store_true", help="one engine on one GPU, tasks one after another (slowest, most robust)")
    r.set_defaults(f=cmd_run)

    s = sub.add_parser("list", help="suites and tasks"); s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(f=cmd_list)
    s = sub.add_parser("prepare", help="download datasets"); s.add_argument("--suite", default="all")
    s.add_argument("--hub", action="store_true", help="also materialise the Hub-hosted harness tasks"); s.set_defaults(f=cmd_prepare)
    s = sub.add_parser("summarize", help="rebuild results of a run directory"); s.add_argument("run_dir"); s.set_defaults(f=cmd_summarize)

    a = ap.parse_args(argv)
    try:
        return a.f(a)
    except FileNotFoundError as e:
        print(f"MISSING: {e}", file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
