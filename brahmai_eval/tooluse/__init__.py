# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tool-use (function-calling) benchmarks, run on the harness's vLLM engine through the model's chat template.

Tasks (see each module for the protocol and the metrics):
  bfcl_single_turn  BFCL v3 single-turn Python categories, official AST checker       (brahmai_eval.datasets.bfcl)
  bfcl_multi_turn   BFCL v3 multi-turn categories, official executor + state checker (brahmai_eval.datasets.bfcl)
  when2call         When2Call test/mcq: when to call, ask, or decline                 (brahmai_eval.datasets.when2call)

`run_task(name, lm, model_dir, samples_dir, limit=None, tool_format="auto")` writes every item to
`<samples_dir>/<name>.jsonl` and returns `{"n": int, "metrics": {name: float in [0, 1]}}`; each task has an `overall`
metric.  `tool_format` selects how the model declares and writes calls ("json", "python" or "auto"; see
brahmai_eval.tooluse.common).  `limit` caps the items per category (a smoke test).
"""
from __future__ import annotations

import os

from ..datasets import bfcl as _bfcl
from ..datasets import when2call as _w2c

TASKS = {
    "bfcl_single_turn": {"cost_secs": 400, "n_docs": sum(n for _, _, n in _bfcl.SINGLE_TURN), "source": _bfcl.SOURCE, "license": _bfcl.LICENSE},
    "bfcl_multi_turn": {"cost_secs": 450, "n_docs": sum(n for _, n in _bfcl.MULTI_TURN), "source": _bfcl.SOURCE, "license": _bfcl.LICENSE},
    "when2call": {"cost_secs": 300, "n_docs": _w2c.ROWS, "source": _w2c.SOURCE, "license": _w2c.LICENSE},
}


def run_task(name: str, lm, model_dir: str, samples_dir: str, limit: int | None = None, tool_format: str = "auto") -> dict:
    if name not in TASKS:
        raise KeyError(f"unknown tool-use task {name!r}; known: {sorted(TASKS)}")
    from ..data import data_dir
    from .common import Engine
    d = data_dir()
    engine = Engine(lm, tool_format)
    print(f"[tooluse] {name}: tool_format={engine.format}{' (structured <arg_key> calls)' if engine.structured else ''}, "
          f"max_model_len={engine.max_len}, stop ids {engine.stop_ids}", flush=True)
    os.makedirs(samples_dir, exist_ok=True)
    out = os.path.join(samples_dir, f"{name}.jsonl")
    if name == "bfcl_single_turn":
        from . import bfcl_single_turn as mod
    elif name == "bfcl_multi_turn":
        from . import bfcl_multi_turn as mod
    else:
        from . import when2call as mod
    n, metrics = mod.run(engine, d, out, limit=limit)
    bad = {k: v for k, v in metrics.items() if not isinstance(v, float) or not 0.0 <= v <= 1.0}
    if bad or "overall" not in metrics:
        raise RuntimeError(f"{name}: metrics must be floats in [0, 1] and include 'overall': {bad or metrics}")
    return {"n": n, "metrics": metrics, "tool_format": engine.format}
