# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests that need neither a GPU nor the network."""
import json
import os

import pytest

from brahmai_eval import data, suites
from brahmai_eval.model import model_name
from brahmai_eval.sharding import lpt_partition
from brahmai_eval.summarize import primary_metric, summarize


def test_suites_resolve_and_dedupe():
    s = suites.all_suites()
    assert "all" in s and "math" in s and "english" in s
    labels = [r["label"] for r in s["all"]]
    assert len(labels) == len(set(labels))
    rows = suites.resolve("math,gsm8k,english")
    assert [r["label"] for r in rows].count("gsm8k") == 1
    with pytest.raises(SystemExit):
        suites.resolve("no_such_suite")


def test_every_row_has_the_schema():
    for r in suites.all_suites()["all"]:
        assert set(r) == {"label", "task", "num_fewshot", "chat", "gen_kwargs", "cost_secs", "runner", "unsafe_code", "limit"}
        assert r["runner"] in ("lmeval", "tooluse") and r["cost_secs"] > 0


def test_lpt_partition_balances_and_covers():
    tasks = [{"label": f"t{i}", "cost_secs": c} for i, c in enumerate([100, 90, 50, 40, 10, 10, 5])]
    parts = lpt_partition(tasks, 3)
    assert sorted(l for p in parts for l in p) == sorted(t["label"] for t in tasks)
    loads = [sum(t["cost_secs"] for t in tasks if t["label"] in p) for p in parts]
    assert max(loads) - min(loads) <= 100


def test_model_name_is_filesystem_safe():
    assert model_name("Qwen/Qwen2.5-1.5B-Instruct") == "Qwen__Qwen2.5-1.5B-Instruct"
    assert "/" not in model_name("/tmp/some ckpt/dir/")


def test_render_replaces_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAHMAI_EVAL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", str(tmp_path / "d"))
    src = tmp_path / "tasks" / "fam"
    src.mkdir(parents=True)
    (src / "x.yaml").write_text("task: x\ndataset_kwargs:\n  data_files: __DATA_DIR__/fam/x.json\n")
    monkeypatch.setattr(data, "TASKS_SRC", str(tmp_path / "tasks"))
    out = data.render_tasks(str(tmp_path / "rendered"))
    txt = open(os.path.join(out, "fam", "x.yaml")).read()
    assert "__DATA_DIR__" not in txt and str(tmp_path / "d") in txt
    assert data.include_paths(out) == [os.path.join(out, "fam")]


def test_primary_metric_preference():
    assert primary_metric({"acc,none": 0.5, "acc_norm,none": 0.6}) == ("acc_norm,none", 0.6)
    assert primary_metric({"exact_match,strict-match": 0.3, "exact_match,flexible-extract": 0.4}) == ("exact_match,strict-match", 0.3)
    assert primary_metric({"secs": 3}) is None


def test_summarize_merges_shards(tmp_path):
    run = tmp_path / "run"
    (run / "shards" / "0").mkdir(parents=True); (run / "shards" / "1").mkdir(parents=True)
    tasks = [{"label": "gsm8k", "task": "gsm8k", "num_fewshot": 5, "chat": True, "gen_kwargs": None, "cost_secs": 1, "runner": "lmeval", "unsafe_code": False, "limit": None},
             {"label": "piqa", "task": "piqa", "num_fewshot": 5, "chat": False, "gen_kwargs": None, "cost_secs": 1, "runner": "lmeval", "unsafe_code": False, "limit": None}]
    json.dump({"model": "m", "suite": "x", "tasks": tasks, "shards": [["gsm8k"], ["piqa"]], "backend": {"max_model_len": 4096, "seed": 1}, "config_hash": "h"},
              open(run / "plan.json", "w"))
    (run / "shards" / "0" / "results.jsonl").write_text(json.dumps({"label": "gsm8k", "task": "gsm8k", "n": 10, "exact_match,strict-match": 0.5, "secs": 1}) + "\n")
    (run / "shards" / "1" / "results.jsonl").write_text(json.dumps({"label": "piqa", "task": "piqa", "error": "boom"}) + "\n")
    res = summarize(str(run))
    assert res["tasks"]["gsm8k"]["primary"] == {"metric": "exact_match,strict-match", "value": 0.5}
    assert res["errors"] == ["piqa"] and res["tasks"]["piqa"]["status"] == "error"
    assert (run / "results.md").exists() and "gsm8k" in (run / "results.md").read_text()


def test_task_tables_are_consistent():
    """Every shipped family table has the fields the suites need, unique labels across families, pinned sources."""
    import glob
    seen = {}
    for p in sorted(glob.glob(os.path.join(data.TASKS_SRC, "*", "TASKS.json"))):
        fam = os.path.basename(os.path.dirname(p))
        for r in json.load(open(p)):
            assert {"label", "task", "num_fewshot", "chat", "cost_secs", "source"} <= set(r), (fam, r.get("label"))
            assert r["label"] not in seen, f"label {r['label']} in both {seen.get(r['label'])} and {fam}"
            seen[r["label"]] = fam
            assert "@" in r["source"], (fam, r["label"], "source must be pinned as <repo>@<sha>")


def test_scheduler_retries_failed_shard_on_exclusive_gpu(tmp_path, monkeypatch):
    """A worker that dies on try 1 is retried at the next fallback level on a GPU of its own; finished shards are not repeated."""
    import types

    from brahmai_eval import sharding

    out = tmp_path / "out"; (out / "shards").mkdir(parents=True)
    launches = []

    class FakeProc:
        def __init__(self, cmd, env, **kw):
            k, lvl = int(cmd[-2]), int(cmd[-1])
            launches.append((k, lvl, env["CUDA_VISIBLE_DEVICES"]))
            sd = out / "shards" / str(k); sd.mkdir(exist_ok=True)
            (sd / "engine.json").write_text("{}")
            ok = not (k == 1 and lvl == 1)              # shard 1 dies on its first try
            if ok:
                (sd / "DONE").write_text("x")
            self.rc = 0 if ok else 1
        def poll(self):
            return self.rc

    monkeypatch.setattr(sharding.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(sharding.time, "sleep", lambda s: None)
    monkeypatch.setattr(sharding, "summarize", lambda d: {"errors": []})
    monkeypatch.setattr(sharding, "visible_gpus", lambda spec: ["0", "1"])
    monkeypatch.setattr(sharding.shutil, "which", lambda c: "/usr/bin/cc")
    monkeypatch.setattr(sharding, "resolve_model", lambda m, r: str(tmp_path))
    monkeypatch.setattr(sharding, "apply_overrides", lambda d, c, s: (d, {}))
    monkeypatch.setattr(sharding, "prepare", lambda: None)
    monkeypatch.setattr(sharding, "render_tasks", lambda d: d)
    monkeypatch.setattr(sharding, "checksums", lambda d: {})
    monkeypatch.setattr(sharding, "resolve", lambda spec: [{"label": f"t{i}", "task": f"t{i}", "num_fewshot": 0, "chat": False, "gen_kwargs": None,
                                                           "cost_secs": 10, "runner": "lmeval", "unsafe_code": False, "limit": None} for i in range(4)])
    monkeypatch.setattr(sharding.M, "environment", lambda: {})
    a = types.SimpleNamespace(gpus=None, tensor_parallel=1, jobs_per_gpu=2, gpu_memory_utilization=None, suite="x", no_code_execution=False,
                              model="m", revision=None, chat_template=None, stop_tokens=None, out=str(out.parent), resume=str(out), limit=None,
                              system_instruction=None, max_model_len=4096, seed=1, dtype="bfloat16", trust_remote_code=False, add_bos=False,
                              max_num_seqs=None, tool_format="auto", serial=False)
    assert sharding.run(a) == 0
    tries = [(k, lvl) for k, lvl, g in launches]
    assert tries.count((1, 1)) == 1 and (1, 2) in tries, launches
    assert all(tries.count((k, 1)) == 1 for k in (0, 2, 3))
    assert len([x for x in launches if x[0] == 1]) == 2
