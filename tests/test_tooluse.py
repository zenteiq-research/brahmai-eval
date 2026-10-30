# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tool-use suite: parsers, chat rendering fallbacks and scorer wiring, without a GPU.

The tests that drive the real BFCL / When2Call data and the official scorer run only when the data has been
prepared once into `<repo>/.hfcache_public/data` (or $BRAHMAI_EVAL_DATA); they are skipped otherwise.  A stand-in
tokenizer renders a simple chat template and a stand-in engine answers with the gold calls, so every scorer path
(AST checker, multi-turn executor + state checker, When2Call log-likelihood and generation) is exercised end to end.
"""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest

from brahmai_eval.datasets import bfcl as bfcl_data
from brahmai_eval.datasets import when2call as w2c_data
from brahmai_eval.tooluse import TASKS, run_task
from brahmai_eval.tooluse import bfcl_multi_turn as mt
from brahmai_eval.tooluse import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("BRAHMAI_EVAL_DATA") or os.path.join(ROOT, ".hfcache_public", "data")
HAVE_DATA = os.path.exists(os.path.join(DATA_DIR, "bfcl", ".complete")) and os.path.exists(os.path.join(DATA_DIR, "when2call", ".complete"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="benchmark data not prepared (run brahmai_eval.datasets.{bfcl,when2call}.prepare once)")


# ── stand-ins ──────────────────────────────────────────────────────────────────────────────────────────────────────
class FakeTok:
    """A character-level tokenizer with a minimal chat template (`<|role|>content<eos>` turns, tools in a system block)."""
    SPECIALS = ["<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"]

    def __init__(self, template: str = "chat template with tools and tool_call blocks", strict: bool = False):
        self.chat_template = template
        self.strict = strict                      # rejects the system and tool roles, like some templates do
        self.all_special_tokens = list(self.SPECIALS)
        self.eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None):
        s = "<bos>"
        if tools:
            s += "<|system|><tools>" + json.dumps(tools) + "</tools><eos>\n"
        for m in messages:
            if self.strict and m["role"] in ("system", "tool"):
                raise ValueError(f"{m['role']} role not supported")
            if m["role"] == "assistant" and m.get("tool_calls"):
                calls = "".join("<tool_call>\n" + json.dumps({"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}) + "\n</tool_call>"
                                for tc in m["tool_calls"])
                s += "<|assistant|>" + (m.get("content") or "") + calls + "<eos>\n"
            else:
                s += f"<|{m['role']}|>{m.get('content') or ''}<eos>\n"
        if add_generation_prompt:
            s += "<|assistant|>"
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids)


class FakeLLM:
    """Answers each prompt through `reply(prompt_text)`; scores prompt log-probs as 0 when a preferred string is present."""

    def __init__(self, reply, preferred=(), max_model_len=10_000_000):
        self.reply, self.preferred = reply, list(preferred)
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len))
        self.calls = 0

    def generate(self, prompts, sp, use_tqdm=False):
        self.calls += 1
        outs = []
        for p in prompts:
            ids = list(p["prompt_token_ids"])
            text = FakeTok.decode(ids)
            if getattr(sp, "prompt_logprobs", None) is not None:   # preferred: (question, gold reply) pairs
                lp = 0.0 if any(q in text and g in text for q, g in self.preferred) else -1.0
                outs.append(SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=[None] + [{i: SimpleNamespace(logprob=lp)} for i in ids[1:]]))
            else:
                r = self.reply(text)
                outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=r, token_ids=list(range(len(r))), finish_reason="stop")]))
        return outs


def fake_lm(reply=lambda t: "", preferred=(), template="chat template with tools and tool_call blocks", strict=False, max_model_len=10_000_000):
    return SimpleNamespace(model=FakeLLM(reply, preferred, max_model_len), tokenizer=FakeTok(template, strict), stop_token_ids=[1], gen_lengths=[])


def last_segment(prompt: str) -> tuple[str, str]:
    """(role, content) of the last rendered turn before the generation prompt."""
    segs = re.findall(r"<\|(user|assistant|tool)\|>(.*?)<eos>", prompt, re.S)
    return segs[-1] if segs else ("", "")


def user_segments(prompt: str) -> list[str]:
    return re.findall(r"<\|user\|>(.*?)<eos>", prompt, re.S)


def gold_string_to_call(entry: dict, s: str) -> dict:
    """Upstream's gold call strings may pass positional arguments; map them onto the doc's parameter order."""
    import ast
    node = ast.parse(s, mode="eval").body
    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    docs = list(entry["function"]) + [f for fs in entry.get("missed_function", {}).values() for f in fs]
    keys = list(next(f for f in docs if f["name"] == name)["parameters"]["properties"])
    args = {keys[i]: ast.literal_eval(a) for i, a in enumerate(node.args)}
    args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    return {name: args}


def calls_as_reply(calls: list[dict], fmt: str) -> str:
    if fmt == "python":
        return "[" + ", ".join(C.calls_to_strings(calls)) + "]"
    return "".join("<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call>" for c in calls for n, a in c.items())


# ── parsers and formats ────────────────────────────────────────────────────────────────────────────────────────────
def test_parse_json_calls_block_and_bare_forms():
    assert C.parse_json_calls('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Pune"}}\n</tool_call>') == [{"get_weather": {"city": "Pune"}}]
    assert C.parse_json_calls('{"name": "f", "arguments": {"a": 1}}') == [{"f": {"a": 1}}]
    assert C.parse_json_calls('[{"name": "f", "arguments": {}}, {"name": "g", "parameters": {"x": [1, 2]}}]') == [{"f": {}}, {"g": {"x": [1, 2]}}]
    assert C.parse_json_calls('<tool_call>{"type": "function", "function": {"name": "a.b", "arguments": "{\\"k\\": true}"}}</tool_call>') == [{"a.b": {"k": True}}]
    two = C.parse_json_calls('ok <tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call> and <tool_call>{"name": "g", "arguments": {}}</tool_call>')
    assert two == [{"f": {"a": 1}}, {"g": {}}]


@pytest.mark.parametrize("text", ["I cannot help with that.", "{not json", '{"answer": 42}', "<tool_call>prose</tool_call>",
                                  '<tool_call>{"name": "f", "arguments": "oops"}</tool_call>', '<tool_call>{"arguments": {}}</tool_call>'])
def test_parse_json_calls_rejects(text):
    with pytest.raises(ValueError):
        C.parse_json_calls(text)


def test_parse_structured_calls_coerces_declared_types():
    funcs = [{"name": "book", "parameters": {"properties": {"n": {"type": "integer"}, "ok": {"type": "boolean"}, "ids": {"type": "array"},
                                                            "price": {"type": "float"}, "note": {"type": "string"}}}}]
    text = ("<tool_call>book<arg_key>n</arg_key><arg_value>3</arg_value><arg_key>ok</arg_key><arg_value>false</arg_value>"
            "<arg_key>ids</arg_key><arg_value>[1, 2]</arg_value><arg_key>price</arg_key><arg_value>9.5</arg_value>"
            "<arg_key>note</arg_key><arg_value>12</arg_value></tool_call>")
    assert C.parse_json_calls(text, funcs, structured=True) == [{"book": {"n": 3, "ok": False, "ids": [1, 2], "price": 9.5, "note": "12"}}]
    assert C.parse_json_calls("<tool_call>noargs</tool_call>", funcs, structured=True) == [{"noargs": {}}]
    with pytest.raises(ValueError):
        C.parse_json_calls("<tool_call>noargs</tool_call>", funcs, structured=False)
    assert C.coerce("maybe", "boolean") == "maybe" and C.coerce("10.0", "integer") == 10 and C.coerce("x", "integer") == "x"


def test_resolve_format_auto():
    assert C.resolve_format("auto", "{% for tool in tools %}...{% endfor %}") == "json"
    assert C.resolve_format("auto", "<tool_call>") == "json"
    assert C.resolve_format("auto", "{{ messages }}") == "python"
    assert C.resolve_format("auto", None) == "python"
    assert C.resolve_format("python", "<tool_call>") == "python"
    with pytest.raises(ValueError):
        C.resolve_format("xml", "")


def test_calls_to_strings_is_the_executable_form():
    assert C.calls_to_strings([{"cd": {"folder": "x"}}, {"ls": {}}, {"mv": {"a": [1, "b"]}}]) == ["cd(folder='x')", "ls()", "mv(a=[1, 'b'])"]


def test_engine_strip_keeps_tool_markers_and_drops_traces():
    eng = C.Engine(fake_lm(), "auto")
    assert eng.format == "json" and eng.max_len == 10_000_000 and eng.stop_ids == [1]
    assert eng.strip("<think>plan</think><|assistant|><tool_call>{}</tool_call><eos>") == "<tool_call>{}</tool_call>"


def test_render_fallbacks_for_strict_templates():
    eng = C.Engine(fake_lm(strict=True), "json")
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    text, flag = eng.render(msgs)
    assert flag == "system_folded" and "<|user|>SYS\n\na\n\nb<eos>" in text and "<|system|>" not in text
    eng = C.Engine(fake_lm(), "json")
    text, flag = eng.render([{"role": "system", "content": "SYS"}, {"role": "user", "content": "a"}])
    assert flag == "" and "<|system|>SYS<eos>" in text
    assert C.merge_consecutive([{"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}]) == [{"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}]


def test_multi_turn_message_forms():
    tc = [{"id": "1", "type": "function", "function": {"name": "ls", "arguments": {}}}]
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "_text": "<tool_call>{}</tool_call>", "tool_calls": tc},
            {"role": "tool", "tool_call_id": "1", "name": "ls", "content": "{'files': []}"}]
    s = mt.materialise(msgs, "structured")
    assert "_text" not in s[1] and s[2]["role"] == "tool"
    t = mt.materialise(msgs, "text")
    assert t[1] == {"role": "assistant", "content": "<tool_call>{}</tool_call>"} and t[2] == {"role": "tool", "name": "ls", "content": "{'files': []}"}
    u = mt.materialise(msgs, "user")
    assert u[2]["role"] == "user" and "'role': 'tool'" in u[2]["content"] and "ls" in u[2]["content"]


def test_check_prompt_carries_tools():
    C.check_prompt_carries_tools("You are an expert in composing functions ... calc(a)", [{"name": "calc"}], "python")
    with pytest.raises(RuntimeError):
        C.check_prompt_carries_tools("Find the area", [{"name": "calc"}], "json")
    with pytest.raises(RuntimeError):
        C.check_prompt_carries_tools("calc", [{"name": "calc"}], "python")


def test_tasks_table_and_suite_registration():
    assert set(TASKS) == {"bfcl_single_turn", "bfcl_multi_turn", "when2call"}
    assert TASKS["bfcl_single_turn"]["n_docs"] == 3491 and TASKS["bfcl_multi_turn"]["n_docs"] == 800 and TASKS["when2call"]["n_docs"] == 3652
    from brahmai_eval.suites import all_suites
    rows = {r["label"]: r for r in all_suites()["tool_use"]}
    assert set(rows) == set(TASKS) and all(r["runner"] == "tooluse" and r["chat"] for r in rows.values())


def test_scorer_patching_rewrites_every_upstream_import():
    src = ("from bfcl.constants.eval_config import UNDERSCORE_TO_DOT\nfrom bfcl.constants.type_mappings import (A, B)\n"
           "from bfcl.eval_checker.ast_eval.type_convertor.java_type_converter import java_type_converter\nimport re\n")
    text, applied = bfcl_data._patch("ast_checker.py", src)
    assert "UNDERSCORE_TO_DOT: list = []" in text and "from .type_mappings import (A, B)" in text and "from .type_convertor.java_type_converter import" in text
    assert len(applied) == 3
    text, _ = bfcl_data._patch("type_convertor/js_type_converter.py", "from bfcl.constants.type_mappings import JS_TYPE_CONVERSION\n")
    assert text.startswith("from ..type_mappings import")
    text, _ = bfcl_data._patch("multi_turn_utils.py", '    "MathAPI": "bfcl.eval_checker.multi_turn_eval.func_source_code.math_api",\n')
    assert '"bfcl_scorer.func_source_code.math_api"' in text
    with pytest.raises(RuntimeError):
        bfcl_data._patch("x.py", "from bfcl.something.else import y\n")
    utils_src = "import os\ndef is_function_calling_format_output(x):\n    return True\n\ndef other():\n    pass\n\ndef is_empty_output(x):\n    return False\n"
    helpers = bfcl_data._extract_helpers(utils_src)
    assert "def is_empty_output" in helpers and "def other" not in helpers and "import os" not in helpers


def test_when2call_macro_f1():
    from brahmai_eval.tooluse.when2call import macro_f1
    f1, per = macro_f1(["tool_call", "cannot_answer", "direct"], ["tool_call", "cannot_answer", "request_for_info"])
    assert per["tool_call"] == 1.0 and per["cannot_answer"] == 1.0 and per["request_for_info"] == 0.0 and abs(f1 - 2 / 3) < 1e-9


# ── wiring against the real data and the official scorer ──────────────────────────────────────────────────────────
def _pick(alternatives: list):
    """The first allowed value ('' = the parameter may be omitted); dict values hold alternatives per inner key."""
    chosen = [v for v in alternatives if v != ""]
    if not chosen:
        return None
    v = chosen[0]
    if isinstance(v, dict):
        return {k: _pick(a) for k, a in v.items() if _pick(a) is not None}
    return v


def gold_call(possible: list[dict]) -> list[dict]:
    """A call that the possible-answer entry accepts."""
    out = []
    for item in possible:
        for name, params in item.items():
            out.append({name: {k: _pick(vals) for k, vals in params.items() if _pick(vals) is not None}})
    return out


@needs_data
def test_scorer_imports_and_ast_checker_on_gold():
    scorer = bfcl_data.load_scorer(DATA_DIR)
    entries = C.load_jsonl(bfcl_data.data_path(DATA_DIR, "BFCL_v3_simple.json"))[:20]
    gold = {g["id"]: g["ground_truth"] for g in C.load_jsonl(bfcl_data.data_path(DATA_DIR, "possible_answer", "BFCL_v3_simple.json"))}
    for e in entries:
        funcs = C.prepare_functions(scorer, e["function"], "simple")
        call = gold_call(gold[e["id"]])
        r = scorer.ast_checker.ast_checker(funcs, call, gold[e["id"]], "Python", "simple", "local")
        assert r["valid"], (e["id"], r)
        text = "[" + ", ".join(C.calls_to_strings(call)) + "]"
        assert C.parse_python_calls(scorer, text) == call
        assert C.parse_json_calls(calls_as_reply(call, "json")) == call
    assert scorer.format_helpers.is_empty_output([]) and not scorer.format_helpers.is_empty_output([{"f": {}}])
    tools = C.functions_to_tools(scorer, C.prepare_functions(scorer, entries[0]["function"], "simple"))
    assert tools[0]["type"] == "function" and tools[0]["function"]["name"] == entries[0]["function"][0]["name"]
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert json.load(open(os.path.join(DATA_DIR, "bfcl", "SOURCE.json")))["source"] == bfcl_data.SOURCE
    assert "Apache License" in open(os.path.join(DATA_DIR, "bfcl", "LICENSE")).read()


@needs_data
def test_multi_turn_gold_replay_passes_the_state_checker():
    scorer = bfcl_data.load_scorer(DATA_DIR)
    for cat in ("base", "miss_func"):
        for e in C.spread(mt.load_entries(scorer, DATA_DIR, cat), 3):
            result = [[turn] if turn else [] for turn in e["_gold"]]
            entry = {k: v for k, v in e.items() if k not in ("function", "_gold")}
            r = scorer.multi_turn_checker.multi_turn_checker(result, e["_gold"], entry, f"multi_turn_{cat}", "replay_test")
            assert r["valid"], (e["id"], r)


@needs_data
@pytest.mark.parametrize("fmt", ["json", "python"])
def test_bfcl_single_turn_end_to_end_with_gold_replies(tmp_path, monkeypatch, fmt):
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", DATA_DIR)
    replies = {}
    for name, kind, _ in bfcl_data.SINGLE_TURN:
        entries = C.spread(C.load_jsonl(bfcl_data.data_path(DATA_DIR, f"BFCL_v3_{name}.json")), 3)
        gold = {g["id"]: g["ground_truth"] for g in C.load_jsonl(bfcl_data.data_path(DATA_DIR, "possible_answer", f"BFCL_v3_{name}.json"))} if kind == "ast" else {}
        for e in entries:
            q = e["question"][0][-1]["content"]
            if kind == "ast":
                replies[q] = calls_as_reply(gold_call(gold[e["id"]]), fmt)
            elif kind == "relevance":
                replies[q] = calls_as_reply([{e["function"][0]["name"]: {}}], fmt)
            else:
                replies[q] = "I cannot help with that request."
    lm = fake_lm(lambda p: replies.get(last_segment(p)[1], "I cannot help with that request."), template="x" if fmt == "python" else "tools")
    r = run_task("bfcl_single_turn", lm, "model", str(tmp_path), limit=3, tool_format="auto")
    assert r["tool_format"] == fmt
    m = r["metrics"]
    assert r["n"] == 33 and m["overall"] == 1.0 and m["decode_rate"] == 1.0 and m["context_overflow"] == 0.0
    assert all(m[c] == 1.0 for c, _, _ in bfcl_data.SINGLE_TURN), m
    recs = [json.loads(l) for l in open(tmp_path / "bfcl_single_turn.jsonl")]
    assert len(recs) == 33 and all(x["correct"] for x in recs) and all(x["prompt"] and x["output"] is not None for x in recs)
    assert all(("You are an expert in composing functions" in x["prompt"]) == (fmt == "python") for x in recs)
    assert lm.gen_lengths and len(lm.gen_lengths) == 33


@needs_data
def test_bfcl_single_turn_wrong_and_overflowing_replies(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", DATA_DIR)
    lm = fake_lm(lambda p: '<tool_call>{"name": "nope", "arguments": {}}</tool_call>')
    m = run_task("bfcl_single_turn", lm, "model", str(tmp_path), limit=2, tool_format="json")["metrics"]
    assert m["decode_rate"] == 1.0 and m["simple"] == 0.0 and m["irrelevance"] == 0.0 and m["live_relevance"] == 1.0
    recs = [json.loads(l) for l in open(tmp_path / "bfcl_single_turn.jsonl")]
    assert {x["why"] for x in recs if x["category"] == "simple"} == {"simple_function_checker:wrong_func_name"}
    lm = fake_lm(max_model_len=600)
    m = run_task("bfcl_single_turn", lm, "model", str(tmp_path), limit=2, tool_format="json")["metrics"]
    assert m["context_overflow"] == 1.0 and m["overall"] == 0.0 and lm.model.calls == 0


@needs_data
@pytest.mark.parametrize("fmt,cat", [("json", "base"), ("python", "base"), ("json", "miss_func"), ("python", "miss_param")])
def test_bfcl_multi_turn_end_to_end_with_gold_replies(tmp_path, monkeypatch, fmt, cat):
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", DATA_DIR)
    monkeypatch.setattr(mt, "MULTI_TURN", [(cat, 200)])
    scorer = bfcl_data.load_scorer(DATA_DIR)
    entries = C.spread(mt.load_entries(scorer, DATA_DIR, cat), 2)
    by_first = {}
    for e in entries:
        first = e["question"][0][0]["content"] if e["question"][0] else None
        by_first[first] = e

    def reply(prompt):
        role, _ = last_segment(prompt)
        if role != "user":
            return "Done."                                   # a tool result came back: end the turn
        users = user_segments(prompt)
        e = by_first[users[0]]                               # the first user turn identifies the entry
        turn = len(users) - 1
        calls = e["_gold"][turn]
        if not calls:
            return "Nothing to do for this request."
        return calls_as_reply([gold_string_to_call(e, s) for s in calls], fmt)

    lm = fake_lm(reply, template="x" if fmt == "python" else "tools")
    r = run_task("bfcl_multi_turn", lm, "model", str(tmp_path), limit=2, tool_format=fmt)
    m = r["metrics"]
    recs = [json.loads(l) for l in open(tmp_path / "bfcl_multi_turn.jsonl")]
    assert r["n"] == 2 and m[cat] == 1.0 and m["overall"] == 1.0 and m["context_overflow"] == 0.0, (m, [x["why"] for x in recs])
    assert all(x["correct"] and x["messages"] is None and x["steps"] for x in recs)
    assert all(s["form"] == ("structured" if fmt == "json" else "text") for x in recs for s in x["steps"])


@needs_data
def test_bfcl_multi_turn_force_termination_and_overflow(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", DATA_DIR)
    monkeypatch.setattr(mt, "MULTI_TURN", [("base", 200)])
    lm = fake_lm(lambda p: '<tool_call>{"name": "pwd", "arguments": {}}</tool_call>')   # calls forever
    m = run_task("bfcl_multi_turn", lm, "model", str(tmp_path), limit=1, tool_format="json")["metrics"]
    recs = [json.loads(l) for l in open(tmp_path / "bfcl_multi_turn.jsonl")]
    assert m["overall"] == 0.0 and recs[0]["why"] == "force_terminated" and len(recs[0]["steps"]) == 21
    lm = fake_lm(max_model_len=2000)
    m = run_task("bfcl_multi_turn", lm, "model", str(tmp_path), limit=1, tool_format="json")["metrics"]
    assert m["overall"] == 0.0 and m["context_overflow"] == 1.0


def test_safe_call_accepts_documented_calls_only():
    allowed = frozenset({"pwd", "cd", "mkdir"})
    assert mt.safe_call("cd", {"folder": "x"}, allowed)
    assert mt.safe_call("pwd", {}, allowed)
    assert not mt.safe_call("open", {"file": "x", "mode": "w"}, allowed)
    assert not mt.safe_call("__import__('os').system('ls') or cd", {}, allowed)
    assert not mt.safe_call("cd", {"folder=1) or __import__('os').system('ls') or f(x": 1}, allowed)
    assert not mt.safe_call("cd", {"": "x"}, allowed) and not mt.safe_call("cd", {1: "x"}, allowed)
    assert mt.refused_result("open") == "Error during execution: name 'open' is not defined"


@needs_data
@pytest.mark.parametrize("fmt", ["json", "python"])
def test_multi_turn_refuses_calls_upstream_would_eval_unsafely(tmp_path, monkeypatch, fmt):
    """A reply naming a builtin, an expression, or an argument that closes the call must never reach `eval`."""
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", DATA_DIR)
    monkeypatch.setattr(mt, "MULTI_TURN", [("base", 200)])
    sentinel = tmp_path / "pwned"
    cmd = f"open({str(sentinel)!r}, 'w').close()"
    attacks = [
        {"name": f"__import__('builtins').eval({cmd!r}) or pwd", "arguments": {}},     # expression as the function name
        {"name": "open", "arguments": {"file": str(sentinel), "mode": "w"}},            # a builtin, not in upstream's block list
        {"name": "pwd", "arguments": {f"x=1) or __import__('builtins').eval({cmd!r}) or pwd(y": 1}},   # argument name closes the call
    ]
    if fmt == "python":   # BFCL's decoder only parses `[f(a=1)]`, so the python format can carry the builtin only
        attacks = attacks[1:2]
    i = {"n": 0}

    def reply(prompt):
        a = attacks[i["n"] % len(attacks)]
        i["n"] += 1
        if fmt == "python":
            return "[" + C.calls_to_strings([{a["name"]: a["arguments"]}])[0] + "]"
        return "<tool_call>" + json.dumps(a) + "</tool_call>"

    lm = fake_lm(reply, template="x" if fmt == "python" else "tools")
    m = run_task("bfcl_multi_turn", lm, "model", str(tmp_path), limit=1, tool_format=fmt)["metrics"]
    recs = [json.loads(l) for l in open(tmp_path / "bfcl_multi_turn.jsonl")]
    assert not sentinel.exists(), "a model-written call was executed"
    assert m["overall"] == 0.0 and recs[0]["why"] == "force_terminated"
    steps = [s for s in recs[0]["steps"] if s.get("calls")]
    assert steps and all(s["refused"] == s["calls"] and all(r.startswith("Error during execution: name ") for r in s["results"]) for s in steps)
    assert all(turn_calls == [] for turn in recs[0]["parsed"] for turn_calls in turn), "a refused call reached the checker's input"


@needs_data
@pytest.mark.parametrize("fmt", ["json", "python"])
def test_when2call_end_to_end_with_gold_replies(tmp_path, monkeypatch, fmt):
    monkeypatch.setenv("BRAHMAI_EVAL_DATA", DATA_DIR)
    from brahmai_eval.tooluse import when2call as w2c_task
    items, seen = [], set()                                   # 3 per class, distinct questions (variants share questions)
    for c in w2c_data.CLASSES:
        for r in w2c_data.load(DATA_DIR):
            if r["correct_answer"] == c and r["question"] not in seen and sum(x["correct_answer"] == c for x in items) < 3:
                items.append(r)
                seen.add(r["question"])
    monkeypatch.setattr(w2c_task, "load", lambda d: items)
    assert len(items) == 9 and {it["correct_answer"] for it in items} == set(w2c_data.CLASSES)
    preferred, replies = [], {}
    for it in items:
        g = it["correct_answer"]
        if g == "tool_call":
            call = json.loads(it["answers"]["tool_call"])
            calls = [{call["name"]: call.get("arguments") or {}}]
            preferred.append((it["question"], C.calls_to_strings(calls)[0] if fmt == "python" else json.dumps({"name": call["name"], "arguments": calls[0][call["name"]]})))
            replies[it["question"]] = calls_as_reply(calls, fmt)
        else:
            preferred.append((it["question"], it["answers"][g]))
            replies[it["question"]] = "I am not able to do that."
    lm = fake_lm(lambda p: replies.get(last_segment(p)[1], "I am not able to do that."), preferred=preferred, template="x" if fmt == "python" else "tools")
    r = run_task("when2call", lm, "model", str(tmp_path), limit=None, tool_format="auto")
    m = r["metrics"]
    assert r["n"] == 9 and r["tool_format"] == fmt
    assert m["overall"] == m["mcq_f1"] == 1.0 and m["mcq_acc"] == 1.0 and m["tool_hallucination"] == 0.0
    assert m["gen_call_recall"] == 1.0 and m["gen_false_call"] == 0.0 and m["call_valid"] == 1.0 and m["context_overflow"] == 0.0
    recs = [json.loads(l) for l in open(tmp_path / "when2call.jsonl")]
    assert len(recs) == 9 and all(x["mcq_pred"] == x["gold"] for x in recs)
    forms = {x["candidate_forms"]["tool_call"] for x in recs}
    assert forms == ({"structured"} if fmt == "json" else {"python"}), forms
    assert all(x["candidates"]["direct"].endswith("<eos>\n") for x in recs)
