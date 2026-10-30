# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for the tool-use tasks: tool-call formats, chat rendering, generation and scoring helpers.

Tool-call formats (`tool_format`):
  * "json":   the tools are declared through the chat template (`tools=`) and the reply carries JSON objects
              `{"name": ..., "arguments": {...}}` inside <tool_call>...</tool_call> blocks (a bare JSON object or array
              is also accepted).  When the chat template renders calls as `<tool_call>name<arg_key>k</arg_key>
              <arg_value>v</arg_value>`, that form is parsed too, with argument values coerced to the declared types.
  * "python": BFCL's prompting convention: the function docs are pasted into the system prompt and the reply is
              `[func(arg=val), ...]`, decoded by the official `default_decode_ast_prompting`.
  * "auto":   "json" when the chat template mentions "tool_call" or "tools", else "python".
"""
from __future__ import annotations

import ast
import copy
import json
import re
from types import SimpleNamespace

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
TOOL_MARKERS = {"<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"}
FORMATS = ("json", "python")
SYSTEM_CANARY = "You are an expert in composing functions"


# ── formats ────────────────────────────────────────────────────────────────────────────────────────────────────────
def resolve_format(tool_format: str, chat_template: str | None) -> str:
    if tool_format in FORMATS:
        return tool_format
    if tool_format != "auto":
        raise ValueError(f"tool_format must be one of {FORMATS + ('auto',)}, got {tool_format!r}")
    t = chat_template or ""
    return "json" if ("tool_call" in t or "tools" in t) else "python"


def functions_to_tools(scorer, functions: list) -> list:
    """BFCL function docs -> `{"type": "function", "function": {...}}` tool schemas for the template's `tools=` slot.

    Uses the official `convert_to_tool` with the one model style that keeps dotted function names and every schema
    field (the gold answers use the names as written)."""
    if not functions:
        return []
    mu, ms, tm = scorer.model_utils, scorer.model_style, scorer.type_mappings
    tools = mu.convert_to_tool(copy.deepcopy(functions), tm.GORILLA_TO_OPENAPI, ms.ModelStyle.FIREWORK_AI)
    if not tools:
        raise RuntimeError(f"convert_to_tool returned nothing for {len(functions)} function(s)")
    return tools


def prepare_functions(scorer, functions: list, category: str) -> list:
    """Upstream's language hint on every function doc (a deep copy; the checker reads the declared types)."""
    return scorer.model_utils.func_doc_language_specific_pre_processing(copy.deepcopy(functions), category)


def python_messages(scorer, messages: list, functions: list, category: str) -> list:
    """BFCL's prompting protocol: the system prompt with the function docs pasted in (upstream formats the docs with
    `str.format`, i.e. as their Python repr)."""
    return scorer.model_utils.system_prompt_pre_processing_chat_model(copy.deepcopy(messages), functions, category)


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────────────────────
def coerce(value: str, type_str: str | None):
    """Recover a typed value from its text form inside <arg_value>, guided by the declared parameter type.
    Anything that does not convert is returned as the original string (the checker then fails it)."""
    v = value
    if type_str in (None, "string", "any"):
        return v
    if type_str == "boolean":
        low = v.strip().lower()
        return low == "true" if low in ("true", "false") else v
    if type_str == "integer":
        try:
            return int(v.strip())
        except ValueError:
            try:
                f = float(v.strip())
                return int(f) if f.is_integer() else v
            except ValueError:
                return v
    if type_str in ("float", "number"):
        try:
            return float(v.strip())
        except ValueError:
            return v
    if type_str in ("array", "tuple", "dict", "object"):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
    return v


def _call_obj(o) -> dict | None:
    """`{"name", "arguments"}`, `{"function": {...}}` or `{"name", "parameters"}` -> {name: args}; None if not a call."""
    if not isinstance(o, dict):
        return None
    fn = o["function"] if isinstance(o.get("function"), dict) else o
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = fn.get("arguments")
    if args is None:
        args = fn.get("parameters")
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise ValueError(f"arguments is not a JSON object: {e}") from e
    if not isinstance(args, dict):
        raise ValueError("arguments is not a JSON object")
    return {name.strip(): args}


def parse_json_calls(text: str, functions: list | None = None, structured: bool = False) -> list[dict]:
    """Reply text -> BFCL's `[{name: {arg: val}}, ...]`.  Raises ValueError when the reply holds no tool call."""
    blocks = TOOL_CALL_RE.findall(text)
    if not blocks:
        s = text.strip()
        if s.startswith(("{", "[")):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                obj = None
            items = obj if isinstance(obj, list) else [obj]
            calls = [_call_obj(o) for o in items] if items else []
            if calls and all(c is not None for c in calls):
                return calls
        raise ValueError("no tool call in the reply")
    props_by_name = {f["name"]: ((f.get("parameters") or {}).get("properties") or {}) for f in (functions or [])}
    calls = []
    for b in blocks:
        s = b.strip()
        if s.startswith(("{", "[")):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSON tool call: {e}") from e
            for o in (obj if isinstance(obj, list) else [obj]):
                c = _call_obj(o)
                if c is None:
                    raise ValueError("tool call without a name")
                calls.append(c)
        elif structured:
            name = s.split("<arg_key>")[0].strip()
            if not name:
                raise ValueError("tool call without a name")
            props = props_by_name.get(name, {})
            args = {k.strip(): coerce(v, (props.get(k.strip()) or {}).get("type")) for k, v in ARG_RE.findall(s)}
            calls.append({name: args})
        else:
            raise ValueError("tool call block is not JSON")
    return calls


def parse_python_calls(scorer, text: str) -> list[dict]:
    """BFCL's `[func(arg=val)]` -> `[{name: {arg: val}}]` via the official decoder.  Raises ValueError on failure."""
    try:
        return scorer.model_utils.default_decode_ast_prompting(text)
    except Exception as e:  # noqa: BLE001 -- the decoder raises SyntaxError/AssertionError/Exception
        raise ValueError(f"{type(e).__name__}: {e}") from e


def is_empty_calls(scorer, decoded) -> bool:
    return scorer.format_helpers.is_empty_output(decoded)


def calls_to_strings(calls: list[dict]) -> list[str]:
    """`[{name: {k: v}}]` -> `["name(k=repr(v), ...)"]`, the executable form upstream feeds its API classes."""
    out = []
    for c in calls:
        for name, args in c.items():
            out.append(f"{name}({','.join(f'{k}={v!r}' for k, v in (args or {}).items())})")
    return out


# ── chat rendering ─────────────────────────────────────────────────────────────────────────────────────────────────
def _plain(m: dict) -> bool:
    return m["role"] in ("user", "assistant") and not m.get("tool_calls")


def merge_consecutive(msgs: list) -> list:
    """Join consecutive same-role user/assistant turns with a blank line (for templates that enforce alternation)."""
    out: list = []
    for m in msgs:
        if out and _plain(m) and _plain(out[-1]) and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": (out[-1].get("content") or "") + "\n\n" + (m.get("content") or "")}
        else:
            out.append(dict(m))
    return out


def fold_system(msgs: list) -> list:
    """Prepend the system text to the first user turn (for templates that reject the system role)."""
    if not msgs or msgs[0]["role"] != "system":
        return msgs
    rest = msgs[1:]
    if rest and rest[0]["role"] == "user":
        first = {"role": "user", "content": msgs[0]["content"] + "\n\n" + (rest[0].get("content") or "")}
        return merge_consecutive([first] + rest[1:])
    return merge_consecutive([{"role": "user", "content": msgs[0]["content"]}] + rest)


def _sampling_params(**kw):
    try:
        from vllm import SamplingParams
    except ImportError:  # unit tests with a stand-in engine
        return SimpleNamespace(**kw)
    return SamplingParams(**kw)


def _tokens_prompt(ids: list[int]):
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        return {"prompt_token_ids": ids}
    return TokensPrompt(prompt_token_ids=ids)


class Engine:
    """The harness model object (`lm.model` is the vLLM engine, `lm.tokenizer` the tokenizer) plus the format logic."""

    def __init__(self, lm, tool_format: str = "auto"):
        self.lm = lm
        self.tok = lm.tokenizer
        self.stop_ids = sorted({int(x) for x in (lm.stop_token_ids or [])})
        self.max_len = int(lm.model.llm_engine.model_config.max_model_len)
        template = getattr(self.tok, "chat_template", None) or ""
        self.format = resolve_format(tool_format, template)
        self.structured = "<arg_key>" in template
        specials = set(getattr(self.tok, "all_special_tokens", None) or []) | set(getattr(self.tok, "additional_special_tokens", None) or [])
        self.specials = sorted((t for t in specials if t and t not in TOOL_MARKERS), key=len, reverse=True)
        self.eos = getattr(self.tok, "eos_token", None) or ""

    # rendering ----------------------------------------------------------------------------------------------------
    def render(self, messages: list, tools: list | None = None, add_generation_prompt: bool = True) -> tuple[str, str]:
        """Apply the chat template; returns (text, flag).  flag records which fallback was needed: "" (none),
        "merged_turns" (consecutive same-role turns joined) or "system_folded" (system text moved into the first user
        turn), the same accommodations upstream makes for templates that enforce alternation or reject `system`."""
        kw = {"tools": tools} if tools else {}
        variants = [(messages, "")]
        merged = merge_consecutive(messages)
        if merged != messages:
            variants.append((merged, "merged_turns"))
        if merged and merged[0]["role"] == "system":
            variants.append((fold_system(merged), "system_folded"))
        err: Exception | None = None
        for msgs, flag in variants:
            try:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw), flag
            except Exception as e:  # noqa: BLE001 -- template errors are not typed
                err = e
        raise RuntimeError(f"the chat template cannot render this conversation: {type(err).__name__}: {err}") from err

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def fits(self, ids: list[int], max_new: int) -> bool:
        return len(ids) + max_new <= self.max_len

    # generation ---------------------------------------------------------------------------------------------------
    def strip(self, text: str) -> str:
        """Drop a reasoning trace and the tokenizer's control tokens; keep the tool-call delimiters."""
        text = THINK_RE.sub("", text)
        for t in self.specials:
            text = text.replace(t, "")
        return text.strip()

    def generate(self, ids_list: list[list[int]], max_new: int) -> list[SimpleNamespace]:
        """Greedy completions.  Special tokens are kept in the decoded text (tool-call delimiters can be special
        tokens) and removed by `strip`."""
        if not ids_list:
            return []
        sp = _sampling_params(temperature=0, max_tokens=max_new, stop_token_ids=self.stop_ids, skip_special_tokens=False)
        outs = self.lm.model.generate([_tokens_prompt(ids) for ids in ids_list], sp, use_tqdm=False)
        gens = []
        for o in outs:
            c = o.outputs[0]
            gens.append(SimpleNamespace(raw=c.text, text=self.strip(c.text), n_tokens=len(c.token_ids), finish=c.finish_reason))
        lengths = getattr(self.lm, "gen_lengths", None)
        if lengths is not None:
            lengths.extend(g.n_tokens for g in gens)
        return gens

    def logprob_sums(self, pairs: list[tuple[list[int], list[int]]]) -> list[tuple[float, int]]:
        """Sum of the log-probabilities of the continuation tokens given the prompt tokens, per (prompt, continuation)."""
        if not pairs:
            return []
        sp = _sampling_params(temperature=0, max_tokens=1, prompt_logprobs=0, skip_special_tokens=True)
        outs = self.lm.model.generate([_tokens_prompt(p + c) for p, c in pairs], sp, use_tqdm=False)
        res = []
        for (p, c), o in zip(pairs, outs):
            ids, plp = list(o.prompt_token_ids), o.prompt_logprobs
            total = 0.0
            for j in range(len(p), len(p) + len(c)):
                entry = plp[j].get(ids[j]) if plp[j] else None
                if entry is None:
                    raise RuntimeError(f"no prompt logprob for token {ids[j]} at position {j}")
                total += float(entry.logprob)
            res.append((total, len(c)))
        return res

    # parsing ------------------------------------------------------------------------------------------------------
    def parse_calls(self, scorer, text: str, functions: list | None = None) -> list[dict]:
        """Stripped reply -> `[{name: {arg: val}}]` in the run's format; raises ValueError when there is no call."""
        if self.format == "python":
            return parse_python_calls(scorer, text)
        return parse_json_calls(text, functions, structured=self.structured)


def check_prompt_carries_tools(prompt: str, functions: list, fmt: str) -> None:
    """Abort rather than score a prompt the template silently stripped the tools from."""
    if fmt == "python" and SYSTEM_CANARY not in prompt:
        raise RuntimeError("the rendered prompt lacks the BFCL system prompt: the chat template drops the system role, so "
                           f"every function doc would be discarded.  Prompt head: {prompt[:300]!r}")
    if functions and functions[0]["name"] not in prompt:
        raise RuntimeError("the function docs are missing from the rendered prompt (the chat template ignored them); "
                           f"the tool-use tasks would measure a template problem.  Prompt head: {prompt[:300]!r}")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_record(fh, rec: dict) -> None:
    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def spread(items, limit):
    """`limit` items evenly spaced over the list (a smoke subset that keeps every class of a class-ordered set)."""
    if not limit or limit >= len(items):
        return list(items)
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]
