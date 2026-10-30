# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""vLLM backend for lm-evaluation-harness with a few protocol fixes that hold for any checkpoint:

* context/continuation pairs are encoded independently, with trailing whitespace of the context moved onto the
  continuation (the harness default encodes jointly and splits, which fails on empty continuations and can
  tokenise across the boundary);
* generated text is decoded with special tokens skipped, so a stop token never leaks into scored text;
* every eos id declared by config.json / generation_config.json is passed as a vLLM stop token;
* optional: a literal BOS token prefixed to every raw (non-chat) prompt (`--add-bos`).
"""
from __future__ import annotations

import json
import os
from typing import Any

from lm_eval.models.vllm_causallms import VLLM


def declared_eos_ids(model_dir: str) -> list[int]:
    ids: set[int] = set()
    for name in ("config.json", "generation_config.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            eos = json.load(open(p)).get("eos_token_id")
            for x in (eos if isinstance(eos, list) else [eos]):
                if x is not None:
                    ids.add(int(x))
    return sorted(ids)


class EvalVLLM(VLLM):
    def __init__(self, pretrained: str, *, add_bos_prefix: bool = False, **kw: Any):
        kw.setdefault("dtype", "bfloat16")
        kw.setdefault("seed", 1234)
        super().__init__(pretrained, **kw)
        self._add_bos_prefix = add_bos_prefix
        self._bos_str = self.tokenizer.bos_token or ""
        self.stop_token_ids = sorted(set(declared_eos_ids(pretrained)) | ({int(self.tokenizer.eos_token_id)} if self.tokenizer.eos_token_id is not None else set()))
        self.gen_lengths: list[int] = []

    # ── encoding ──────────────────────────────────────────────────────────────────────────────
    def _bos(self, s: str) -> str:
        if self._add_bos_prefix and self._bos_str and s and not s.startswith(self._bos_str):
            return self._bos_str + s
        return s

    def tok_encode(self, string, add_special_tokens=None, **kwargs):  # type: ignore[override]
        if add_special_tokens is None:            # a context, not a continuation
            string = [self._bos(s) for s in string] if isinstance(string, list) else self._bos(string)
        return super().tok_encode(string, add_special_tokens=add_special_tokens, **kwargs)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        n = len(context) - len(context.rstrip())
        if n > 0:
            continuation = context[-n:] + continuation
            context = context[:-n]
        return self.tok_encode(context), self.tok_encode(continuation, add_special_tokens=False)

    # ── generation ────────────────────────────────────────────────────────────────────────────
    def modify_gen_kwargs(self, gen_kwargs, eos=None, default_max_gen_toks=256):  # type: ignore[override]
        kwargs, until, max_gen_toks = VLLM.modify_gen_kwargs(gen_kwargs, eos=eos, default_max_gen_toks=default_max_gen_toks)
        kwargs["skip_special_tokens"] = True
        kwargs["stop_token_ids"] = sorted(set(kwargs.get("stop_token_ids") or []) | set(self.stop_token_ids))
        return kwargs, until, max_gen_toks

    def _model_generate(self, requests, generate=False, sampling_params=None):
        outputs = super()._model_generate(requests, generate=generate, sampling_params=sampling_params)
        if generate:
            self.gen_lengths.extend(len(o.outputs[0].token_ids) for o in outputs)
        return outputs

    def reset_diag(self) -> None:
        self.gen_lengths = []

    def diag(self) -> dict:
        if not self.gen_lengths:
            return {}
        g = sorted(self.gen_lengths)
        return {"gen_n": len(g), "gen_toks_p50": g[len(g) // 2], "gen_toks_max": g[-1]}

    # ── identity ──────────────────────────────────────────────────────────────────────────────
    def engine_info(self) -> dict:
        import vllm
        mc = self.model.llm_engine.model_config
        return {"engine": "vllm", "vllm_version": vllm.__version__, "dtype": str(mc.dtype), "max_model_len": int(mc.max_model_len),
                "seed": self.model_args.get("seed"), "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.model_args.get("gpu_memory_utilization"),
                "stop_token_ids": self.stop_token_ids, "add_bos_prefix": self._add_bos_prefix,
                "chat_template": bool(self.hf_chat_template)}

    def probe_generate(self, prompt: str, max_new: int) -> tuple[str, int]:
        from vllm import SamplingParams
        o = self.model.generate([prompt], SamplingParams(temperature=0, max_tokens=max_new, stop_token_ids=self.stop_token_ids,
                                                         skip_special_tokens=True), use_tqdm=False)[0].outputs[0]
        return o.text, len(o.token_ids)


def tokenizer_parity_check(lm: EvalVLLM, model_dir: str) -> dict:
    """vLLM may wrap or re-load the tokenizer; prove it encodes like transformers' AutoTokenizer from the same files."""
    from transformers import AutoTokenizer
    ref = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=bool(lm.model_args.get("trust_remote_code")))
    probes = ["What is 2+2? Answer briefly.", " Answer: (B)", "नमस्ते दुनिया", "def f(x):\n    return x\n"]
    mism = [{"text": p, "hf": a, "vllm": b} for p in probes
            if (a := list(ref(p, add_special_tokens=False).input_ids)) != (b := list(lm.tokenizer(p, add_special_tokens=False).input_ids))]
    if mism:
        raise SystemExit(f"ABORT: vLLM tokenizer disagrees with transformers on {len(mism)} probe(s): {json.dumps(mism[:2], ensure_ascii=False)[:400]}")
    return {"probes": len(probes), "mismatch": 0}
