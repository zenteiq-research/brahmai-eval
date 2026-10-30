# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Resolve `--model` (a Hugging Face repo id or a local directory) into a local checkpoint directory, and apply the
optional protocol overrides (chat template file, extra stop tokens)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

from .data import home


def _is_local(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def resolve_model(spec: str, revision: str | None = None) -> str:
    """Return a directory holding config.json + weights + tokenizer.  Hub ids are downloaded into the HF cache."""
    if _is_local(spec):
        return os.path.abspath(spec)
    if os.path.isdir(spec):
        raise SystemExit(f"{spec} is a directory but has no config.json; pass a Hugging Face checkpoint directory.")
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", spec):
        raise SystemExit(f"--model {spec!r}: not an existing directory and not a Hugging Face repo id (org/name).")
    from huggingface_hub import snapshot_download
    print(f"[model] downloading {spec}" + (f"@{revision}" if revision else "") + " from the Hugging Face Hub", flush=True)
    return snapshot_download(spec, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "*.tiktoken"])


def model_name(spec: str) -> str:
    """A filesystem-safe name for the results directory."""
    s = spec.rstrip("/").replace("/", "__")
    return re.sub(r"[^\w.\-]+", "_", s)


def checksums(model_dir: str) -> dict[str, str]:
    """md5 of every protocol-relevant small file; sha256 of weight files is not computed (size only)."""
    out = {}
    for f in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            continue
        if f.endswith((".json", ".jinja", ".txt", ".model")) and os.path.getsize(p) < 64 << 20:
            out[f] = hashlib.md5(open(p, "rb").read()).hexdigest()
        elif f.endswith(".safetensors"):
            out[f] = f"size:{os.path.getsize(p)}"
    return out


def apply_overrides(model_dir: str, chat_template: str | None, stop_tokens: list[str]) -> tuple[str, dict]:
    """Materialise a copy of the checkpoint's small files with the overrides applied (weights are hard-linked or
    symlinked, never copied), so the harness and vLLM see one consistent directory.  Returns (dir, applied)."""
    applied: dict = {}
    if not chat_template and not stop_tokens:
        return model_dir, applied
    key = hashlib.md5((model_dir + "|" + (chat_template or "") + "|" + ",".join(stop_tokens)).encode()).hexdigest()[:10]
    dst = os.path.join(home(), "model-overrides", os.path.basename(model_dir.rstrip("/")) + "-" + key)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in os.listdir(model_dir):
        s, d = os.path.join(model_dir, f), os.path.join(dst, f)
        if os.path.isdir(s):
            continue
        try:
            os.link(os.path.realpath(s), d)
        except OSError:
            os.symlink(os.path.realpath(s), d)
    if chat_template:
        tpl = open(chat_template, encoding="utf-8").read()
        tc_p = os.path.join(dst, "tokenizer_config.json")
        tc = json.load(open(tc_p)) if os.path.exists(tc_p) else {}
        tc["chat_template"] = tpl
        os.unlink(tc_p) if os.path.exists(tc_p) else None
        json.dump(tc, open(tc_p, "w"), indent=1)
        jp = os.path.join(dst, "chat_template.jinja")
        if os.path.exists(jp):
            os.unlink(jp)
            open(jp, "w", encoding="utf-8").write(tpl)
        applied["chat_template"] = os.path.abspath(chat_template)
    if stop_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(dst)
        ids = []
        for t in stop_tokens:
            i = tok.convert_tokens_to_ids(t)
            if i is None or i < 0 or i == tok.unk_token_id:
                raise SystemExit(f"--stop-tokens: {t!r} is not a token of this tokenizer")
            ids.append(int(i))
        for name in ("config.json", "generation_config.json"):
            p = os.path.join(dst, name)
            if not os.path.exists(p):
                continue
            cfg = json.load(open(p))
            eos = cfg.get("eos_token_id")
            cur = [] if eos is None else (eos if isinstance(eos, list) else [eos])
            cfg["eos_token_id"] = sorted(set(int(x) for x in cur) | set(ids))
            os.unlink(p)
            json.dump(cfg, open(p, "w"), indent=1)
        applied["stop_tokens"] = {t: i for t, i in zip(stop_tokens, ids)}
    return dst, applied
