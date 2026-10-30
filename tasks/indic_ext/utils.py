# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Loaders, prompt builders and metrics for the indic_ext suite.

Every task YAML in this directory names a public Hugging Face dataset, pinned to a
commit, in `dataset_kwargs` (path / name / revision) and points `custom_dataset` at
one of the `<family>_dataset` builders below.  A builder downloads the official
splits and returns a `DatasetDict` with two splits:

  test     the scored rows (the full official test split, or a seeded random
           subsample of it when the family is capped), in source order;
  fewshot  the exemplar pool, always disjoint from `test` by content: the official
           dev / validation split where one exists, otherwise seeded held-out rows
           of the test split.

All sampling uses `random.Random(seed)` with the `seed` string given in the YAML, so
the scored set and the exemplar pool are reproducible from the Hub alone.
"""

import random

import datasets
import transformers.data.metrics.squad_metrics as squad_metrics

FEWSHOT_POOL = 64
CAPS = {"gmmlu": 2000, "milu": 500, "immlupro": 500, "indicxnli": 2000, "indicqa": 500}
QA_EXEMPLAR_MAX_CHARS = 1000  # exemplar passages only; scored passages are never length-filtered
INCLUDE_MIN_POOL = 10  # below this many clean validation rows, hold out test rows ...
INCLUDE_POOL_TARGET = 16  # ... until the exemplar pool has this many
LETTERS = "ABCDEFGHIJ"


# ── sampling helpers ────────────────────────────────────────────────────────────────────
def split_sample(rows, cap, rng):
    """Seeded random `cap` rows (kept in original order) and the disjoint remainder."""
    if cap is None or cap >= len(rows):
        return list(rows), []
    idx = set(rng.sample(range(len(rows)), cap))
    return [r for i, r in enumerate(rows) if i in idx], [r for i, r in enumerate(rows) if i not in idx]


def pool(rows, n, rng):
    return split_sample(rows, n, rng)[0]


def content_key(r):
    """What makes two rows the same item, per family (ignores metadata and option order)."""
    if "choice1" in r:  # IndicCOPA (its `question` is cause/effect)
        return "copa|" + r["premise"].strip() + "|" + r["choice1"].strip() + "|" + r["choice2"].strip()
    if "hypothesis" in r:  # IndicXNLI
        return "xnli|" + r["premise"].strip() + "|" + r["hypothesis"].strip()
    if "review" in r:  # IndicSentiment
        return "senti|" + r["review"].strip()
    if "context" in r:  # IndicQA
        return "qa|" + r["context"].strip() + "|" + r["question"].strip()
    return "mc|" + r["question"].strip()  # Global-MMLU / INCLUDE / MILU / IndicMMLU-Pro


def _splits(test_rows, fewshot_rows):
    """Drop every exemplar that is the same item as a scored row, then build the DatasetDict."""
    scored = {content_key(r) for r in test_rows}
    fewshot_rows = [r for r in fewshot_rows if content_key(r) not in scored]
    assert len(fewshot_rows) >= 6, f"only {len(fewshot_rows)} clean exemplars"
    return datasets.DatasetDict(
        {"test": datasets.Dataset.from_list(test_rows), "fewshot": datasets.Dataset.from_list(fewshot_rows)}
    )


def _load(split, path, name=None, revision=None, data_files=None):
    kw = {"data_files": data_files} if data_files else {}
    return datasets.load_dataset(path, name, revision=revision, split=split, **kw).to_list()


def _hf_url(path, revision, relpath):
    return f"hf://datasets/{path}@{revision}/{relpath}"


def _load_json(path, revision, relpath, field=None):
    return datasets.load_dataset(
        "json", data_files={"x": _hf_url(path, revision, relpath)}, field=field, split="x"
    ).to_list()


# ── Global-MMLU: seeded 2000 of the test split, exemplars = the dev split ───────────────
def global_mmlu_dataset(seed, path, name=None, revision=None, data_files=None, **_):
    rng = random.Random(seed)
    test = _load("test", path, name, revision, data_files)
    dev = _load("dev", path, name, revision, data_files)
    keep, _ = split_sample(test, CAPS["gmmlu"], rng)

    def norm(r):
        return {
            "sample_id": r["sample_id"], "subject": r["subject"], "subject_category": r["subject_category"],
            "question": r["question"], "option_a": r["option_a"], "option_b": r["option_b"],
            "option_c": r["option_c"], "option_d": r["option_d"], "answer": r["answer"],
            "cultural_sensitivity_label": r["cultural_sensitivity_label"],
        }

    return _splits([norm(r) for r in keep], [norm(r) for r in dev])


# ── INCLUDE: full test split, exemplars = validation rows not also in test ──────────────
def include_dataset(seed, path, name=None, revision=None, data_files=None, **_):
    test = _load("test", path, name, revision, data_files)
    val = _load("validation", path, name, revision, data_files)

    def norm(r):
        return {
            "question": r["question"], "option_a": r["option_a"], "option_b": r["option_b"],
            "option_c": r["option_c"], "option_d": r["option_d"], "answer": int(r["answer"]),
            "domain": r["domain"], "subject": r["subject"], "country": r["country"],
            "regional_feature": r["regional_feature"], "level": r["level"],
        }

    bad = [r for r in test if not (0 <= int(r["answer"]) <= 3)]
    assert not bad, f"include: {len(bad)} rows with answer outside 0..3"
    test = [norm(r) for r in test]
    tkeys = {content_key(r) for r in test}
    val_clean = [norm(r) for r in val if content_key(norm(r)) not in tkeys]
    held = []
    if len(val_clean) < INCLUDE_MIN_POOL:
        # validation is mostly a copy of test: hold out seeded test rows to fill the pool
        held, test = split_sample(test, INCLUDE_POOL_TARGET - len(val_clean), random.Random(seed))
    return _splits(test, val_clean + held)


# ── MILU: seeded 500 of the test split, exemplars = seeded 64 of validation ─────────────
def milu_dataset(seed, path, name=None, revision=None, data_files=None, **_):
    rng = random.Random(seed)
    test = _load("test", path, name, revision, data_files)
    val = _load("validation", path, name, revision, data_files)
    keep, _ = split_sample(test, CAPS["milu"], rng)
    opt_idx = {"option1": 0, "option2": 1, "option3": 2, "option4": 3}

    def norm(r):
        return {
            "question": r["question"], "option_a": r["option1"], "option_b": r["option2"],
            "option_c": r["option3"], "option_d": r["option4"], "answer": opt_idx[r["target"]],
            "domain": r["domain"], "subject": r["subject"], "is_translated": r["is_translated"],
        }

    return _splits([norm(r) for r in keep], [norm(r) for r in pool(val, FEWSHOT_POOL, rng)])


# ── IndicMMLU-Pro: seeded 500 of the test split, exemplars = seeded 64 held-out rows ────
def indicmmlu_pro_dataset(seed, path, name=None, revision=None, data_files=None, **_):
    rng = random.Random(seed)
    rows = [
        {
            "question_id": r["question_id"], "question": r["question"], "options": list(r["options"]),
            "answer": r["answer"], "answer_index": int(r["answer_index"]), "category": r["category"],
            "src": r["src"],
        }
        for r in _load("test", path, name, revision, data_files)
    ]
    # A row whose gold letter and gold index disagree has no usable gold: drop it.
    bad = [
        r for r in rows
        if not (0 <= r["answer_index"] < len(r["options"])) or LETTERS[r["answer_index"]] != r["answer"]
    ]
    rows = [r for r in rows if r not in bad]
    keep, rest = split_sample(rows, CAPS["immlupro"], rng)
    return _splits(keep, pool(rest, FEWSHOT_POOL, rng))


# ── IndicXNLI (forward translation): seeded 2000 of test, exemplars = seeded 64 of dev ──
def indicxnli_dataset(seed, path, name, revision, **_):
    rng = random.Random(seed)
    test = _load_json(path, revision, f"forward/test/xnli_{name}.json", field="test")
    dev = _load_json(path, revision, f"forward/dev/xnli_{name}.json", field="validation")
    assert all(r["label"] in (0, 1, 2) for r in test + dev), "indicxnli: unexpected label"
    keep, _ = split_sample(test, CAPS["indicxnli"], rng)

    def norm(r):
        return {"premise": r["premise"], "hypothesis": r["hypothesis"], "label": int(r["label"])}

    return _splits([norm(r) for r in keep], [norm(r) for r in pool(dev, FEWSHOT_POOL, rng)])


# ── IndicCOPA: full test split minus 32 seeded held-out exemplar rows ───────────────────
def indiccopa_dataset(seed, path, name, revision, **_):
    rng = random.Random(seed)
    rows = [
        {
            "idx": r["idx"], "premise": r["premise"], "choice1": r["choice1"], "choice2": r["choice2"],
            "question": r["question"], "label": int(r["label"]), "changed": r.get("changed"),
        }
        for r in _load_json(path, revision, f"data/test.{name}.jsonl")
    ]
    assert all(r["label"] in (0, 1) and r["question"] in ("cause", "effect") for r in rows), name
    held, keep = split_sample(rows, 32, rng)
    return _splits(keep, held)


# ── IndicSentiment: labelled test rows, exemplars = seeded 64 of validation ─────────────
def indicsentiment_dataset(seed, path, name, revision, **_):
    # `en` is the ENGLISH REVIEW column of the hi file (every language file carries the
    # same 1,000 source reviews); the other languages use their INDIC REVIEW column.
    src_lang, col = ("hi", "ENGLISH REVIEW") if name == "en" else (name, "INDIC REVIEW")

    def rows(split):
        out = []
        for r in _load_json(path, revision, f"data/{split}/{src_lang}.json"):
            lab = r.get("LABEL")
            if lab not in ("Positive", "Negative") or not (r.get(col) or "").strip():
                continue
            out.append({
                "review": r[col].strip(), "english_review": (r.get("ENGLISH REVIEW") or "").strip(),
                "label": 1 if lab == "Positive" else 0, "category": r.get("GENERIC CATEGORIES"),
            })
        return out

    test = rows("test")
    val = rows("validation")
    return _splits(test, pool(val, FEWSHOT_POOL, random.Random(seed)))


# ── IndicQA: answerable rows only, seeded 500; exemplars = seeded 64 short held-out rows ─
def indicqa_dataset(seed, path, name, revision, **_):
    rng = random.Random(seed)
    rows = []
    for art in _load_json(path, revision, f"data/indicqa.{name}.json", field="data"):
        for para in art["paragraphs"]:
            for q in para["qas"]:
                texts = [a["text"].strip() for a in q["answers"] if (a.get("text") or "").strip()]
                if not texts:  # unanswerable (empty gold): not scored
                    continue
                rows.append({
                    "id": f"{name}-{q['id']}", "title": art.get("title") or "", "context": para["context"],
                    "question": q["question"].strip(),
                    "answers": {
                        "text": texts,
                        "answer_start": [a.get("answer_start") for a in q["answers"] if (a.get("text") or "").strip()],
                    },
                })
    keep, rest = split_sample(rows, CAPS["indicqa"], rng)
    # Exemplars come from the short end of the held-out rows so that five of them fit in
    # the context window: cut = max(1000 chars, the language's p25 of held-out passage length).
    lengths = sorted(len(r["context"]) for r in rest)
    cut = max(QA_EXEMPLAR_MAX_CHARS, lengths[len(lengths) // 4])
    short = [r for r in rest if len(r["context"]) <= cut]
    return _splits(keep, pool(short, FEWSHOT_POOL, rng))


# ── IndicMMLU-Pro prompts: 3..10 options per row, so the choice list is per doc ─────────
def immlupro_doc_to_text(doc):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(doc["options"]))
    return f"{doc['question'].strip()}\n{opts}\nAnswer:"


def immlupro_doc_to_choice(doc):
    # Exactly as long as the row's option list, so the gold letter is compared against the
    # options that exist and the normalisation is not diluted for 4-option rows.
    return list(LETTERS[: len(doc["options"])])


# ── IndicCOPA: cause/effect question, in-language alternatives as continuations ─────────
_COPA_Q = {"cause": "What was the cause of this?", "effect": "What happened as a result?"}


def copa_doc_to_text(doc):
    return f"{doc['premise'].strip()}\nQuestion: {_COPA_Q[doc['question']]}\nAnswer:"


# ── IndicQA: SQuAD-style extractive QA, scored with the SQuAD exact-match / F1 ──────────
def process_results_qa(doc, results):
    pred = results[0].strip()
    ref = doc["answers"]["text"][0]
    return {"f1": squad_metrics.compute_f1(ref, pred), "exact_match": squad_metrics.compute_exact(ref, pred)}
