# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Metrics for the IndicGenBench tasks (XQuAD-IN, XorQA-IN, Flores-IN, CrossSum-IN).

Extractive QA is scored with the SQuAD exact-match and token-F1 implementation from
`transformers`; translation and summarisation with sentence-level chrF++ from sacrebleu.
"""
import transformers.data.metrics.squad_metrics as squad_metrics
from sacrebleu.metrics import CHRF

# word_order=2 is chrF++; character n-grams make it comparable across scripts.
_CHRF = CHRF(word_order=2)


def _first(ans):
    """Text of the first answer. IndicGenBench stores answers as a list of
    {"answer_start", "text"} dicts; an empty list yields ""."""
    if not ans:
        return ""
    a = ans[0]
    return a.get("text", "") if isinstance(a, dict) else str(a)


def process_results_qa(doc, results):
    """SQuAD F1 and exact match of the generation against the first gold answer."""
    pred = results[0].strip()
    ref = _first(doc["answers"])
    return {"f1": squad_metrics.compute_f1(ref, pred),
            "exact_match": squad_metrics.compute_exact(ref, pred)}


def process_results_xorqa(doc, results):
    """XorQA-IN rows carry an English answer (`answers`) and an in-language answer
    (`translated_answers`). One generation is scored against both: `*_inlang` against the
    in-language answer (falling back to the English one when it is empty), `*_en` against
    the English answer."""
    pred = results[0].strip()
    ref_en = _first(doc["answers"])
    ref_xx = _first(doc["translated_answers"]) or ref_en
    return {"f1_en": squad_metrics.compute_f1(ref_en, pred),
            "em_en": squad_metrics.compute_exact(ref_en, pred),
            "f1_inlang": squad_metrics.compute_f1(ref_xx, pred),
            "em_inlang": squad_metrics.compute_exact(ref_xx, pred)}


def process_results_chrf(doc, results):
    """Sentence chrF++ of the generation against the reference (`target` for translation,
    `summary` for summarisation), rescaled from 0-100 to 0-1."""
    pred = results[0].strip()
    ref = doc.get("target") or doc.get("summary") or ""
    return {"chrf": _CHRF.sentence_score(pred, [ref]).score / 100.0}
