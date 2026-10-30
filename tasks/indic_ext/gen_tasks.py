#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Write the indic_ext task YAMLs and TASKS.json from the table below.

Run from anywhere: `python tasks/indic_ext/gen_tasks.py`.  The committed YAMLs are what
the runner uses; this script only keeps them consistent.  Each task loads a public
Hugging Face dataset pinned to a commit (see SOURCES) through `utils.<family>_dataset`,
which builds the scored `test` split and the disjoint `fewshot` exemplar split with the
task's fixed seed string.

Protocol: multiple-choice tasks are raw text, 5-shot, loglikelihood over the option
letters (Global-MMLU, INCLUDE, MILU, IndicMMLU-Pro) or over the in-language alternatives
(IndicXNLI as True/Neither/False, IndicCOPA, IndicSentiment); IndicQA is chat, 5-shot,
greedy extractive QA scored with SQuAD exact-match / F1.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1234

# family -> (hub repo, pinned commit, license as stated by the dataset card, builder)
SOURCES = {
    "gmmlu": ("CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993", "apache-2.0", "global_mmlu_dataset"),
    "include": ("CohereLabs/include-base-44", "d2e1f6015f67a43c02a9a68db98e2298e2d6a660", "apache-2.0", "include_dataset"),
    "milu": ("ai4bharat/MILU", "946c423e72cd2657674a7a65d739e212e9a5f876", "cc-by-4.0", "milu_dataset"),
    "immlupro": ("LinguaLift/IndicMMLU-Pro", "9cadffd98bb7e2853736949593bd54a6d96e4f13", "not stated", "indicmmlu_pro_dataset"),
    "indicxnli": ("Divyanshu/indicxnli", "7092c27872e919f31d0496fb8b9c47bd2cba3f6c", "cc0-1.0", "indicxnli_dataset"),
    "indiccopa": ("ai4bharat/IndicCOPA", "d356ef19a4eb287e88a51d07a56b73ba88c7f188", "cc-by-4.0", "indiccopa_dataset"),
    "indicsentiment": ("ai4bharat/IndicSentiment", "dc8f3f66886531c6897fedffca1e938a68fc5013", "not stated", "indicsentiment_dataset"),
    "indicqa": ("ai4bharat/IndicQA", "d7e91454f5c8a85b82e2549453521a2b7a185c44", "cc-by-4.0", "indicqa_dataset"),
}

# The dataset config (Global-MMLU, INCLUDE, MILU, IndicMMLU-Pro) or the language code used
# in the raw file names (IndicXNLI, IndicCOPA, IndicSentiment, IndicQA) per task language.
CONFIG = {
    "gmmlu": {lg: lg for lg in ["bn", "hi", "ne", "si", "te", "en"]},
    "include": {"bn": "Bengali", "hi": "Hindi", "ml": "Malayalam", "ne": "Nepali", "ta": "Tamil",
                "te": "Telugu", "ur": "Urdu"},
    "milu": {"bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam",
             "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "en": "English"},
    "immlupro": {"bn": "bengali", "gu": "gujarati", "hi": "hindi", "kn": "kannada", "mr": "marathi",
                 "pa": "punjabi", "ta": "tamil", "te": "telugu", "ur": "urdu"},
    "indicxnli": {lg: lg for lg in ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]},
    "indiccopa": {lg: lg for lg in ["as", "bn", "gom", "gu", "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa",
                                    "sa", "sat", "sd", "ta", "te", "ur", "en"]},
    "indicsentiment": {lg: lg for lg in ["as", "bd", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta",
                                         "te", "ur", "en"]},
    "indicqa": {lg: lg for lg in ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]},
}

# (task, num_fewshot, chat, cost_secs, n_docs): the suite table; n_docs = scored rows the
# builder yields (full split, capped sample, or full split minus held-out exemplars).
TASKS = [
    *[(f"gmmlu_{lg}", 5, False, 320, 2000) for lg in CONFIG["gmmlu"]],
    ("include_bn", 5, False, 88, 548), ("include_hi", 5, False, 88, 547), ("include_ml", 5, False, 75, 467),
    ("include_ne", 5, False, 80, 500), ("include_ta", 5, False, 88, 550), ("include_te", 5, False, 88, 548),
    ("include_ur", 5, False, 55, 341),
    *[(f"milu_{lg}", 5, False, 80, 500) for lg in CONFIG["milu"]],
    ("immlupro_bn", 5, False, 188, 500), ("immlupro_gu", 5, False, 190, 500), ("immlupro_hi", 5, False, 187, 500),
    ("immlupro_kn", 5, False, 189, 500), ("immlupro_mr", 5, False, 189, 500), ("immlupro_pa", 5, False, 190, 500),
    ("immlupro_ta", 5, False, 191, 500), ("immlupro_te", 5, False, 190, 500), ("immlupro_ur", 5, False, 191, 500),
    *[(f"indicxnli_{lg}", 5, False, 60, 2000) for lg in CONFIG["indicxnli"]],
    *[(f"indiccopa_{lg}", 5, False, 10, {"gu": 416, "hi": 417, "mr": 417}.get(lg, 468)) for lg in CONFIG["indiccopa"]],
    *[(f"indicsentiment_{lg}", 5, False, 20, 998) for lg in CONFIG["indicsentiment"]],
    *[(f"indicqa_{lg}", 5, True, 150, 500) for lg in CONFIG["indicqa"]],
]

HEAD = """\
# {note}
# Source: {repo} @ {rev} ({license}).
task: {task}
custom_dataset: !function utils.{builder}
dataset_kwargs:
  path: {repo}
  name: {name}
  revision: {rev}
  seed: "{seed}"
test_split: test
fewshot_split: fewshot
"""

MC4_YAML = HEAD + """\
output_type: multiple_choice
doc_to_text: "{{{{question.strip()}}}}\\nA. {{{{option_a}}}}\\nB. {{{{option_b}}}}\\nC. {{{{option_c}}}}\\nD. {{{{option_d}}}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

IMMLUPRO_YAML = HEAD + """\
output_type: multiple_choice
doc_to_text: !function utils.immlupro_doc_to_text
doc_to_choice: !function utils.immlupro_doc_to_choice
doc_to_target: answer_index
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

XNLI_YAML = HEAD + """\
output_type: multiple_choice
doc_to_text: "{{{{premise}}}}\\nQuestion: {{{{hypothesis}}}} True, False, or Neither?\\nAnswer:"
doc_to_choice: ["True", "Neither", "False"]
doc_to_target: label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

COPA_YAML = HEAD + """\
output_type: multiple_choice
doc_to_text: !function utils.copa_doc_to_text
doc_to_choice: "{{{{[choice1, choice2]}}}}"
doc_to_target: label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

SENTI_YAML = HEAD + """\
output_type: multiple_choice
doc_to_text: "Review: {{{{review}}}}\\nSentiment:"
doc_to_choice: ["Negative", "Positive"]
doc_to_target: label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

QA_YAML = HEAD + """\
output_type: generate_until
doc_to_text: "Passage: {{{{context}}}}\\n\\nQuestion: {{{{question}}}}\\n\\nAnswer:"
doc_to_target: '{{{{answers["text"][0]}}}}'
process_results: !function utils.process_results_qa
target_delimiter: ' '
generation_kwargs:
  until:
    - "\\n"
  do_sample: false
  temperature: 0.0
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
  - metric: f1
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

BODY = {"gmmlu": MC4_YAML, "include": MC4_YAML, "milu": MC4_YAML, "immlupro": IMMLUPRO_YAML,
        "indicxnli": XNLI_YAML, "indiccopa": COPA_YAML, "indicsentiment": SENTI_YAML, "indicqa": QA_YAML}

NOTE = {
    "gmmlu": "Global-MMLU {name}: seeded random 2000 of the 14042 test rows; exemplars = the official dev split.",
    "include": "INCLUDE {name}: full test split; exemplars = validation rows not also in test "
               "(seeded held-out test rows fill the pool when validation is mostly a copy of test).",
    "milu": "MILU {name}: seeded random 500 of the test rows (option1..4 -> A..D); exemplars = seeded 64 validation rows.",
    "immlupro": "IndicMMLU-Pro {name}: seeded random 500 of the test rows, 3..10 options per row; "
                "exemplars = seeded 64 held-out test rows.",
    "indicxnli": "IndicXNLI {name} (forward translation): seeded random 2000 of the 5010 test rows, MNLI "
                 "True/Neither/False form; exemplars = seeded 64 dev rows.",
    "indiccopa": "IndicCOPA {name}: the test split minus 32 seeded held-out rows used as exemplars (no dev split exists).",
    "indicsentiment": "IndicSentiment {name}: labelled test rows, label 0/1 = Negative/Positive; "
                      "exemplars = seeded 64 validation rows.",
    "indicqa": "IndicQA {name}: seeded random 500 answerable rows (empty-gold rows dropped); "
               "exemplars = seeded 64 short held-out rows.",
}


def main():
    rows = []
    for task, nshot, chat, cost, n_docs in TASKS:
        family, lg = task.rsplit("_", 1)
        repo, rev, lic, builder = SOURCES[family]
        name = CONFIG[family][lg]
        # The seed string is part of the task definition: it fixes the published subsets.
        seed = f"{SEED}-{task}_local"
        text = BODY[family].format(task=task, repo=repo, rev=rev, license=lic, builder=builder, name=name,
                                   seed=seed, note=NOTE[family].format(name=name))
        with open(os.path.join(HERE, f"{task}.yaml"), "w") as fh:
            fh.write(text)
        rows.append({"task": task, "label": task, "num_fewshot": nshot, "chat": chat, "gen_kwargs": None,
                     "cost_secs": cost, "n_docs": n_docs, "source": f"{repo}@{rev}", "license": lic})
    with open(os.path.join(HERE, "TASKS.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
        fh.write("\n")
    print(f"{len(rows)} tasks written to {HERE}")


if __name__ == "__main__":
    main()
