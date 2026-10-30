# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copied from / adapted from the lm-evaluation-harness v0.4.12 (lm_eval/tasks/xquad/utils.py), Copyright (c) 2020 EleutherAI,
# MIT License: see tasks/LICENSE-lm-evaluation-harness.
import transformers.data.metrics.squad_metrics as squad_metrics


def process_results_qa(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    f1_sum = squad_metrics.compute_f1(reference, preds)
    exact_match = squad_metrics.compute_exact(reference, preds)
    return {"f1": f1_sum, "exact_match": exact_match}
