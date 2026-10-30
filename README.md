<p align="center"><img src="assets/x.png" alt="X mark" width="110"></p>

<h1 align="center">BrahmAI Eval</h1>

<p align="center">by ZenteiQ AiTech Innovations Pvt. Ltd.</p>

<p align="center">
<a href="https://github.com/zenteiq-research/brahmai-eval/releases">Releases</a> ·
<a href="https://github.com/zenteiq-research/brahmai-eval/issues">Issues</a> ·
<a href="https://github.com/zenteiq-research/brahmai-eval/discussions">Discussions</a> ·
<a href="CITATION.cff">Cite</a> ·
<a href="#acknowledgements">Acknowledgements</a>
</p>

Evaluate a language model on open benchmarks, in one command, reproducibly.

Give this tool a Hugging Face checkpoint (a Hub id or a local directory) and it runs the model on
228 tasks across English understanding, code, mathematics, Indian languages, Indic generation and
tool use, with vLLM doing the inference and the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the official
[BFCL](https://github.com/ShishirPatil/gorilla) checkers doing the scoring. Every dataset comes from
a public source (the tasks defined here pin a dataset revision; the tasks taken unchanged from the
harness load what lm-eval 0.4.12 names), every run writes its full provenance, and every scored
item is kept, so a published number can be checked and reproduced.

It was built to evaluate Dhīr, ZenteiQ's models, which are not yet released. The models and their
results are not part of this repository.

*Developed under the BrahmAI project, supported by the IndiaAI Mission (see
[Acknowledgements](#acknowledgements)).*

**Contents:**
[Quick start](#quick-start) ·
[Naming a model](#naming-a-model) ·
[Benchmarks](#benchmarks) ·
[Output](#output) ·
[Reproducible runs](#reproducible-runs) ·
[Robustness](#robustness) ·
[Docker](#docker) ·
[Customisation](#customisation) ·
[Adding a benchmark](#adding-a-benchmark) ·
[Development](#development) ·
[Licence and citation](#licence-and-citation) ·
[Acknowledgements](#acknowledgements)

## Quick start

Requires Linux, one or more NVIDIA GPUs (CUDA 12.8-capable driver), Python 3.10–3.12, a C compiler
on `PATH` (`apt install build-essential`; vLLM compiles kernels at start-up) and internet access on
the first run.

```bash
git clone https://github.com/zenteiq-research/brahmai-eval.git
cd brahmai-eval
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # pins vllm 0.25.1 and lm-eval 0.4.12
hf auth login                          # optional: only the gated MILU dataset needs it

brahmai-eval run --model Qwen/Qwen2.5-1.5B-Instruct            # every benchmark, every visible GPU
brahmai-eval run --model /path/to/checkpoint --suite code,math  # a subset, a local checkpoint
```

A run prints one line per task as it finishes and ends with the path of `results.md`. Other
commands: `brahmai-eval list` (suites and tasks; `-v` for every task), `brahmai-eval prepare`
(download all data up front, e.g. for an offline machine), `brahmai-eval summarize <run dir>`
(rebuild the result tables). Add `--help` to any of them.

### Naming a model

| You pass | Loaded as |
|---|---|
| `org/name` (optionally `--revision`) | downloaded from the Hugging Face Hub into the HF cache |
| a directory | a local checkpoint: `config.json`, safetensors weights, tokenizer |

The checkpoint's own chat template and `eos_token_id` are used as they are; nothing is rewritten.
Models whose repository ships custom code need `--trust-remote-code`. A short preflight proves the
model can be prompted the way the suite prompts it (a chat template exists, a greedy reply stops on
a declared stop token) and aborts with a specific message otherwise.

## Benchmarks

All sources are public. The 191 tasks defined in this repository record their dataset revision and
licence in `tasks/*/TASKS.json`; the 34 tasks taken unchanged from lm-eval 0.4.12 are listed in
`brahmai_eval/suites.py`, and the 3 tool-use tasks pin their sources in `brahmai_eval/datasets/`.
`brahmai-eval list -v` prints each task with its shot count and whether it goes through the chat
template.

| Suite | Tasks | Protocol |
|---|---|---|
| `english` | HellaSwag (10-shot), MMLU (5), ARC-Challenge (25), ARC-Easy (5), PIQA, Winogrande, SciQ, CommonsenseQA, OpenBookQA, BoolQ, TruthfulQA-MC2, CoQA, TriviaQA (5), Global-MMLU English (5), DROP, IFEval | log-likelihood multiple choice; DROP and IFEval generate through the chat template |
| `code` | HumanEval, MBPP, HumanEval+, MBPP+ | pass@1, greedy; the model's programs are executed locally |
| `math` | GSM8K (5-shot), GSM-Plus, AIME 2025, MATH (Minerva 4-shot, 7 subjects), MGSM Bengali and Telugu | exact match on the extracted answer |
| `mmlu_pro` | MMLU-Pro, 14 categories (5-shot chain of thought) | exact match |
| `indic` | Belebele in 21 Indic language/script variants, XNLI hi/ur/en, XStoryCloze hi/te/en, XQuAD hi/en | multiple choice, extractive QA |
| `indic_ext` | Global-MMLU (bn, hi, ne, si, te, en), INCLUDE (7 languages), MILU (11), IndicMMLU-Pro (9), IndicXNLI (11), IndicCOPA (19), IndicSentiment (14), IndicQA (11) | 5-shot; the larger sets are fixed seeded samples (500 or 2,000 items), identical on every run |
| `igb` | IndicGenBench: XQuAD-IN, XOR-QA-IN, Flores-IN (both directions), CrossSum-IN; 12 Indic languages plus English | F1 / EM, chrF++ |
| `tool_use` | BFCL v3 single-turn (11 Python categories, official AST checker), BFCL v3 multi-turn (base, missing function, missing parameter, long context; official state checker), When2Call (MCQ macro-F1, tool hallucination, call recall and false-call rates) | tools are declared through the model's chat template |
| `all` | everything above (228 tasks) | |

`--suite` takes suite names and task labels mixed: `--suite code,gsm8k,belebele_hin_Deva`.

## Output

Each run writes `results/<model>/<timestamp>/`:

| File | Content |
|---|---|
| `results.md`, `results.json` | one row per task (primary metric, every metric, item count, seconds) and per-suite means |
| `plan.json` | the exact task list and shard assignment, engine settings, model checksums (MD5 of the configuration and tokenizer files; weight files by size), software versions, GPUs, brahmai-eval version, and a `config_hash` over all of it |
| `tasks/` | the task definitions this run used |
| `shards/<k>/samples/<task>.jsonl` | every scored item: rendered prompt, model output, target, per-item metrics |
| `shards/<k>/log.txt`, `engine.json` | worker log with every attempt, effective engine settings, preflight record |
| `manifest.json` | status, duration, task errors |

A run with task errors exits 1 and names them; `--resume results/<model>/<timestamp>` (with the
same `--model`) re-runs only what is missing. `--limit N` scores N items per task for a smoke test
(the first N of each lm-eval task, N evenly spaced items of each tool-use task); its results are
marked as such.

## Reproducible runs

Greedy decoding, seed 1234 everywhere, pinned engine and harness versions, pinned dataset revisions,
and a `config_hash` that covers the task list, engine settings and model file checksums. Two runs
with the same hash on the same hardware class reproduce each other to within vLLM's batching noise
(about ±0.005 on a 1,000-item accuracy). Numbers taken on a different inference engine or engine
version can differ by around 0.01 on generative tasks; re-run the baseline rather than compare
across engines.

To reproduce a published number: install the pinned versions, run the same `--suite` with the
default settings (`--max-model-len 8192 --seed 1234`), and compare `plan.json`'s `config_hash` and
model checksums with the published run.

## Robustness

Correct results matter more than speed, so the scheduler degrades on its own:

* **One GPU is enough.** With one visible GPU everything runs in one engine, task after task.
  `--serial` forces that layout on a multi-GPU machine.
* **Engine and memory failures fall back automatically.** Tasks are split across GPUs by estimated
  cost, one engine per GPU. A shard whose worker dies (engine failed to start, CUDA out of memory,
  killed) is retried on an exclusive GPU with a smaller batch, then once more with the most
  conservative engine settings (eager mode, four sequences at a time). Scored tasks are never
  repeated; `shards/<k>/log.txt` records every attempt.
* **One bad task does not sink the run.** A task that raises is recorded as an error, the others
  continue, and the exit code and `results.md` name it.
* **Two engines per GPU** (`--jobs-per-gpu 2`) is an opt-in optimisation for small models; engines
  sharing a GPU start one after the other so each sizes its cache from the memory that is really
  free. Tool-use shards get a 16k context window automatically.

## Docker

The image is built from the public `vllm/vllm-openai:v0.25.1` image and PyPI; nothing is pulled from
a private registry.

```bash
docker build -t brahmai-eval .
docker run --rm --gpus all --ipc=host -v $PWD/results:/work/results -v $HOME/.cache:/root/.cache \
    brahmai-eval brahmai-eval run --model Qwen/Qwen2.5-1.5B-Instruct
```

## Customisation

* **Chat template** — `--chat-template my_template.jinja` replaces the tokenizer's template for the
  run (the weights are untouched; a hard-linked copy of the small files is made under
  `~/.cache/brahmai-eval/model-overrides`).
* **Stop tokens** — if the preflight reports that replies do not stop, the template's end-of-turn
  token is not declared as an `eos_token_id`: `--stop-tokens '<|im_end|>'` adds it (several: comma
  separated).
* **Base models** — raw suites need no chat template; select tasks explicitly, e.g.
  `--suite hellaswag,mmlu,arc_challenge,indic`. `--add-bos` prefixes the BOS token to raw prompts
  for models that expect it.
* **System prompt** — `--system-instruction "..."` is inserted into every chat prompt (default none).
* **Tool-call format** — `--tool-format json` (tools declared through the template's `tools=`
  argument; calls read as JSON inside `<tool_call>` tags or as a bare JSON object; templates that
  render calls with `<arg_key>` tags are parsed too) or `python` (BFCL's prompting convention:
  function docs in the system prompt, calls as `[func(arg=val)]`); `auto` picks `json` when the
  template mentions tools. Every item's rendered prompt, raw output and parse are in the samples.
* **Throughput** — `--jobs-per-gpu 2` for models under ~4B parameters; `--tensor-parallel 2` to
  split a large model over GPUs; `--max-model-len` (default 8192; longer prompts are left-truncated,
  tool-use items that do not fit are counted as `context_overflow`), `--gpu-memory-utilization`,
  `--max-num-seqs`, `--dtype` go straight to vLLM.
* **Code execution** — the `code` suite runs model-generated Python on the machine;
  `--no-code-execution` skips it. The BFCL multi-turn task executes the model's tool calls against
  the benchmark's simulated APIs (an in-process file system, trading bot and so on), restricted to
  the documented functions; see [SECURITY.md](SECURITY.md).
* **Data locations** — `BRAHMAI_EVAL_HOME` (default `~/.cache/brahmai-eval`) holds downloaded data;
  `HF_HOME` controls the Hugging Face cache. `brahmai-eval prepare --hub` fetches everything for an
  offline machine.

## Adding a benchmark

Drop an lm-eval task YAML into `tasks/<family>/` and add a row to that family's `TASKS.json`
(`label`, `task`, `num_fewshot`, `chat`, `cost_secs`, `source`, `license`). Data must come from a
public source: a Hub id with a pinned `revision` in the YAML, or a downloader in
`brahmai_eval/datasets/<name>.py` with `prepare(data_dir)` and the `__DATA_DIR__` placeholder in the
YAML. A function-calling benchmark is a module in `brahmai_eval/tooluse/` registered in `TASKS`.

## Development

```bash
pip install -e ".[dev]"
ruff check brahmai_eval tests
pytest -q                 # no GPU, no network beyond the one-time tool-use data download
```

The tests cover suite resolution, work partitioning, the scheduler's retry ladder (with a fake
worker), result merging, task rendering, the task tables (unique labels, pinned sources), and the
tool-use parsers and scorers on fixtures and gold replays.

## Licence and citation

| | Licence |
|---|---|
| **Code** | Apache-2.0 (`LICENSE`, `NOTICE`) |
| **Data** | Not covered by the code licence. Each benchmark keeps its own licence and attribution, recorded per task in `tasks/*/TASKS.json`. |

**Data notes**

* IndicGenBench CrossSum-IN is CC-BY-NC-SA-4.0 (non-commercial); XQuAD-IN and Flores-IN are
  CC-BY-SA-4.0.
* `ai4bharat/MILU` is gated on the Hub: accept its terms and log in, or its 11 tasks report a
  "gated dataset" error and everything else still runs.
* A few Hub datasets state no licence on their card; the table says so.
* When you report results on a benchmark, cite its original paper.

**Citing this software.** Use `CITATION.cff` (GitHub's "Cite this repository" button reads it).

**Questions and contributions**

* Issues are welcome for bug reports and benchmark suggestions. Use Discussions for questions and
  ideas.
* We do not accept pull requests at this time.
* Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
* Security problems: see [SECURITY.md](SECURITY.md).
* Contact: research@zenteiq.ai.

## Acknowledgements

This tool was developed as part of the BrahmAI project, supported by the IndiaAI Mission. ZenteiQ acknowledges the support of the IndiaAI Mission, Ministry of Electronics and Information Technology (MeitY), under the Foundation Model initiative.

### Benchmarks and how to cite them

The benchmarks are the work of others. Thanks to their creators and maintainers, and please cite
them when you report results on their data.

| Benchmark | Creators |
|---|---|
| HellaSwag, MMLU, MMLU-Pro, ARC, PIQA, Winogrande, SciQ, CommonsenseQA, OpenBookQA, BoolQ, TruthfulQA, CoQA, TriviaQA, DROP, IFEval, GSM8K, GSM-Plus, MATH, AIME, MGSM, HumanEval, MBPP | their respective authors, run through the EleutherAI lm-evaluation-harness |
| HumanEval+, MBPP+ | EvalPlus |
| Belebele, XNLI, XStoryCloze, XQuAD | Meta AI, Google and the dataset authors |
| Global-MMLU, INCLUDE | Cohere Labs and collaborators |
| MILU, IndicCOPA, IndicSentiment, IndicQA, IndicXNLI | AI4Bharat (IndicXNLI: Divyanshu Aggarwal et al.) |
| IndicMMLU-Pro | the LinguaLift authors |
| IndicGenBench | Google Research |
| BFCL v3 | the Gorilla team, UC Berkeley |
| When2Call | NVIDIA |
