# Security

Report a security problem privately to research@zenteiq.ai, not in a public issue. Include the
version or commit, what you observed, and the steps to reproduce it.

This covers the code in this repository. Three things it does by design run code on your machine,
so use them only with models you trust:

* the `code` suite executes programs written by the model under evaluation (HumanEval/MBPP
  protocol); skip it with `--no-code-execution`;
* the `tool_use` suite's BFCL multi-turn task executes the model's tool calls against the
  benchmark's simulated APIs (upstream's in-process classes: a file system, a trading bot, a
  message service and so on), through upstream's `eval`-based executor and checker. brahmai-eval
  hands them only calls whose name is a documented function of the entry's APIs and whose argument
  names are plain identifiers; anything else the model writes is refused without being executed.
  The simulated APIs themselves never touch the real file system or network;
* `--trust-remote-code` runs code from a model's own repository.

Problems in a benchmark dataset or a model hosted elsewhere belong with its maintainers.
