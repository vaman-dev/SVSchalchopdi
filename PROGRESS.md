# PROGRESS.md

## Current Experiment
Experiment 2 — ANN Classification

## Current Status
Local implementation, runtime verification, and independent review complete.
Direct execution in Google Colab has not been performed.

## Completed Work
- Replaced the MNIST-specific Codex scaffold with reusable lab instructions.
- Replaced three specialized agents with generic `coder` and `reviewer` agents.
- Replaced two MNIST-specific skills with `lab-implementation` and
  `lab-validation`.
- Removed the obsolete package, tests, notebook, docs, artifacts scaffold,
  packaging file, README, and separate workflow file requested for removal.
- Implemented the standalone `experiment_02_ann.py` program.
- Installed the required libraries into the retained `env/` environment.
- Completed a five-epoch CPU training run and independent read-only review.
- Added `README.md` with virtual-environment setup, complete dependency
  installation, local execution, and Google Colab instructions.
- Removed the existing `env/` contents from Git tracking while preserving the
  local virtual environment; `.gitignore` continues to exclude it.

## Files Implemented
- `experiment_02_ann.py`
- `AGENTS.md`
- `PLAN.md`
- `ARCHITECTURE.md`
- `PROGRESS.md`
- `EXPERIMENT_SPEC.md`
- `.gitignore`
- `requirements.txt`
- `README.md`
- `.codex/agents/coder.toml`
- `.codex/agents/reviewer.toml`
- `.agents/skills/lab-implementation/SKILL.md`
- `.agents/skills/lab-validation/SKILL.md`

## Tests Performed
- `env\Scripts\python.exe --version`
  - Observed Python 3.10.11.
- `env\Scripts\python.exe -c "import ast, pathlib; ast.parse(...)"`
  - AST syntax check passed for `experiment_02_ann.py`.
- `env\Scripts\python.exe -m pip install -r requirements.txt`
  - The first sandboxed attempt could not access the network.
  - The approved network-enabled retry completed successfully.
- `$env:KERAS_HOME='C:\Users\Apoorva\OneDrive\Desktop\checking\runtime_cache';`
  `$env:MPLBACKEND='Agg';`
  `env\Scripts\python.exe experiment_02_ann.py`
  - Exit code 0 after downloading MNIST and completing five epochs.
  - Runtime assertions for split sizes, disjoint training/validation indices,
    model output shape, prediction shape, and probability sum passed.
- Independent static validation and code review found no confirmed ANN logic
  defects.

## Results
- Environment: Python 3.10.11, TensorFlow 2.21.0, NumPy 2.2.6,
  Matplotlib 3.10.9, CPU execution on Windows.
- Final training accuracy: 0.9810
- Final training loss: 0.0636
- Final validation accuracy: 0.9658
- Final validation loss: 0.1129
- Test accuracy: 0.9682
- Test loss: 0.1029
- Sample predicted digit: 7
- Sample actual digit: 7
- Prediction confidence: 0.9999

These values are from one seeded run and can vary slightly by TensorFlow
version and hardware.

## Errors / Blockers
- The active retained environment uses Python 3.10.11 rather than the preferred
  Python 3.11, but the complete program ran successfully.
- The verification run used Matplotlib's non-interactive `Agg` backend, so the
  plot function executed but emitted a warning that the window could not be
  shown. Normal local interactive backends and Colab execute `plt.show()`.
- Google Colab execution remains unverified.
- Tracked deletions of older experiment scripts and images were already present
  before this task and were intentionally left untouched.

## Next Action
Paste `experiment_02_ann.py` into one Google Colab cell and run it to complete
the remaining Colab verification checkbox. Then review and commit this completed
experiment before replacing the experiment-specific documents for Experiment 3.

## Handoff Rules
Before context compaction, replace this status with actual:
- Milestone completed
- Files changed
- Commands run and their results
- Measured metrics
- Outstanding work
- Exact next action
