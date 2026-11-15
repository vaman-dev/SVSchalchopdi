# PROGRESS.md

## Current Experiment
Experiment 04 - DNN Image Classification Using MNIST

## Current Status
Standalone implementation, local runtime validation, and two independent
read-only reviews are complete. Direct Google Colab execution remains pending.

## Completed Work
- Reviewed the supplied Experiment 04 requirements and the previous Git diff.
- Replaced the active experiment specification, plan, and architecture with
  Experiment 04 content.
- Implemented `experiment_04_dnn.py` as a standalone TensorFlow/Keras program.
- Used a seeded 54,000/6,000 training-validation split derived only from the
  MNIST training data.
- Built the required Flatten, Dense(128), Dense(64), and Dense(10) DNN.
- Completed five epochs of CPU training and final test evaluation.
- Verified one selected-image prediction and the visualization code path.
- Completed independent code and validation reviews; neither found a confirmed
  code defect.
- Preserved `experiment_02_ann.py` unchanged from `HEAD`.

## Files Implemented
- `experiment_04_dnn.py` (new standalone program)
- `EXPERIMENT_SPEC.md` (active Experiment 04 requirements)
- `PLAN.md` (Experiment 04 milestones and status)
- `ARCHITECTURE.md` (Experiment 04 design)
- `PROGRESS.md` (actual execution record)
- `README.md` (Experiment 04 setup and run information)

## Tests Performed
- Inspected `git diff`, `git status`, and commit `34eb569` before starting.
  - The working tree initially contained only the interrupted deletion of the
    four replaceable experiment documents.
  - `experiment_02_ann.py` was not modified.
- `env\Scripts\python.exe -m py_compile experiment_04_dnn.py`
  - Passed with exit code 0.
- Imported the module and called `build_model()` using the retained environment.
  - Input shape: `(None, 28, 28)`.
  - Output shape: `(None, 10)`.
  - Layers: Flatten, Dense, Dense, Dense.
  - Optimizer: Adam.
  - Loss: sparse categorical cross-entropy.
- `$env:KERAS_HOME='<workspace>\env\.keras'; $env:MPLBACKEND='Agg';`
  `env\Scripts\python.exe experiment_04_dnn.py`
  - Downloaded MNIST into the ignored local environment cache.
  - Completed five epochs and exited with code 0.
  - Dataset shape, split-disjointness, normalization, model-shape, prediction-
    shape, and Softmax-sum assertions passed.
- Independent reviewer checks:
  - Source parsing and imports passed.
  - Synthetic `(2, 28, 28)` forward pass returned `(2, 10)` probabilities;
    rows were finite, nonnegative, and summed to 1.0.
  - Synthetic `train_on_batch` with integer labels passed.
  - Headless plotting check passed with the expected non-interactive warning.
  - `git diff --check` found no whitespace errors.
  - Both reviewers confirmed no diff for `experiment_02_ann.py`.

## Results
- Environment: Python 3.10.11, TensorFlow 2.21.0, NumPy 2.2.6,
  Matplotlib 3.10.9, CPU execution on Windows.
- Final training accuracy: 0.9810
- Final training loss: 0.0636
- Final validation accuracy: 0.9658
- Final validation loss: 0.1129
- Test accuracy: 0.9682
- Test loss: 0.1029
- Selected image index: 0
- Predicted digit: 7
- Actual digit: 7
- Prediction confidence: 0.9999

These measurements are from one seeded run and can vary slightly across
TensorFlow versions, platforms, and hardware.

## Review Report
### Confirmed bugs
None found by either independent reviewer.

### Possible risks / limitations
- Direct execution in Google Colab has not been performed.
- MNIST requires network access on its first download.
- Runtime checks use `assert`, which is disabled if Python runs with `-O`.
- Missing dependencies or download failures raise standard Python exceptions
  rather than custom beginner-facing error messages.
- Native Windows TensorFlow 2.21 uses CPU; its GPU warning is informational.
- Automated visualization used Matplotlib's non-interactive `Agg` backend, so
  the figure code executed but no GUI window opened during validation.

## Errors / Blockers
No local execution blocker and no confirmed implementation bug remain. Google
Colab execution is the only unverified acceptance item.

## Next Action
Paste `experiment_04_dnn.py` into one Google Colab cell and run it. If it
completes, check the remaining Colab item in `PLAN.md`, record the result here,
then commit Experiment 04 before replacing these documents for a new experiment.

## Handoff Rules
Before context compaction, replace this status with actual:
- Milestones completed
- Files changed
- Commands run and their results
- Measured metrics
- Outstanding work
- Exact next action
