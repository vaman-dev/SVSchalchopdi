# PROGRESS.md

## Current Experiment
Experiment 03 - Performance Analysis of Activation Functions

## Current Status
Standalone implementation, local execution, numerical edge-case testing,
plotting-path validation, and code review are complete. Commit/push and direct
Google Colab execution remain pending.

## Completed Work
- Reviewed the supplied aim, objectives, theory, comparative analysis,
  procedure, expected output, result, and conclusion.
- Created `experiment_03_activation_functions.py` without modifying any prior
  experiment script.
- Implemented ReLU with `numpy.maximum`.
- Implemented Sigmoid with `logaddexp` to avoid exponential overflow.
- Implemented Tanh with `numpy.tanh`.
- Implemented stable Softmax using maximum subtraction and axis-aware
  normalization for vectors and batches.
- Printed a common-input comparison and verified the Softmax sum.
- Added four behavior visualizations: three scalar curves and a sample Softmax
  class-probability chart.
- Added repeatable NumPy timing for the same 100,000-value array and a timing
  chart with explicit interpretation limits.
- Printed output ranges, suitable applications, advantages, and limitations.
- Updated the active specification, plan, architecture, and README.

## Files Implemented
- `experiment_03_activation_functions.py` (new standalone program)
- `docs/EXPERIMENT_SPEC.md` (Experiment 03 requirements)
- `docs/PLAN.md` (completed local milestones and remaining handoff tasks)
- `docs/ARCHITECTURE.md` (numerical and timing design)
- `docs/PROGRESS.md` (verified execution record)
- `docs/README.md` (Experiment 03 setup and usage instructions)

The existing dependency manifest already contained NumPy and Matplotlib, so no
new package was added.

## Tests Performed
- `env\Scripts\python.exe -m py_compile experiment_03_activation_functions.py`
  - Passed with exit code 0 after the validation correction.
- `env\Scripts\python.exe -m pip check`
  - Reported no broken requirements.
- `git diff --exit-code -- experiment_02_ann.py experiment_04_dnn.py experiment_05_cnn_features.py Experiment_06_ethics.py`
  - Passed; every earlier experiment script remains unchanged.
- Imported the module and tested inputs `[-1000, -1, 0, 1, 1000]`.
  - Sigmoid, Tanh, and Softmax outputs were finite.
  - ReLU removed all negative values.
  - One-dimensional Softmax summed to 1.
  - Both rows of a two-dimensional, axis-aware Softmax summed to 1.
- Ran the complete program with `MPLBACKEND=Agg`.
  - Output comparison, mathematical assertions, two plotting paths, timing,
    and application/limitation reporting completed with exit code 0.

## Results
### Common input values
Input: `[-5, -2, -1, 0, 1, 2, 5]`

- ReLU: `[0, 0, 0, 0, 1, 2, 5]`
- Sigmoid: `[0.006693, 0.119203, 0.268941, 0.500000, 0.731059,
  0.880797, 0.993307]`
- Tanh: `[-0.999909, -0.964028, -0.761594, 0, 0.761594, 0.964028,
  0.999909]`
- Softmax: `[0.000042, 0.000846, 0.002299, 0.006249, 0.016986,
  0.046173, 0.927406]`
- Softmax probability sum: `1.000000`

### Observed NumPy timing
Environment: Python 3.10.11, NumPy 2.2.6, Matplotlib 3.10.9 on Windows.
Best average time per call on a 100,000-value array:

- ReLU: `0.0593 ms`
- Sigmoid: `5.3819 ms`
- Tanh: `0.8095 ms`
- Softmax: `1.5645 ms`

These values are measurements from one local run and will vary with hardware,
operating-system load, Python/NumPy versions, array size, and timing method.
Softmax additionally performs reductions and vector normalization, so its
timing is not directly equivalent to the element-wise functions.

## Review Report
### Confirmed issue found and fixed
The first extreme-input check reused a strict assertion that every Sigmoid
result must be greater than zero and less than one. Floating-point arithmetic
correctly rounds Sigmoid(-1000) to 0 and Sigmoid(1000) to 1. The validation was
corrected to accept the closed floating-point range `[0, 1]` while retaining
finite-output and Sigmoid(0) checks. The edge-case retest then passed.

### Remaining limitations
- The timing case study measures NumPy implementations, not GPU kernels or
  complete neural-network training performance.
- Softmax operates across a vector while the other functions are element-wise.
- Timing does not measure gradient behavior or model accuracy.
- The non-interactive backend executed plot creation but did not visually
  inspect a GUI window.
- Direct Google Colab execution has not been performed.

## Errors / Blockers
No local blocker remains. The Matplotlib warnings only stated that the
intentionally non-interactive `Agg` backend cannot open display windows.

## Next Action
Commit and push the verified Experiment 03 changes to `origin/main`. Then run
the script in one Google Colab cell to confirm interactive rendering.

## Handoff Rules
Before context compaction, replace this status with actual:
- Milestones completed
- Files changed
- Commands run and observed results
- Measured metrics
- Outstanding work
- Exact next action
