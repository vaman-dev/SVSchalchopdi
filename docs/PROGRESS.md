# PROGRESS.md

## Current Experiment
Experiment 05 - CNN Feature Extraction and Visualization

## Current Status
Standalone implementation, local five-epoch execution, visualization-path
validation, and code review are complete. Direct Google Colab execution remains
pending.

## Completed Work
- Reviewed the supplied Experiment 05 aim, theory, procedure, observations, and
  expected result.
- Created `experiment_05_cnn_features.py` without changing Experiments 02 or 04.
- Added a seeded, leakage-free 54,000/6,000 training-validation split and
  expanded MNIST images to `(28, 28, 1)`.
- Built a CNN with 8-filter and 16-filter convolutional layers, max pooling,
  ReLU activations, and a 10-class Softmax classifier.
- Used same padding, valid padding, pooling, and convolution stride settings
  whose effects can be compared through their output dimensions.
- Trained for five epochs and evaluated on the untouched MNIST test set.
- Extracted first-convolution, pooled, and deeper-convolution activations.
- Added visualizations for training history, both layers' learned filters, the
  selected test image, and all intermediate feature-map channels.
- Updated the active specification, plan, architecture, and README.

## Files Implemented
- `experiment_05_cnn_features.py` (new standalone program)
- `docs/EXPERIMENT_SPEC.md` (Experiment 05 requirements)
- `docs/PLAN.md` (completed milestones and pending Colab check)
- `docs/ARCHITECTURE.md` (CNN design and dimensions)
- `docs/PROGRESS.md` (verified execution record)
- `docs/README.md` (Experiment 05 setup and run instructions)

## Tests Performed
- `env\Scripts\python.exe -m py_compile experiment_05_cnn_features.py`
  - Passed with exit code 0.
- Imported the module and called `build_model()` and
  `build_feature_extractor()` with two synthetic images.
  - Class probabilities had shape `(2, 10)` and each row summed to 1.
  - Activation shapes were `(2, 28, 28, 8)`, `(2, 14, 14, 8)`, and
    `(2, 6, 6, 16)`.
- Called both learned-filter plots and all three feature-map plots using the
  non-interactive Matplotlib `Agg` backend.
  - Every visualization code path completed; expected non-interactive display
    warnings were observed.
- Ran the complete program with the retained environment, local MNIST cache,
  and `MPLBACKEND=Agg`.
  - All split, normalization, model-shape, probability, and activation-shape
    assertions passed.
  - Five training epochs, final test evaluation, prediction, activation
    extraction, and all plots completed with exit code 0.
- Reviewed data isolation, layer settings, output dimensions, filter indexing,
  feature-map indexing, and Colab portability.
  - One Keras input-structure warning found during lightweight checking was
    corrected by using the model's single input tensor.
  - No confirmed implementation bug remains.

## Results
- Environment: Python 3.10.11, TensorFlow 2.21.0, CPU execution on Windows.
- Final training accuracy: 0.9819
- Final training loss: 0.0591
- Final validation accuracy: 0.9780
- Final validation loss: 0.0650
- Test accuracy: 0.9824
- Test loss: 0.0544
- Selected image index: 0
- Predicted digit: 7
- Actual digit: 7
- Prediction confidence: 0.9997
- First convolution maps: `(1, 28, 28, 8)`
- Pooled maps: `(1, 14, 14, 8)`
- Second convolution maps: `(1, 6, 6, 16)`

These values are from one seeded local run and can vary slightly across
TensorFlow versions, platforms, and hardware.

## Review Report
### Confirmed bugs
None remain.

### Possible risks / limitations
- Direct Google Colab execution has not been performed.
- MNIST needs internet access on its first download.
- Headless validation executed all plotting code but did not visually inspect
  an interactive GUI window.
- Second-layer kernels contain weights for eight input channels; the filter
  plot displays their channel-wise mean as one compact 2-D view per filter.
- Native Windows TensorFlow 2.21 used the CPU; its GPU warning is informational.

## Errors / Blockers
No local blocker remains. Direct Colab execution is the only unverified item.

## Next Action
Review the local changes. Then paste `experiment_05_cnn_features.py` into one
Google Colab cell and run it to confirm interactive plot rendering. Commit and
push the experiment after approval.

## Handoff Rules
Before context compaction, replace this status with actual:
- Milestones completed
- Files changed
- Commands run and observed results
- Measured metrics
- Outstanding work
- Exact next action
