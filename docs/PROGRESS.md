# PROGRESS.md

## Current Experiment
Experiment 06 - Ethics and Responsible AI

## Current Status
Standalone implementation, dependency setup, local execution, plotting-path
validation, and code review are complete. Direct Google Colab execution remains
pending.

## Completed Work
- Reviewed the supplied aim, objectives, theory, practical work, and expected
  output.
- Created `Experiment_06_ethics.py` without modifying Experiments 02, 04, or 05.
- Created a seeded 500-row synthetic dataset with 400 Group A and 100 Group B
  samples and clearly labelled it as an educational example.
- Calculated group counts and percentages, identified the underrepresented
  group, and plotted sample counts.
- Split the synthetic data into isolated training and test partitions using a
  group-and-label stratification key.
- Excluded the audit group and target from model inputs, trained a scaled
  Logistic Regression model, and compared held-out accuracy by group.
- Loaded the bundled Iris dataset, trained a depth-3 Decision Tree, and plotted
  normalized impurity-based feature importance.
- Added an AI recruitment case study covering bias, fairness, privacy, safety,
  explainability, accountability, and human involvement.
- Added autonomous-system considerations covering safety, reliability,
  accountability, privacy, security, and human control.
- Added Pandas and Scikit-learn to the dependency manifest and updated the
  README installation, execution, Colab, and project-structure guidance.

## Files Implemented
- `Experiment_06_ethics.py` (new standalone program)
- `config/requirements.txt` (Pandas and Scikit-learn dependencies)
- `docs/EXPERIMENT_SPEC.md` (Experiment 06 requirements)
- `docs/PLAN.md` (completed local milestones and pending Colab check)
- `docs/ARCHITECTURE.md` (data flow, models, safeguards, and validation design)
- `docs/PROGRESS.md` (verified execution record)
- `docs/README.md` (Experiment 06 setup and usage instructions)

## Tests Performed
- `env\Scripts\python.exe -m pip install -r config\requirements.txt`
  - Installed Pandas 2.3.3 and Scikit-learn 1.7.2 into the retained `env/`.
  - Existing TensorFlow, NumPy, and Matplotlib installations were retained.
- Imported Pandas and Scikit-learn and printed their module locations.
  - Both resolved inside `env\Lib\site-packages`, not global Python.
- `env\Scripts\python.exe -m py_compile Experiment_06_ethics.py`
  - Passed with exit code 0.
- `env\Scripts\python.exe -m pip check`
  - Reported no broken requirements.
- `git diff --exit-code -- experiment_02_ann.py experiment_04_dnn.py experiment_05_cnn_features.py`
  - Passed; all earlier experiment scripts remain unchanged.
- Ran the complete program with `MPLBACKEND=Agg`.
  - Representation, split isolation, model-feature exclusion, group metrics,
    feature importance, and probability/range assertions passed.
  - All three plot code paths and Parts A through E completed with exit code 0.
- Reviewed the code against the supplied practical procedure.
  - No confirmed implementation bug remains.

## Results
- Environment: Python 3.10.11 on Windows
- Pandas: 2.3.3
- Scikit-learn: 1.7.2
- Synthetic dataset shape: `(500, 5)`
- Group A: 400 samples, 80.00% representation
- Group B: 100 samples, 20.00% representation
- Underrepresented group: Group B
- Training rows: 375
- Held-out test rows: 125
- Group A held-out accuracy: 0.8800 (`n=100`)
- Group B held-out accuracy: 0.7600 (`n=25`)
- Absolute group accuracy gap: 0.1200
- Iris Decision Tree held-out accuracy: 0.8947
- Iris feature importance:
  - Petal length: 0.9507
  - Petal width: 0.0493
  - Sepal width: 0.0000
  - Sepal length: 0.0000

These are seeded educational results, not a conclusion about a real group or
production system. A different dataset, split, model, or library version can
change the measurements.

## Review Report
### Confirmed bugs
None found.

### Interpretation limits / possible risks
- Representation imbalance and a group accuracy gap are screening signals, not
  automatic proof of discrimination or fairness.
- Accuracy alone does not capture false-positive/false-negative tradeoffs,
  calibration, intersectional effects, or application-specific harm.
- Group B has only 25 held-out samples, so its accuracy estimate is less stable.
- The synthetic group labels do not represent real demographic groups.
- Decision Tree impurity importance explains this fitted model's split usage;
  it does not prove causal influence.
- Headless validation executed plot code but did not visually inspect a GUI.
- Direct Google Colab execution has not been performed.

## Errors / Blockers
No local blocker remains. Direct Colab execution is the only unverified item.
The Matplotlib warnings during validation only stated that the intentionally
non-interactive `Agg` backend cannot open a window.

## Next Action
Review the local Experiment 06 changes. Then paste `Experiment_06_ethics.py`
into one Google Colab cell and run it to confirm interactive chart rendering.
Commit and push the experiment after approval.

## Handoff Rules
Before context compaction, replace this status with actual:
- Milestones completed
- Files changed
- Commands run and observed results
- Measured metrics
- Outstanding work
- Exact next action
