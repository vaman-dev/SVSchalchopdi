# ARCHITECTURE.md

## Experiment
Experiment 06 - Ethics and Responsible AI

## Approach
A standalone educational script combines two small machine-learning exercises
with written ethical analysis. A reproducible synthetic dataset demonstrates
group representation and performance auditing. Scikit-learn's bundled Iris
dataset demonstrates interpretable feature importance with a Decision Tree.

## Data flow
```text
Seeded synthetic grouped data
    |
    +--> group counts and percentages --> representation bar chart
    |
    `--> stratified train/test split
             |
             v
       numeric features only (group excluded)
             |
             v
       Logistic Regression
             |
             v
       held-out predictions by group
             |
             `--> group accuracy + absolute gap + comparison chart

Scikit-learn Iris dataset
    |
    v
Stratified train/test split
    |
    v
Decision Tree classifier
    |
    +--> held-out accuracy
    `--> normalized feature importances --> bar chart

Printed analysis
    +--> AI recruitment case study
    `--> autonomous-system safety and governance checklist
```

## Synthetic dataset design
- Two audit groups have intentionally unequal sample counts.
- Two numerical predictive features and one binary target are generated with a
  fixed random seed.
- Different noise levels allow group performance to be compared without using
  the group identity as a predictor.
- A composite group-and-label key is used for stratification so each test group
  retains both target classes.

The synthetic example is an educational demonstration, not evidence about any
real population or production system.

## Model configuration
### Fairness demonstration
- Model: Logistic Regression
- Inputs: two numerical features
- Audit attribute: group, excluded from model inputs
- Evaluation: held-out accuracy per group and absolute accuracy gap
- Random seed: 42

### Explainability demonstration
- Dataset: Iris
- Model: Decision Tree, maximum depth 3
- Explanation: normalized impurity-based feature importance
- Evaluation: held-out accuracy
- Random seed: 42

## Program organization
`Experiment_06_ethics.py` contains small functions for dataset creation,
representation analysis, fairness evaluation, Iris feature importance,
visualization, case-study printing, and the main execution flow. It has no
local data-file or module dependencies.

## Verification strategy
- Compile and import the source using the existing virtual environment.
- Assert expected columns, group counts, and percentage totals.
- Assert that the group attribute is not included in model features.
- Assert train/test row isolation and valid group accuracies.
- Assert Iris feature importances are nonnegative and sum to one.
- Execute the complete script with Matplotlib's non-interactive backend.
- Inspect the implementation against all five practical-work sections.

## Constraints and interpretation
- Preserve every earlier experiment script.
- Avoid sensitive real-person data; use clearly labelled synthetic groups.
- Do not describe one metric as a complete fairness assessment.
- Impurity-based feature importance describes model behavior, not causation.
- Keep the program suitable for a single Google Colab cell.
