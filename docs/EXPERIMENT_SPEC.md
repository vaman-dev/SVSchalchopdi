# Experiment 06: Ethics and Responsible AI

## Aim
Study Responsible AI principles and practically examine dataset representation,
group fairness, explainability, ethical risks, and autonomous-system safety.

## Objectives
- Identify representation imbalance in a dataset.
- Evaluate a classifier separately for different groups.
- Explain why a group metric difference requires investigation rather than an
  automatic conclusion of unfairness.
- Train an explainable classifier on the Iris dataset.
- Visualize decision-tree feature importance.
- Review ethical issues in an AI case study and autonomous systems.

## Required libraries and data
- Python 3
- NumPy, Pandas, Matplotlib, and Scikit-learn
- A reproducible synthetic grouped classification dataset for Parts A and B
- The Iris dataset bundled with Scikit-learn for Part C

## Required procedure
### Part A - Bias identification
1. Create a dataset containing two groups with intentionally unequal sample
   counts.
2. Count the samples and calculate percentage representation for each group.
3. Identify the least-represented group.
4. Plot group sample counts.

### Part B - Fairness evaluation
1. Split the grouped data into isolated training and test partitions.
2. Keep the group column for auditing, but do not use it as a model feature.
3. Train a classification model using the training data only.
4. Calculate test accuracy separately for each group.
5. Plot group accuracies and report the absolute accuracy gap.
6. State that the gap is a screening signal that needs contextual analysis.

### Part C - Explainable AI
1. Load the Iris dataset.
2. Train a Decision Tree classifier using a stratified train/test split.
3. Report its held-out test accuracy.
4. Extract and plot the model's feature importances.

### Part D - Responsible AI case study
Print a structured AI recruitment case study covering bias, fairness, privacy,
safety, explainability, accountability, and human involvement.

### Part E - Autonomous systems
Print a structured review of safety, reliability, accountability, privacy,
security, and meaningful human control.

## Required deliverable
- `Experiment_06_ethics.py`
- Printed group counts and percentages
- Group-representation bar chart
- Group-wise held-out accuracy and accuracy-gap chart
- Iris Decision Tree test accuracy and feature-importance chart
- Responsible-AI and autonomous-systems discussion

The program must be a standalone Python file that can run locally or be pasted
into one Google Colab cell. Measured results must not be reported until the
program is executed.
