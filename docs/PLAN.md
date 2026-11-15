# PLAN.md

## Active Experiment
Experiment 06 - Ethics and Responsible AI

## Goal
Create a reproducible standalone program that demonstrates dataset imbalance,
group-wise model evaluation, decision-tree explainability, and structured
ethical analysis.

## Milestone 1 - Requirements
- [x] Review the Experiment 06 requirements
- [x] Select appropriate datasets and models
- [x] Confirm required dependencies and expected outputs

## Milestone 2 - Architecture
- [x] Design the reproducible grouped synthetic dataset
- [x] Define leakage-free training and group auditing
- [x] Define the Iris Decision Tree explanation workflow
- [x] Define visualizations and ethical case-study output

## Milestone 3 - Implementation
- [x] Create `Experiment_06_ethics.py`
- [x] Implement representation analysis and visualization
- [x] Implement group-wise held-out fairness evaluation
- [x] Implement Iris feature importance and visualization
- [x] Add responsible-AI and autonomous-system discussions
- [x] Update dependency and usage documentation

## Milestone 4 - Verification
- [x] Check Python syntax and imports
- [x] Verify representation percentages sum to 100%
- [x] Verify the group attribute is excluded from model inputs
- [x] Verify train/test isolation and group metrics
- [x] Verify feature importances are valid
- [x] Run the complete program and record observed results

## Milestone 5 - Review and Handoff
- [x] Review correctness and requirement coverage
- [x] Fix confirmed issues
- [x] Update `docs/PROGRESS.md`
- [x] Report local verification status
- [ ] Run directly in Google Colab

## Rules
Complete milestones sequentially. Check boxes only after completing the work.
Treat fairness metrics as diagnostic evidence, not automatic proof of fairness
or discrimination. Record failures and unverified behavior honestly.
