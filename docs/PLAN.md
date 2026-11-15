# PLAN.md

## Active Experiment
Experiment 05 - CNN Feature Extraction and Visualization

## Goal
Train a simple MNIST CNN and visualize learned filters and intermediate feature
maps so that changes between early and deeper feature extraction are clear.

## Milestone 1 - Requirements
- [x] Review the Experiment 05 requirements
- [x] Identify the dataset, CNN operations, and required visualizations
- [x] Confirm TensorFlow, NumPy, and Matplotlib cover the dependencies

## Milestone 2 - Architecture
- [x] Design leakage-free preprocessing and channel expansion
- [x] Define two convolutional stages and the classifier
- [x] Define intermediate activation extraction and visualization
- [x] Define verification criteria for dimensions and probabilities

## Milestone 3 - Implementation
- [x] Create `experiment_05_cnn_features.py`
- [x] Load, split, normalize, and reshape MNIST
- [x] Build, compile, train, and evaluate the CNN
- [x] Predict a selected test image
- [x] Extract intermediate feature maps
- [x] Plot history, learned filters, and feature maps

## Milestone 4 - Verification
- [x] Check Python syntax and imports
- [x] Verify data isolation and model input/output dimensions
- [x] Verify expected intermediate activation dimensions
- [x] Verify prediction probabilities
- [x] Run the complete program
- [x] Record measured metrics and observed outputs

## Milestone 5 - Review and Handoff
- [x] Review algorithm, visualization logic, and requirement compliance
- [x] Fix confirmed issues
- [x] Update `docs/PROGRESS.md`
- [x] Report local and Colab verification status
- [ ] Run directly in Google Colab

## Rules
Complete milestones sequentially. Check boxes only after completing the work.
Record failures and unverified behavior instead of assuming success.
