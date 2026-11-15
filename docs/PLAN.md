# PLAN.md

## Active Experiment
Experiment 04 - DNN Image Classification Using MNIST

## Goal
Implement and verify a standalone TensorFlow/Keras DNN that classifies MNIST
handwritten digits and predicts a selected test image.

## Milestone 1 - Requirements
- [x] Review Experiment 04 requirements
- [x] Identify the dataset, required model, and outputs
- [x] Confirm the existing dependencies satisfy the experiment

## Milestone 2 - Architecture
- [x] Design normalization and validation splitting
- [x] Define the 128/64-neuron DNN
- [x] Define training, final evaluation, prediction, and plots

## Milestone 3 - Implementation
- [x] Create `experiment_04_dnn.py`
- [x] Load and normalize MNIST
- [x] Create leakage-free training and validation partitions
- [x] Build and compile the DNN
- [x] Train and evaluate the model
- [x] Predict and display a selected test digit
- [x] Display accuracy and loss graphs

## Milestone 4 - Verification
- [x] Check Python syntax and imports
- [x] Verify split sizes and test isolation
- [x] Verify the model input/output shapes
- [x] Verify output probabilities sum approximately to one
- [x] Run the complete experiment
- [x] Record measured metrics

## Milestone 5 - Independent Review
- [x] Obtain independent code review
- [x] Separate confirmed defects from possible risks
- [x] Address confirmed defects (none found)
- [x] Update `docs/PROGRESS.md`
- [x] Report local and Colab verification status
- [ ] Run directly in Google Colab

## Rules
Complete milestones sequentially. Check boxes only after completing the work.
Record failures and unverified behavior instead of assuming success.
