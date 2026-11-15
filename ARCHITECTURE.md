# ARCHITECTURE.md

## Experiment
Experiment 2 — MNIST ANN Classification

## Approach
A fully connected artificial neural network
classifies handwritten digits into 10 categories.

## Data Flow
MNIST Dataset
    |
    v
Normalize Images (pixel / 255)
    |
    v
Train / Validation Split
    |
    v
Input Layer (28 x 28)
    |
    v
Flatten (784 features)
    |
    v
Dense Layer (128, ReLU)
    |
    v
Dense Layer (64, ReLU)
    |
    v
Output Layer (10, Softmax)
    |
    v
Model Training (Adam)
    |
    v
Final Test Evaluation
    |
    v
Prediction + Accuracy/Loss Graphs

## Dataset
- 60,000 original MNIST training images
- 54,000 training samples
- 6,000 validation samples
- 10,000 held-out test images
- Labels: integers from 0 to 9

## Model Configuration
Framework: TensorFlow/Keras
Loss: Sparse Categorical Crossentropy
Optimizer: Adam
Epochs: 5
Batch size: 128

## Deliverable
experiment_02_ann.py

The script must be executable independently
and compatible with Google Colab.

## Constraints
- No CNN or pretrained architecture.
- Do not train on test data.
- Report observed accuracy, not guessed values.
- Avoid unnecessary multi-file dependencies.
