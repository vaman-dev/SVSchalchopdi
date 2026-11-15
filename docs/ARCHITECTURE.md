# ARCHITECTURE.md

## Experiment
Experiment 04 - DNN Image Classification Using MNIST

## Approach
A standalone TensorFlow/Keras program trains a fully connected deep neural
network with two hidden layers. It keeps MNIST test images isolated until final
evaluation and then predicts one selected test digit.

## Data flow
```text
MNIST dataset
    |
    v
Normalize pixels to [0, 1]
    |
    v
Seeded 54,000 training / 6,000 validation split
    |
    v
Input (28 x 28)
    |
    v
Flatten (784 features)
    |
    v
Dense (128, ReLU)
    |
    v
Dense (64, ReLU)
    |
    v
Dense (10, Softmax)
    |
    v
Adam training with sparse categorical cross-entropy
    |
    v
Final evaluation on 10,000 untouched test images
    |
    v
Selected-image prediction and visualizations
```

## Model configuration
- Framework: TensorFlow/Keras
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Metric: accuracy
- Initial epochs: 5
- Batch size: 128
- Random seed: 42

## Program organization
The single `experiment_04_dnn.py` file will contain small functions for data
preparation, model construction, training-history visualization, selected-digit
visualization, and the main execution flow. It will have no local imports or
file dependencies.

## Verification strategy
- Parse and compile the Python source.
- Use lightweight assertions for dataset shapes, split isolation, model shapes,
  and Softmax probability sum.
- Run the full five-epoch program using the retained virtual environment.
- Use a non-interactive Matplotlib backend for automated local execution.
- Request independent read-only reviews for algorithmic and ML correctness.

## Constraints
- Use a fully connected DNN, not a CNN or pretrained model.
- Do not train on or tune against test data.
- Preserve `experiment_02_ann.py`.
- Report only measured results.
- Keep the file suitable for a single Google Colab cell.
