# ARCHITECTURE.md

## Experiment
Experiment 05 - CNN Feature Extraction and Visualization

## Approach
A standalone TensorFlow/Keras program trains a compact CNN on MNIST. A second
Keras model reuses the trained network to return intermediate activations for
one selected test image. Plotting helpers display learned kernels and activation
channels without saving or loading local files.

## Data flow
```text
MNIST images (28 x 28)
    |
    v
Normalize and add channel dimension (28 x 28 x 1)
    |
    v
Seeded 54,000 training / 6,000 validation split
    |
    v
Conv2D: 8 filters, 3 x 3, stride 1, same padding -> 28 x 28 x 8
    |
    v
MaxPool2D: 2 x 2 -> 14 x 14 x 8
    |
    v
Conv2D: 16 filters, 3 x 3, stride 2, valid padding -> 6 x 6 x 16
    |
    v
Flatten -> Dense(64, ReLU) -> Dense(10, Softmax)
    |
    v
Final test evaluation and selected-image prediction
    |
    +--> learned filter grids
    `--> first-convolution, pooled, and deeper feature-map grids
```

## Dimension interpretation
- Same padding with stride 1 preserves the first convolution's 28 x 28 size.
- 2 x 2 max pooling halves 28 x 28 to 14 x 14.
- A 3 x 3 valid convolution with stride 2 reduces 14 x 14 to 6 x 6.
- Each output channel corresponds to one learned filter and one feature map.

## Model configuration
- Framework: TensorFlow/Keras
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Metric: accuracy
- Epochs: 5
- Batch size: 128
- Random seed: 42

## Program organization
`experiment_05_cnn_features.py` contains small functions for data preparation,
model construction, training-history plotting, filter visualization, feature-map
extraction/visualization, and the main execution flow. It has no local imports
or data-file dependencies.

## Verification strategy
- Parse and compile the source.
- Import the module and verify model/layer shapes.
- Run lightweight forward-pass checks for probabilities and activation shapes.
- Run the complete five-epoch experiment in the existing virtual environment
  with a non-interactive Matplotlib backend.
- Inspect the implementation for leakage and visualization indexing errors.

## Constraints
- Preserve all earlier experiment scripts.
- Keep MNIST test data isolated until final evaluation.
- Use a CNN, not a fully connected-only network or pretrained model.
- Report only observed metrics.
- Keep the program suitable for a single Google Colab cell.
