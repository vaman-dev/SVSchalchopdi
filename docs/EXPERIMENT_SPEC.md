# Experiment 05: CNN Feature Extraction and Visualization

## Aim
Build and train a simple Convolutional Neural Network (CNN) on MNIST, then
visualize its learned filters and intermediate feature maps to understand how
image features change across convolutional layers.

## Learning objectives
- Build and train a basic TensorFlow/Keras CNN.
- Extract intermediate activations from selected layers.
- Visualize learned convolutional filters and feature maps.
- Compare features produced by early and deeper convolutional layers.
- Observe how padding, convolution stride, and pooling affect spatial size.

## Dataset
MNIST contains 60,000 training and 10,000 test grayscale digit images. Each
image has shape 28 x 28. Images must be normalized to `[0, 1]` and expanded to
shape `(28, 28, 1)` for convolution.

## Required CNN
- Input shape: `(28, 28, 1)`
- First convolution: 8 filters, 3 x 3, ReLU, stride 1, same padding
- Max pooling: 2 x 2
- Second convolution: 16 filters, 3 x 3, ReLU, stride 2, valid padding
- Flatten and fully connected classification layers
- Output: 10 Softmax probabilities
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Metric: accuracy

## Required procedure
1. Load and normalize MNIST.
2. Create training and validation partitions without using test data.
3. Add a channel dimension to every image.
4. Build, compile, and train the CNN.
5. Evaluate it once on the untouched test set.
6. Predict a selected test image.
7. Extract outputs from the first convolution, pooling, and second
   convolution layers.
8. Visualize first- and second-layer learned filters.
9. Visualize and compare intermediate feature maps.
10. Print layer dimensions that demonstrate padding, pooling, and stride.

## Required deliverable
- `experiment_05_cnn_features.py`
- Printed data shapes, model summary, test metrics, selected prediction, and
  intermediate activation shapes
- Training accuracy/loss graphs
- Learned-filter visualizations
- Feature-map visualizations from multiple stages

The program must remain a standalone Python file suitable for local execution
or one Google Colab cell. Numerical results must not be claimed until the full
program is executed.
