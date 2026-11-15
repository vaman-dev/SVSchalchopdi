# Experiment 2: Simple Artificial Neural Network (ANN) for Classification

## Aim
Implement an ANN for classification using TensorFlow/Keras.

## Objectives
- Understand input, hidden, and output layers.
- Build a basic ANN in TensorFlow/Keras.
- Train using labeled handwritten-digit data.
- Evaluate performance on a held-out test set.
- Predict class labels for unseen images.

## Dataset and theory
Use MNIST, consisting of 28 x 28 grayscale handwritten digits 0-9. Scale
pixels from [0, 255] to [0, 1]. `Flatten` converts each image to 784
features. Two `Dense` hidden layers with ReLU learn nonlinear patterns. A
10-unit Softmax output represents class probabilities. Adam optimizes sparse
categorical cross-entropy.

## Procedure
1. Import libraries.
2. Load MNIST classification data.
3. Separate train, validation, and test data without leakage.
4. Normalize input values.
5. Define a Keras Sequential ANN.
6. Add Flatten, two Dense hidden layers, and an output layer.
7. Compile the model with Adam and an appropriate loss and metric.
8. Train on training data while monitoring validation data.
9. Evaluate once on untouched test data.
10. Predict the class of a test example.

## Required deliverables
- One reproducible standalone Python program compatible with local execution
  and Google Colab.
- Training history plots for accuracy and loss.
- Measured test loss, test accuracy, and one prediction example.

Do not state a numerical result until the code has run.
