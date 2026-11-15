# Experiment 04: DNN Image Classification Using MNIST

## Aim
Implement a Deep Neural Network (DNN) with TensorFlow/Keras to classify MNIST
handwritten digit images.

## Objectives
- Understand how multiple fully connected hidden layers form a DNN.
- Load and preprocess the MNIST dataset.
- Build and train a TensorFlow/Keras DNN.
- Evaluate the trained model on unseen test images.
- Predict the digit shown in a selected test image.

## Dataset and theory
MNIST contains 28 x 28 grayscale images of handwritten digits from 0 to 9.
Pixel values are normalized from [0, 255] to [0, 1]. A `Flatten` layer converts
each image into 784 features. Dense hidden layers with 128 and 64 neurons use
ReLU activations to learn nonlinear patterns. A 10-neuron Softmax output layer
returns one probability per digit class.

MNIST provides 60,000 training images and 10,000 test images. A validation
partition must be derived only from the training data. The supplied test data
must remain untouched until final evaluation.

## Required model
- Input shape: `(28, 28)`
- Flatten layer
- Dense hidden layer: 128 neurons, ReLU
- Dense hidden layer: 64 neurons, ReLU
- Output layer: 10 neurons, Softmax
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Metric: accuracy

## Procedure
1. Import TensorFlow/Keras, NumPy, and Matplotlib.
2. Set a random seed for repeatability.
3. Load MNIST.
4. Normalize image pixels.
5. Derive training and validation partitions from the supplied training split.
6. Build and compile the required DNN.
7. Train the model while monitoring validation performance.
8. Evaluate once on the untouched test set.
9. Predict a selected test image.
10. Display the selected image and training history.

## Required deliverable
- `experiment_04_dnn.py`, runnable locally and by pasting into Google Colab.
- Printed dataset shapes and model summary.
- Printed test loss and test accuracy.
- Printed predicted digit, actual digit, and prediction confidence.
- Training/validation accuracy and loss graphs plus the selected test image.

Do not state numerical results until the complete program has executed.
