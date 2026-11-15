# Experiment 03: Performance Analysis of Activation Functions

## Aim
Study and compare the behavior, performance, characteristics, and suitable
applications of ReLU, Sigmoid, Tanh, and Softmax activation functions.

## Objectives
- Understand how activation functions introduce non-linearity.
- Implement ReLU, Sigmoid, Tanh, and Softmax using NumPy.
- Apply every function to common input values and compare the outputs.
- Verify the mathematical range and important properties of each function.
- Visualize the shapes of the activation functions.
- Measure illustrative NumPy execution time and interpret it carefully.
- Identify suitable neural-network applications for each function.

## Required libraries and input
- Python 3
- NumPy
- Matplotlib
- Sample values: `[-5, -2, -1, 0, 1, 2, 5]`
- Dense values from `-6` to `6` for smooth plots

No external dataset or network download is required.

## Activation functions
### ReLU
`max(0, x)`. It produces zero for non-positive input and returns positive input
unchanged. It is commonly used in hidden layers.

### Sigmoid
`1 / (1 + exp(-x))`. It maps scalar values into `(0, 1)` and is commonly used
for binary output probabilities. The implementation must avoid overflow for
large positive or negative values.

### Tanh
`tanh(x)`. It maps scalar values into `(-1, 1)` and is zero-centered.

### Softmax
`exp(x_i) / sum(exp(x_j))`. It converts a vector of class scores into positive
probabilities whose sum is one. The implementation must subtract the maximum
score before exponentiation for numerical stability.

## Required procedure
1. Define common sample input values.
2. Implement all four functions without TensorFlow.
3. Apply them and print a readable output table.
4. Assert output ranges, special values, finite extreme-input behavior, and the
   Softmax probability sum.
5. Plot ReLU, Sigmoid, and Tanh curves plus sample Softmax probabilities.
6. Benchmark repeated NumPy calls on the same large input array.
7. Plot the measured milliseconds per call.
8. Print characteristics, limitations, and suitable applications.

## Required deliverable
- `experiment_03_activation_functions.py`
- Printed output comparison for the common input values
- Printed Softmax probability sum
- Activation-function visualization
- Illustrative timing results and timing chart
- Printed application/limitation comparison

Timing results must be described as environment-specific. Softmax timing must
not be presented as a perfectly equivalent operation because Softmax performs
vector normalization while the other functions operate element by element.
