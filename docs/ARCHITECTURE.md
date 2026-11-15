# ARCHITECTURE.md

## Experiment
Experiment 03 - Performance Analysis of Activation Functions

## Approach
A standalone NumPy and Matplotlib program applies four activation functions to
the same sample values, validates their defining properties, visualizes their
behavior, and records an illustrative vectorized timing comparison.

## Program flow
```text
Sample values [-5, -2, -1, 0, 1, 2, 5]
    |
    +--> ReLU ---------> element-wise outputs in [0, infinity)
    +--> Sigmoid ------> element-wise outputs in (0, 1)
    +--> Tanh ---------> element-wise outputs in (-1, 1)
    `--> Softmax ------> one probability distribution with sum 1
             |
             v
    printed comparison and mathematical assertions

Dense values from -6 to 6
    |
    `--> ReLU, Sigmoid, and Tanh curve plots

Sample values
    `--> Softmax class-probability bar chart

Large deterministic NumPy input
    |
    `--> repeated timing --> milliseconds per call --> timing bar chart

Static comparison records
    `--> output range, common use, advantages, and limitations
```

## Numerical design
- ReLU uses `numpy.maximum`.
- Sigmoid uses separate positive and negative branches to avoid overflow.
- Tanh uses `numpy.tanh`.
- Softmax subtracts the maximum value along the selected axis, exponentiates,
  and divides by the sum along that axis.
- Softmax supports both one-dimensional vectors and two-dimensional batches.

## Timing design
- Generate a deterministic 100,000-value array once outside the timed calls.
- Run each function repeatedly using `timeit.repeat`.
- Report the best average milliseconds per call across several repeats to
  reduce incidental system noise.
- Use the same numeric array for every function.
- Explicitly state that Softmax has additional reduction/normalization work, so
  the chart is an implementation case study rather than an equal-complexity
  scientific ranking.

## Program organization
`experiment_03_activation_functions.py` contains small functions for the four
activations, property validation, output printing, visualization, timing,
comparison reporting, and the main execution flow. It has no external dataset,
local module, or file dependency.

## Verification strategy
- Compile and import the source using the retained virtual environment.
- Test representative values and arrays containing `-1000` and `1000`.
- Assert finite outputs and mathematical ranges.
- Assert Sigmoid(0) is 0.5, Tanh(0) is 0, and ReLU removes negative values.
- Assert one-dimensional and batched Softmax results each sum to one.
- Execute the complete program with Matplotlib's non-interactive backend.
- Review timing labels and explanatory text for misleading comparisons.

## Constraints
- Preserve Experiments 02, 04, 05, and 06.
- Do not train a neural network because the specified procedure compares the
  mathematical activation transformations directly.
- Do not claim timing portability across environments.
- Keep the file suitable for one Google Colab cell.
