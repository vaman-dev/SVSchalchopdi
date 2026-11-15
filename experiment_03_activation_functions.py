"""Experiment 03: compare ReLU, Sigmoid, Tanh, and Softmax activations."""

import timeit

import matplotlib.pyplot as plt
import numpy as np


SAMPLE_INPUTS = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])


def relu(values):
    """Return the Rectified Linear Unit activation element by element."""
    values = np.asarray(values, dtype=np.float64)
    return np.maximum(0.0, values)


def sigmoid(values):
    """Return a numerically stable Sigmoid activation element by element."""
    values = np.asarray(values, dtype=np.float64)
    # exp(-logaddexp(0, -x)) is equivalent to 1 / (1 + exp(-x)) but
    # remains finite for very large positive and negative inputs.
    return np.exp(-np.logaddexp(0.0, -values))


def tanh(values):
    """Return the hyperbolic tangent activation element by element."""
    values = np.asarray(values, dtype=np.float64)
    return np.tanh(values)


def softmax(values, axis=-1):
    """Convert class scores into stable probabilities along the selected axis."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("Softmax requires a vector or a batch of vectors.")

    # Subtracting the maximum prevents overflow without changing probabilities.
    shifted_values = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted_values)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def calculate_outputs(inputs):
    """Apply all four activation functions to common input values."""
    return {
        "ReLU": relu(inputs),
        "Sigmoid": sigmoid(inputs),
        "Tanh": tanh(inputs),
        "Softmax": softmax(inputs),
    }


def validate_activations(inputs, outputs):
    """Check defining properties and extreme-input numerical stability."""
    relu_values = outputs["ReLU"]
    sigmoid_values = outputs["Sigmoid"]
    tanh_values = outputs["Tanh"]
    softmax_values = outputs["Softmax"]

    assert np.all(relu_values[inputs < 0] == 0.0)
    assert np.allclose(relu_values[inputs > 0], inputs[inputs > 0])
    # Finite-precision arithmetic may round extreme Sigmoid inputs to 0 or 1.
    assert np.all((sigmoid_values >= 0.0) & (sigmoid_values <= 1.0))
    assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
    assert np.all((tanh_values >= -1.0) & (tanh_values <= 1.0))
    assert np.isclose(tanh(np.array([0.0]))[0], 0.0)
    assert np.all((softmax_values >= 0.0) & (softmax_values <= 1.0))
    assert np.isclose(softmax_values.sum(), 1.0)

    extreme_inputs = np.array([-1000.0, 0.0, 1000.0])
    assert np.isfinite(sigmoid(extreme_inputs)).all()
    assert np.isfinite(tanh(extreme_inputs)).all()
    assert np.isfinite(softmax(extreme_inputs)).all()
    assert np.isclose(softmax(extreme_inputs).sum(), 1.0)

    batched_scores = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0]])
    batched_probabilities = softmax(batched_scores, axis=1)
    assert batched_probabilities.shape == batched_scores.shape
    assert np.allclose(batched_probabilities.sum(axis=1), np.ones(2))


def print_output_comparison(inputs, outputs):
    """Print the transformations of the shared sample values."""
    print("\nACTIVATION OUTPUT COMPARISON")
    print("Softmax treats the complete input vector as competing class scores.")
    print(f"{'Input':>8} {'ReLU':>10} {'Sigmoid':>12} {'Tanh':>12} {'Softmax':>12}")
    print("-" * 58)

    for index, input_value in enumerate(inputs):
        print(
            f"{input_value:8.2f} "
            f"{outputs['ReLU'][index]:10.4f} "
            f"{outputs['Sigmoid'][index]:12.6f} "
            f"{outputs['Tanh'][index]:12.6f} "
            f"{outputs['Softmax'][index]:12.6f}"
        )

    print(f"Softmax probability sum: {outputs['Softmax'].sum():.6f}")


def plot_activation_behavior(inputs, outputs):
    """Plot three scalar curves and the sample Softmax distribution."""
    curve_inputs = np.linspace(-6.0, 6.0, 500)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(curve_inputs, relu(curve_inputs), color="#4C78A8")
    axes[0, 0].set_title("ReLU: max(0, x)")
    axes[0, 0].set_ylabel("Output")

    axes[0, 1].plot(curve_inputs, sigmoid(curve_inputs), color="#F58518")
    axes[0, 1].set_title("Sigmoid: output between 0 and 1")
    axes[0, 1].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(curve_inputs, tanh(curve_inputs), color="#54A24B")
    axes[1, 0].set_title("Tanh: zero-centered output")
    axes[1, 0].set_xlabel("Input")
    axes[1, 0].set_ylabel("Output")
    axes[1, 0].set_ylim(-1.1, 1.1)

    labels = [f"{value:g}" for value in inputs]
    axes[1, 1].bar(labels, outputs["Softmax"], color="#E45756")
    axes[1, 1].set_title("Softmax probabilities for sample scores")
    axes[1, 1].set_xlabel("Input class score")
    axes[1, 1].set_ylabel("Probability")

    for axis in axes.flat[:3]:
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.axvline(0.0, color="black", linewidth=0.7)
        axis.grid(alpha=0.25)

    figure.suptitle("Activation Function Behavior")
    figure.tight_layout()
    plt.show()


def benchmark_activations(array_size=100_000, calls=50, repeats=5):
    """Measure illustrative vectorized NumPy execution time per call."""
    rng = np.random.default_rng(42)
    benchmark_values = rng.normal(0.0, 3.0, array_size)
    functions = {
        "ReLU": relu,
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "Softmax": softmax,
    }
    milliseconds_per_call = {}

    for name, function in functions.items():
        durations = timeit.repeat(
            lambda selected_function=function: selected_function(benchmark_values),
            repeat=repeats,
            number=calls,
        )
        milliseconds_per_call[name] = min(durations) * 1000.0 / calls

    return milliseconds_per_call


def print_and_plot_timings(timings, array_size):
    """Print and plot measured execution times with interpretation limits."""
    print("\nILLUSTRATIVE NUMPY TIMING")
    print(f"Array size: {array_size:,} values")
    for name, milliseconds in timings.items():
        print(f"{name:8}: {milliseconds:.4f} ms per call")
    print(
        "Timing is environment-specific. Softmax performs a maximum, "
        "exponentiation, sum, and normalization across the vector, so this is "
        "not an equal-complexity ranking of interchangeable operations."
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(timings.keys(), timings.values(), color="#72B7B2")
    axis.bar_label(
        bars,
        labels=[f"{value:.3f} ms" for value in timings.values()],
        padding=3,
    )
    axis.set_title(f"Illustrative NumPy Time per Call ({array_size:,} Values)")
    axis.set_xlabel("Activation function")
    axis.set_ylabel("Milliseconds per call (lower is faster in this run)")
    axis.set_ylim(0, max(timings.values()) * 1.2)
    figure.tight_layout()
    plt.show()


def print_application_guide():
    """Print the main characteristics and suitable applications."""
    rows = [
        (
            "ReLU",
            "[0, infinity)",
            "Hidden layers",
            "Fast; reduces saturation for positive inputs",
            "Negative inputs have zero gradient; neurons may become inactive",
        ),
        (
            "Sigmoid",
            "(0, 1)",
            "Binary output layer",
            "Smooth probability-like scalar output",
            "Saturates and can cause vanishing gradients",
        ),
        (
            "Tanh",
            "(-1, 1)",
            "Zero-centered hidden/state outputs",
            "Represents positive and negative values",
            "Saturates and can cause vanishing gradients",
        ),
        (
            "Softmax",
            "Each (0, 1); total 1",
            "Multi-class output layer",
            "Creates one categorical probability distribution",
            "Class probabilities are coupled; sensitive to score scale",
        ),
    ]

    print("\nAPPLICATION AND LIMITATION GUIDE")
    for name, output_range, common_use, advantage, limitation in rows:
        print(f"\n{name}")
        print(f"  Output range: {output_range}")
        print(f"  Common use: {common_use}")
        print(f"  Advantage: {advantage}")
        print(f"  Limitation: {limitation}")


def main():
    """Run the complete activation-function case study."""
    print("EXPERIMENT 03 - PERFORMANCE ANALYSIS OF ACTIVATION FUNCTIONS")
    print("Input values:", SAMPLE_INPUTS)

    outputs = calculate_outputs(SAMPLE_INPUTS)
    validate_activations(SAMPLE_INPUTS, outputs)
    print_output_comparison(SAMPLE_INPUTS, outputs)
    plot_activation_behavior(SAMPLE_INPUTS, outputs)

    benchmark_size = 100_000
    timings = benchmark_activations(array_size=benchmark_size)
    assert all(np.isfinite(value) and value > 0.0 for value in timings.values())
    print_and_plot_timings(timings, benchmark_size)
    print_application_guide()

    print("\nAll mathematical checks passed.")
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
