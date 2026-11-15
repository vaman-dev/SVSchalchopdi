"""Experiment 2: classify MNIST digits with a fully connected ANN."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
VALIDATION_SIZE = 6_000
EPOCHS = 5
BATCH_SIZE = 128


def load_and_prepare_data():
    """Load MNIST, normalize its images, and make a seeded validation split."""
    (source_images, source_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    print("Original training images:", source_images.shape)
    print("Testing images:", test_images.shape)

    # Dense networks train more reliably when pixel values are scaled to [0, 1].
    source_images = source_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Select validation examples only from MNIST's supplied training split.
    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(len(source_images))
    validation_indices = shuffled_indices[:VALIDATION_SIZE]
    training_indices = shuffled_indices[VALIDATION_SIZE:]

    train_images = source_images[training_indices]
    train_labels = source_labels[training_indices]
    validation_images = source_images[validation_indices]
    validation_labels = source_labels[validation_indices]

    # These checks catch accidental changes to the intended 54k/6k split.
    assert train_images.shape == (54_000, 28, 28)
    assert validation_images.shape == (6_000, 28, 28)
    assert test_images.shape == (10_000, 28, 28)
    assert np.intersect1d(training_indices, validation_indices).size == 0

    print("Training images after split:", train_images.shape)
    print("Validation images:", validation_images.shape)

    return (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    )


def build_model():
    """Create and compile the required fully connected neural network."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            # Flatten changes each 28 x 28 image into 784 input features.
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Softmax returns one probability for each digit from 0 to 9.
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    assert model.output_shape == (None, 10)
    return model


def plot_training_history(history):
    """Display the training and validation accuracy/loss curves."""
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 4))

    accuracy_axis.plot(epochs, history.history["accuracy"], label="Training")
    accuracy_axis.plot(epochs, history.history["val_accuracy"], label="Validation")
    accuracy_axis.set_title("Model Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()

    loss_axis.plot(epochs, history.history["loss"], label="Training")
    loss_axis.plot(epochs, history.history["val_loss"], label="Validation")
    loss_axis.set_title("Model Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()

    figure.tight_layout()
    plt.show()


def main():
    # This sets Python, NumPy, and TensorFlow seeds for more repeatable runs.
    tf.keras.utils.set_random_seed(SEED)

    (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    ) = load_and_prepare_data()

    model = build_model()
    model.summary()

    history = model.fit(
        train_images,
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(validation_images, validation_labels),
    )

    # The supplied MNIST test split remains untouched until final evaluation.
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(test_images[:1], verbose=0)[0]
    assert probabilities.shape == (10,)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)

    predicted_digit = int(np.argmax(probabilities))
    actual_digit = int(test_labels[0])
    print("Sample probabilities:", np.round(probabilities, 4))
    print("Predicted digit:", predicted_digit)
    print("Actual digit:", actual_digit)
    print(f"Prediction confidence: {probabilities[predicted_digit]:.4f}")

    plot_training_history(history)


if __name__ == "__main__":
    main()
