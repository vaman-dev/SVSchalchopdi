"""Experiment 04: classify MNIST digits with a deep neural network."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
VALIDATION_SIZE = 6_000
EPOCHS = 5
BATCH_SIZE = 128
SELECTED_IMAGE_INDEX = 0


def load_and_prepare_data():
    """Load MNIST, normalize its images, and create a validation partition."""
    (source_images, source_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    print("Original training images:", source_images.shape)
    print("Testing images:", test_images.shape)

    # Scale grayscale pixel values from [0, 255] to [0, 1].
    source_images = source_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Derive validation data only from the supplied MNIST training split.
    random_generator = np.random.default_rng(SEED)
    shuffled_indices = random_generator.permutation(len(source_images))
    validation_indices = shuffled_indices[:VALIDATION_SIZE]
    training_indices = shuffled_indices[VALIDATION_SIZE:]

    train_images = source_images[training_indices]
    train_labels = source_labels[training_indices]
    validation_images = source_images[validation_indices]
    validation_labels = source_labels[validation_indices]

    # Lightweight checks guard the expected split and normalized value range.
    assert train_images.shape == (54_000, 28, 28)
    assert validation_images.shape == (6_000, 28, 28)
    assert test_images.shape == (10_000, 28, 28)
    assert np.intersect1d(training_indices, validation_indices).size == 0
    assert 0.0 <= float(train_images.min()) <= float(train_images.max()) <= 1.0
    assert 0.0 <= float(test_images.min()) <= float(test_images.max()) <= 1.0

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
    """Build and compile the required fully connected DNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            # Flatten converts each 28 x 28 image into 784 input features.
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Softmax produces one probability for each digit class.
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    assert model.input_shape == (None, 28, 28)
    assert model.output_shape == (None, 10)
    return model


def plot_results(history, selected_image, actual_digit, predicted_digit):
    """Display accuracy, loss, and the selected test-image prediction."""
    epoch_numbers = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epoch_numbers, history.history["accuracy"], label="Training")
    axes[0].plot(
        epoch_numbers,
        history.history["val_accuracy"],
        label="Validation",
    )
    axes[0].set_title("DNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epoch_numbers, history.history["loss"], label="Training")
    axes[1].plot(
        epoch_numbers,
        history.history["val_loss"],
        label="Validation",
    )
    axes[1].set_title("DNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    axes[2].imshow(selected_image, cmap="gray")
    axes[2].set_title(f"Actual: {actual_digit} | Predicted: {predicted_digit}")
    axes[2].axis("off")

    figure.tight_layout()
    plt.show()


def main():
    """Train, evaluate, predict, and visualize the MNIST DNN experiment."""
    # Set Python, NumPy, and TensorFlow seeds for more repeatable runs.
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

    # Evaluate only after training, using MNIST's untouched test split.
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    selected_image = test_images[SELECTED_IMAGE_INDEX]
    actual_digit = int(test_labels[SELECTED_IMAGE_INDEX])
    probabilities = model.predict(selected_image[np.newaxis, ...], verbose=0)[0]

    assert probabilities.shape == (10,)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)

    predicted_digit = int(np.argmax(probabilities))
    prediction_confidence = float(probabilities[predicted_digit])

    print("Selected test image index:", SELECTED_IMAGE_INDEX)
    print("Class probabilities:", np.round(probabilities, 4))
    print("Predicted digit:", predicted_digit)
    print("Actual digit:", actual_digit)
    print(f"Prediction confidence: {prediction_confidence:.4f}")

    plot_results(
        history,
        selected_image,
        actual_digit,
        predicted_digit,
    )


if __name__ == "__main__":
    main()
