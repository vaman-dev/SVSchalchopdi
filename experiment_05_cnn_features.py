"""Experiment 05: extract and visualize learned CNN features on MNIST."""

import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
VALIDATION_SIZE = 6_000
EPOCHS = 5
BATCH_SIZE = 128
SELECTED_IMAGE_INDEX = 0


def load_and_prepare_data():
    """Load MNIST, normalize images, add channels, and split validation data."""
    (source_images, source_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    print("Original training images:", source_images.shape)
    print("Testing images:", test_images.shape)

    # CNN inputs use normalized floating-point pixels and one grayscale channel.
    source_images = source_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    source_images = source_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]

    # Create validation data only from the official MNIST training partition.
    random_generator = np.random.default_rng(SEED)
    shuffled_indices = random_generator.permutation(len(source_images))
    validation_indices = shuffled_indices[:VALIDATION_SIZE]
    training_indices = shuffled_indices[VALIDATION_SIZE:]

    train_images = source_images[training_indices]
    train_labels = source_labels[training_indices]
    validation_images = source_images[validation_indices]
    validation_labels = source_labels[validation_indices]

    assert train_images.shape == (54_000, 28, 28, 1)
    assert validation_images.shape == (6_000, 28, 28, 1)
    assert test_images.shape == (10_000, 28, 28, 1)
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
    """Build and compile the CNN used for classification and visualization."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            # Same padding and stride 1 preserve the 28 x 28 spatial size.
            tf.keras.layers.Conv2D(
                8,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                activation="relu",
                name="conv_1",
            ),
            # Pooling halves each spatial dimension from 28 to 14.
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),
            # Valid padding and stride 2 reduce 14 x 14 to 6 x 6.
            tf.keras.layers.Conv2D(
                16,
                kernel_size=(3, 3),
                strides=2,
                padding="valid",
                activation="relu",
                name="conv_2",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    assert model.get_layer("conv_1").output.shape == (None, 28, 28, 8)
    assert model.get_layer("pool_1").output.shape == (None, 14, 14, 8)
    assert model.get_layer("conv_2").output.shape == (None, 6, 6, 16)
    return model


def build_feature_extractor(model):
    """Return a model that exposes selected intermediate CNN activations."""
    layer_names = ("conv_1", "pool_1", "conv_2")
    return tf.keras.Model(
        inputs=model.inputs[0],
        outputs=[model.get_layer(name).output for name in layer_names],
        name="feature_extractor",
    )


def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    epoch_numbers = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epoch_numbers, history.history["accuracy"], label="Training")
    axes[0].plot(
        epoch_numbers,
        history.history["val_accuracy"],
        label="Validation",
    )
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epoch_numbers, history.history["loss"], label="Training")
    axes[1].plot(
        epoch_numbers,
        history.history["val_loss"],
        label="Validation",
    )
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    plt.show()


def plot_filters(layer):
    """Plot one two-dimensional view of every learned filter in a Conv2D layer."""
    kernels = layer.get_weights()[0]
    # Later-layer filters span several input channels; their mean gives one
    # compact 2-D view per output filter for an educational comparison.
    filter_images = kernels.mean(axis=2)
    number_of_filters = filter_images.shape[-1]
    columns = min(8, number_of_filters)
    rows = math.ceil(number_of_filters / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(1.8 * columns, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for filter_index in range(number_of_filters):
        filter_image = filter_images[:, :, filter_index]
        limit = max(float(np.abs(filter_image).max()), 1e-7)
        axes[filter_index].imshow(
            filter_image,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axes[filter_index].set_title(f"Filter {filter_index + 1}")
        axes[filter_index].axis("off")

    for unused_axis in axes[number_of_filters:]:
        unused_axis.axis("off")

    figure.suptitle(f"Learned Filters: {layer.name}")
    figure.tight_layout()
    plt.show()


def plot_feature_maps(activation, layer_name):
    """Plot every channel from one intermediate activation tensor."""
    feature_maps = activation[0]
    number_of_maps = feature_maps.shape[-1]
    columns = min(8, number_of_maps)
    rows = math.ceil(number_of_maps / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for map_index in range(number_of_maps):
        axes[map_index].imshow(feature_maps[:, :, map_index], cmap="viridis")
        axes[map_index].set_title(f"Map {map_index + 1}")
        axes[map_index].axis("off")

    for unused_axis in axes[number_of_maps:]:
        unused_axis.axis("off")

    figure.suptitle(f"Feature Maps: {layer_name} ({feature_maps.shape})")
    figure.tight_layout()
    plt.show()


def plot_selected_image(image, actual_digit, predicted_digit, confidence):
    """Display the image used to generate the intermediate feature maps."""
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(
        f"Actual: {actual_digit} | Predicted: {predicted_digit}\n"
        f"Confidence: {confidence:.4f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    """Train, evaluate, extract features, and visualize the CNN experiment."""
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

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    selected_image = test_images[SELECTED_IMAGE_INDEX]
    actual_digit = int(test_labels[SELECTED_IMAGE_INDEX])
    probabilities = model.predict(selected_image[np.newaxis, ...], verbose=0)[0]

    assert probabilities.shape == (10,)
    assert np.all(np.isfinite(probabilities))
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])
    print("Selected test image index:", SELECTED_IMAGE_INDEX)
    print("Predicted digit:", predicted_digit)
    print("Actual digit:", actual_digit)
    print(f"Prediction confidence: {confidence:.4f}")

    feature_extractor = build_feature_extractor(model)
    activations = feature_extractor.predict(
        selected_image[np.newaxis, ...],
        verbose=0,
    )
    expected_shapes = ((1, 28, 28, 8), (1, 14, 14, 8), (1, 6, 6, 16))
    layer_names = ("conv_1", "pool_1", "conv_2")

    for layer_name, activation, expected_shape in zip(
        layer_names,
        activations,
        expected_shapes,
    ):
        assert activation.shape == expected_shape
        print(f"{layer_name} feature-map shape:", activation.shape)

    print("Dimension observation: same padding preserves 28 x 28 at conv_1.")
    print("Dimension observation: pooling reduces 28 x 28 to 14 x 14.")
    print("Dimension observation: valid padding with stride 2 reduces 14 x 14 to 6 x 6.")

    plot_selected_image(selected_image, actual_digit, predicted_digit, confidence)
    plot_training_history(history)
    plot_filters(model.get_layer("conv_1"))
    plot_filters(model.get_layer("conv_2"))
    for layer_name, activation in zip(layer_names, activations):
        plot_feature_maps(activation, layer_name)


if __name__ == "__main__":
    main()
