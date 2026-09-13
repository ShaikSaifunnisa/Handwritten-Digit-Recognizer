"""
Handwritten Digit Recognizer
-----------------------------
A Convolutional Neural Network (CNN) that recognizes handwritten
digits (0-9) using the MNIST dataset.

Requirements:
    pip install tensorflow matplotlib numpy scikit-learn seaborn

Run:
    python Handwritten_Digit_Recognizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# ============================================================
# 1. LOAD AND PREPROCESS THE DATA
# ============================================================

def load_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize images
    x_train = x_train.reshape(
        -1, 28, 28, 1
    ).astype("float32") / 255.0

    x_test = x_test.reshape(
        -1, 28, 28, 1
    ).astype("float32") / 255.0

    # Convert labels to categorical format
    y_train_cat = to_categorical(
        y_train,
        num_classes=10
    )

    y_test_cat = to_categorical(
        y_test,
        num_classes=10
    )

    print(
        f"Training samples: {x_train.shape[0]}"
    )

    print(
        f"Test samples:     {x_test.shape[0]}"
    )

    return (
        x_train,
        y_train,
        y_train_cat,
        x_test,
        y_test,
        y_test_cat
    )


# ============================================================
# 2. PREVIEW SAMPLE IMAGES
# ============================================================

def preview_samples(x_train, y_train, n=10):

    plt.figure(figsize=(10, 2))

    for i in range(n):

        plt.subplot(
            1,
            n,
            i + 1
        )

        plt.imshow(
            x_train[i].reshape(28, 28),
            cmap="gray"
        )

        plt.title(
            str(y_train[i])
        )

        plt.axis("off")

    plt.suptitle(
        "Sample Training Digits"
    )

    plt.tight_layout()

    plt.savefig(
        "sample_digits.png"
    )

    plt.close()

    print(
        "Saved sample_digits.png"
    )


# ============================================================
# 3. BUILD THE CNN MODEL
# ============================================================

def build_model():

    model = models.Sequential([

        layers.Input(
            shape=(28, 28, 1)
        ),

        # First convolution block
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.25
        ),

        # Second convolution block
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.25
        ),

        # Fully connected layers
        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.5
        ),

        # Output layer: digits 0-9
        layers.Dense(
            10,
            activation="softmax"
        )

    ])


    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )


    return model


# ============================================================
# 4. TRAIN THE MODEL
# ============================================================

def train_model(
    model,
    x_train,
    y_train_cat,
    x_test,
    y_test_cat,
    epochs=10,
    batch_size=128
):

    early_stop = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True

    )


    history = model.fit(

        x_train,

        y_train_cat,

        validation_data=(
            x_test,
            y_test_cat
        ),

        epochs=epochs,

        batch_size=batch_size,

        callbacks=[
            early_stop
        ],

        verbose=1

    )


    return history


# ============================================================
# 5. PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(history):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )


    # Accuracy
    axes[0].plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Val Accuracy"
    )

    axes[0].set_title(
        "Model Accuracy"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Accuracy"
    )

    axes[0].legend()


    # Loss
    axes[1].plot(
        history.history["loss"],
        label="Train Loss"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Val Loss"
    )

    axes[1].set_title(
        "Model Loss"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Loss"
    )

    axes[1].legend()


    plt.tight_layout()

    plt.savefig(
        "training_history.png"
    )

    plt.close()

    print(
        "Saved training_history.png"
    )


# ============================================================
# 6. EVALUATE THE MODEL
# ============================================================

def evaluate_model(
    model,
    x_test,
    y_test,
    y_test_cat
):

    test_loss, test_acc = model.evaluate(
        x_test,
        y_test_cat,
        verbose=0
    )


    print(
        f"\nTest Accuracy: {test_acc * 100:.2f}%"
    )

    print(
        f"Test Loss:     {test_loss:.4f}"
    )


    # Make predictions
    y_pred_probs = model.predict(
        x_test,
        verbose=0
    )


    y_pred = np.argmax(
        y_pred_probs,
        axis=1
    )


    # Classification report
    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )


    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(
        figsize=(8, 6)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )


    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.tight_layout()


    plt.savefig(
        "confusion_matrix.png"
    )

    plt.close()


    print(
        "Saved confusion_matrix.png"
    )


    return y_pred


# ============================================================
# 7. SHOW SAMPLE PREDICTIONS
# ============================================================

def show_predictions(
    x_test,
    y_test,
    y_pred,
    n=10
):

    idxs = np.random.choice(
        len(x_test),
        n,
        replace=False
    )


    plt.figure(
        figsize=(12, 3)
    )


    for i, idx in enumerate(idxs):

        plt.subplot(
            1,
            n,
            i + 1
        )


        plt.imshow(
            x_test[idx].reshape(28, 28),
            cmap="gray"
        )


        color = (
            "green"
            if y_pred[idx] == y_test[idx]
            else "red"
        )


        plt.title(
            f"P:{y_pred[idx]}\nT:{y_test[idx]}",
            color=color,
            fontsize=10
        )


        plt.axis("off")


    plt.suptitle(
        "Predictions (Green=Correct, Red=Wrong)"
    )


    plt.tight_layout()


    plt.savefig(
        "sample_predictions.png"
    )

    plt.close()


    print(
        "Saved sample_predictions.png"
    )


# ============================================================
# 8. PREDICT A SINGLE CUSTOM IMAGE
# ============================================================

def predict_single_image(
    model,
    image_path
):

    from PIL import Image


    # Open image
    img = Image.open(
        image_path
    ).convert("L")


    # Resize to MNIST size
    img = img.resize(
        (28, 28)
    )


    # Convert image to NumPy array
    img_arr = np.array(
        img
    ).astype("float32") / 255.0


    # Add required dimensions
    img_arr = img_arr.reshape(
        1,
        28,
        28,
        1
    )


    # Predict
    pred = model.predict(
        img_arr,
        verbose=0
    )


    # Get predicted digit
    digit = np.argmax(
        pred
    )


    # Get confidence
    confidence = (
        np.max(pred) * 100
    )


    print(
        f"Predicted digit: {digit} "
        f"(confidence: {confidence:.2f}%)"
    )


    return digit


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print(
        "Loading MNIST dataset..."
    )


    # Load data
    (
        x_train,
        y_train,
        y_train_cat,
        x_test,
        y_test,
        y_test_cat
    ) = load_data()


    # Preview training samples
    preview_samples(
        x_train,
        y_train
    )


    # Build model
    print(
        "\nBuilding model..."
    )


    model = build_model()


    model.summary()


    # Train model
    print(
        "\nTraining model..."
    )


    history = train_model(
        model,
        x_train,
        y_train_cat,
        x_test,
        y_test_cat,
        epochs=10
    )


    # Plot training history
    plot_training_history(
        history
    )


    # Evaluate model
    print(
        "\nEvaluating model..."
    )


    y_pred = evaluate_model(
        model,
        x_test,
        y_test,
        y_test_cat
    )


    # Show predictions
    show_predictions(
        x_test,
        y_test,
        y_pred
    )


    # Save trained model
    print(
        "\nSaving model..."
    )


    model.save(
        "digit_recognizer_model.h5"
    )


    print(
        "Model saved as digit_recognizer_model.h5"
    )


    print(
        "\nDone! Check the generated PNG files for visualizations."
    )


    # Example:
    # predict_single_image(
    #     model,
    #     "my_digit.png"
    # )
