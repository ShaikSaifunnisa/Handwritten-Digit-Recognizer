# Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) that recognizes handwritten digits (0–9) trained on the MNIST dataset. Built as part of the 1-Month AI Internship project requirements.

## Overview

This project trains a CNN in TensorFlow/Keras on the classic MNIST dataset (70,000 grayscale images of handwritten digits) and evaluates how accurately it can classify unseen digits.

## Features

- Data preprocessing (normalization, reshaping, one-hot encoding)
- CNN architecture with Conv2D, BatchNormalization, MaxPooling, and Dropout layers
- Early stopping to prevent overfitting
- Full evaluation: accuracy/loss curves, classification report, confusion matrix
- Saves the trained model (`digit_recognizer_model.h5`) for reuse
- Utility function to predict digits from custom images

## Requirements

```bash
pip install tensorflow matplotlib numpy scikit-learn seaborn pillow
```

## How to Run

```bash
python handwritten_digit_recognizer.py
```

## Results

Achieved **99.38% test accuracy** on 10,000 unseen MNIST test images after 10 training epochs, with precision/recall/F1 scores above 0.98 for every digit class (0–9).

## Output Files

| File | What it shows |
|---|---|
| `sample_digits.png` | Preview of training digits |
| `training_history.png` | Accuracy & loss curves over training |
| `confusion_matrix.png` | Heatmap of predicted vs. true labels |
| `sample_predictions.png` | Sample test predictions (green = correct, red = wrong) |
| `digit_recognizer_model.h5` | The trained model file |

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 60,000 training images and 10,000 test images of handwritten digits, loaded via `tensorflow.keras.datasets.mnist`.
