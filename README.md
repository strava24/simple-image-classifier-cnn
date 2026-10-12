# Simple Image Classifier (CNN)

A small convolutional neural network built with TensorFlow/Keras that classifies 32x32 color images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset into 10 classes.

All code lives in a single notebook: [`cnn.ipynb`](cnn.ipynb).

## Dataset

CIFAR-10 contains 60,000 color images (32x32 RGB) in 10 mutually exclusive classes, 6,000 images per class:

- 50,000 training images
- 10,000 test images

Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

The dataset is downloaded automatically through `tf.keras.datasets.cifar10.load_data()`. Pixel values are normalized to the `[0, 1]` range.

## Model architecture

| Layer | Output shape | Params |
|-------|--------------|--------|
| Conv2D (32 filters, 3x3, ReLU) | 30x30x32 | 896 |
| MaxPooling2D (2x2) | 15x15x32 | 0 |
| Conv2D (64 filters, 3x3, ReLU) | 13x13x64 | 18,496 |
| MaxPooling2D (2x2) | 6x6x64 | 0 |
| Conv2D (64 filters, 3x3, ReLU) | 4x4x64 | 36,928 |
| Flatten | 1024 | 0 |
| Dense (64, ReLU) | 64 | 65,600 |
| Dense (10) | 10 | 650 |

Total: **122,570 trainable parameters** (~479 KB).

## Training

- Optimizer: Adam
- Loss: `SparseCategoricalCrossentropy(from_logits=True)`
- Metric: accuracy
- Epochs: 10
- Validation data: the CIFAR-10 test set

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | ~70.5% |
| Test loss | ~0.876 |

Training accuracy keeps increasing across epochs while validation accuracy flattens and starts to drop slightly, a sign of overfitting. Early stopping would likely give a better result.

## Getting started

1. Install the dependencies:

   ```bash
   pip install tensorflow matplotlib jupyter
   ```

2. Launch the notebook and run all cells:

   ```bash
   jupyter notebook cnn.ipynb
   ```

## Ideas for improvement

- Add early stopping to halt training when validation accuracy stops improving
- Add dropout and/or data augmentation to reduce overfitting
- Add batch normalization or more convolutional layers
- Use a separate validation split instead of the test set
