                                                                                           # Image Colorization using GANs

# GAN for Image Colorization

This repository contains an implementation of a Generative Adversarial Network (GAN) for image colorization. The network is designed to colorize grayscale images by learning the AB channels in the LAB color space. The grayscale image (L channel) is used as the input to the generator, which predicts the AB channels.

## Dataset Preparation

The dataset consists of grayscale images (L channel) and corresponding color information (AB channels). The L channel data is loaded from grayscale images, while the AB channels are provided separately.

### Steps:
1. Grayscale images are loaded and reshaped to (25000, 224, 224, 1) dimensions.
2. The AB channels are loaded in three batches and concatenated into a single array.
3. Both the grayscale (L channel) and color (AB channels) images are normalized to the range [-1, 1] and converted to PyTorch tensors.
4. A `TensorDataset` is created for training and testing, with the first 20000 images for training and the remaining for testing.

## Model Architecture

### Generator

The generator is modeled as a U-Net-like architecture, with encoding and decoding blocks. The encoder uses convolutional layers, batch normalization, and LeakyReLU activation. The decoder employs transposed convolutions and skip connections to recover spatial information. The generator outputs the AB color channels.

### Discriminator

The discriminator is a convolutional neural network (CNN) that takes the AB channels as input and outputs a probability indicating whether the input is real or generated. It uses binary cross-entropy loss for training.

## Training

The GAN is trained using a combination of adversarial loss (binary cross-entropy) and L1 loss for reconstruction. The adversarial loss ensures the generator produces realistic colorizations, while the L1 loss ensures accurate reconstruction of the AB channels.

### Optimizers

Both the generator and discriminator use the Adam optimizer with learning rate 2e-4 and beta parameters (0.5, 0.999). The generator is trained to minimize the combination of adversarial loss and L1 loss, while the discriminator is trained using adversarial loss only.

### Key Hyperparameters:
- Batch size: 16
- Learning rate: 2e-4
- λ for L1 loss: 100
- Number of epochs: 100

## Usage

To use this GAN model for image colorization:

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Place your grayscale image data in `data/l/gray_scale.npy` and color data (AB channels) in `data/ab/ab1.npy`, `data/ab/ab2.npy`, `data/ab/ab3.npy`.

3. Install the required dependencies:
    ```bash
    pip install torch numpy
    ```

4. Run the training script:
    ```bash
    python train_gan.py
    ```

During training, you will be prompted whether to save the model and adjust the learning rate after each epoch.

