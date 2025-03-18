# Autoencoder
# Autoencoder for MNIST Dataset

## Overview
This project implements a **basic autoencoder** using TensorFlow/Keras to reconstruct images from the **MNIST dataset**. The autoencoder compresses the 28x28 grayscale images into a smaller representation and then reconstructs them back.

## Project Structure
- **Load and Preprocess Data**: Normalize pixel values and flatten images.
- **Define Autoencoder Model**: Uses fully connected layers for encoding and decoding.
- **Train the Autoencoder**: Uses binary cross-entropy loss and Adam optimizer.
- **Reconstruction and Visualization**: Displays original vs. reconstructed images.

## Dependencies
Make sure you have the following installed:
```bash
pip install numpy tensorflow keras matplotlib
```

## Dataset
The MNIST dataset consists of **handwritten digit images** (0-9), each of size **28x28 pixels**.

## Model Architecture
- **Input Layer**: `784` neurons (flattened image input)
- **Encoder**:
  - `Dense(64, activation='relu')`
  - `Dense(32, activation='relu')` (Bottleneck layer)
- **Decoder**:
  - `Dense(64, activation='relu')`
  - `Dense(784, activation='sigmoid')` (Reshaped to 28x28 output image)

## Training
```python
autoencoder.fit(x_train, x_train, epochs=25, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
```
- **Epochs**: 25
- **Batch Size**: 256
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam

## Visualization
The code plots:
1. **Original Images**
2. **Reconstructed Images** (Autoencoder Output)

```python
plt.imshow(x_test[i].reshape(28, 28))  # Original
plt.imshow(reconstructed[i].reshape(28, 28))  # Reconstructed
```

## Expected Output
The reconstructed images should closely resemble the original MNIST digits, though they may be slightly blurry due to compression.

## Possible Improvements
- Increase bottleneck size for better feature retention.
- Use Convolutional Autoencoders (CNNs) for improved performance.
- Train for more epochs for better reconstruction quality.

## License
This project is open-source and available for educational purposes.
