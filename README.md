# Autoencoder for Denoising MNIST Images

## Overview
This project implements an autoencoder using TensorFlow and Keras to denoise images from the MNIST dataset. The autoencoder is trained to reconstruct clean images from noisy inputs, demonstrating the ability of neural networks to perform denoising tasks.

## Dataset
The dataset used is the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is preprocessed by normalizing the pixel values to the range [0,1] and flattening the images into 1D vectors of size 784.

## Model Architecture
The autoencoder consists of:
- **Encoder**: Compresses input images into a low-dimensional representation (bottleneck layer).
- **Bottleneck Layer**: A dense layer with 32 neurons that acts as the compressed feature representation.
- **Decoder**: Reconstructs the original images from the compressed representation.

### Layers Used:
- `Dense(64, activation='relu')`
- `Dense(32, activation='relu')` (Bottleneck layer)
- `Dense(64, activation='relu')`
- `Dense(784, activation='sigmoid')`

## Training
The autoencoder is trained using the **binary cross-entropy loss function** and the **Adam optimizer** for 25 epochs with a batch size of 256.

### Adding Noise
To train the denoising autoencoder, Gaussian noise is added to the training and testing images before feeding them into the model. The autoencoder learns to remove this noise and reconstruct the clean images.

```python
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
```

## Results Visualization
A comparison of noisy vs. denoised images is visualized using Matplotlib. The first row shows noisy input images, the second row shows the denoised output from the autoencoder, and the third row displays the original test images.

```python
for i in range(n):
    ax = plt.subplot(4, n, i+1)
    plt.imshow(x_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(4, n, i+1+n)
    plt.imshow(reconstructed_noisy2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(4, n, i+1+n+n)
    plt.imshow(reconstructed_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(4, n, i+1+n+n+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## Dependencies
- TensorFlow
- Keras
- NumPy
- Matplotlib

## How to Run
1. Install dependencies: `pip install tensorflow keras numpy matplotlib`
2. Run the script: `python autoencoder.py`
3. The training process will begin, and after completion, denoised images will be displayed.

## Conclusion
This project demonstrates the use of autoencoders for image denoising. The model successfully learns to remove noise from MNIST images and reconstruct clear digits. It can be extended to other image datasets and further improved with convolutional layers for better feature extraction.

