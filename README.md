# Autoencoders for Image Compression, Denoising, and Anomaly Detection

This repository contains implementations of three different autoencoder-based projects using TensorFlow and Keras:

1. **Image Compression using Autoencoders (Fashion-MNIST dataset)**  
2. **Image Denoising using Autoencoders (Fashion-MNIST dataset)**  
3. **Anomaly Detection in ECG data using Autoencoders**

---

## Requirements

Make sure you have the following libraries installed:
```
pip install numpy pandas matplotlib tensorflow scikit-learn
```

---

## Project 1: Image Compression using Autoencoders

### Overview
This project implements a simple autoencoder to compress and reconstruct images from the Fashion-MNIST dataset.

### Steps
1. Load and normalize the Fashion-MNIST dataset.  
2. Define an autoencoder model with an encoder and decoder.  
3. Train the model to minimize reconstruction loss (Mean Squared Error).  
4. Visualize original vs. reconstructed images.

### Key Components
- **Autoencoder(Model):** Defines the autoencoder architecture.  
- **autoencoder.fit():** Trains the model.  
- **plt.imshow():** Displays original and reconstructed images.

---

## Project 2: Image Denoising using Autoencoders

### Overview
This project adds noise to Fashion-MNIST images and then uses an autoencoder to remove the noise.

### Steps
1. Load and normalize the Fashion-MNIST dataset.  
2. Add random noise to images.  
3. Train a convolutional autoencoder to reconstruct clean images from noisy ones.  
4. Compare noisy vs. denoised images.

### Key Components
- **Denoise(Model):** Defines the denoising autoencoder model.  
- **autoencoder.fit():** Trains the model.  
- **plt.imshow():** Displays noisy and denoised images.

---

## Project 3: Anomaly Detection in ECG Data

### Overview
This project trains an autoencoder on normal ECG signals to detect anomalies.

### Steps
1. Load ECG dataset from TensorFlow storage.  
2. Normalize data and split into training/testing sets.  
3. Train an autoencoder on normal ECG signals.  
4. Evaluate reconstruction loss to identify anomalies.

### Key Components
- **AnomalyDetector(Model):** Defines the anomaly detection autoencoder.  
- **autoencoder.fit():** Trains the model.  
- **plt.plot():** Visualizes loss and ECG signals.

---

## Usage

To run the notebook locally:
```
jupyter notebook
```
Or open the notebook in Google Colab.

---

## Results
- **Image Compression:** Successfully reconstructs Fashion-MNIST images with a reduced latent representation.  
- **Image Denoising:** Removes noise effectively from images compared to the noisy inputs.  
- **Anomaly Detection:** Accurately identifies abnormal ECG signals through higher reconstruction loss.

---

## License
This project is open-source and available under the MIT License.
