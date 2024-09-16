# Handwritten Digit Recognizer with Neural Networks

## Overview

This project implements a neural network for recognizing handwritten digits (using the MNIST dataset) and generating captions. The model combines a **Convolutional Neural Network (CNN)** for feature extraction from images and a **Recurrent Neural Network (RNN)** for variable-length caption generation. Developed in **PyTorch**, the model is trained and evaluated using **Google Colab**.

### Key Features:
- **Image Feature Extraction with CNN**: Classifies handwritten digits.
- **Caption Generation with RNN**: Generates a sequence of tokens describing the digits.
- **Modular Design**: Allows for end-to-end training or transfer learning using pre-trained CNN models.
- **Training on MNIST Dataset**: Recognizes and captions digit images with high accuracy.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch, TorchVision, Matplotlib, NumPy
- Google Colab for cloud-based execution

### Installation
1. Clone the repository:
    ```bash
    git clone <repo-url>
    ```
2. Install dependencies:
    ```bash
    pip install torch torchvision matplotlib numpy
    ```
3. Open the notebook `prog4.ipynb` in Google Colab.

### Dataset
- **MNIST Dataset**: Automatically downloaded and pre-processed into DataLoader objects for efficient batch processing.

## Model Architecture

### CNN Encoder
- Extracts image features using three convolutional layers and fully connected layers, outputting an 84-dimensional vector.
- **Input:** 28x28 grayscale image
- **Output:** 84-dimensional feature vector

### RNN Decoder
- Generates captions by predicting the token sequence from the image features.
- **Input:** CNN feature vector and previous token
- **Output:** One-hot encoded token sequence

## Training Modes
- **Encoder Mode:** Trains the CNN for digit classification, achieving >95% accuracy.
- **Decoder Mode:** Trains the RNN to generate correct token sequences.
- **Modular Mode:** Combines CNN and RNN for caption generation.
- **End-to-End Mode:** Jointly trains CNN and RNN for captioning.

## Training and Evaluation

1. Train the CNN for digit recognition:
    ```python
    train(epochs, enc_optimizer, cnnClassifier, mode='ENCODER')
    ```
2. Train the RNN for sequence generation:
    ```python
    train(epochs, dec_optimizer, rnnModel, mode='DECODER')
    ```
3. Evaluate performance:
    ```python
    evaluate(model, test_loader, mode='MODULAR')
    ```

### Accuracy Goals
- **CNN (digit recognition):** >95%
- **CNN-RNN (caption generation):** High accuracy on captioning digits.

## Inference
Generate captions for unseen digits:
```python
infer(model, num_images=10)
