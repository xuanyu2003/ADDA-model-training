ADDA: MNIST → USPS Domain Adaptation (Colab GPU)

This project implements ADversarial Domain Adaptation (ADDA) to transfer a classifier trained on MNIST (source domain) to USPS (target domain).
The method trains:

A source encoder + classifier on MNIST.

A target encoder initialized from the source encoder.

A domain discriminator trained adversarially to align MNIST and USPS latent feature distributions.

The experiment is implemented in PyTorch and runs efficiently on Google Colab GPU.
After adaptation, the model achieves ~95% accuracy on USPS.

🚀 Features

Full ADDA pipeline (PyTorch)

MNIST → USPS domain adaptation

GPU-accelerated training in Google Colab

Latent feature extraction from encoder

Two t-SNE visualizations:

Initial USPS features (before ADDA)

Adapted USPS features (after ADDA)

Model checkpoint saving:

src_encoder.pth

tgt_encoder_adapted.pth

classifier.pth

📌 Method Overview

ADDA follows a three-stage training process:

1. Source Pretraining

Train a CNN encoder + classifier on MNIST (supervised).
This encoder represents clean, well-separated digit features.

2. Adversarial Adaptation

Freeze the source encoder.
Train:

A discriminator to classify source vs target latent features.

A target encoder to fool the discriminator (make USPS features look like MNIST features).

This aligns the latent feature spaces.

3. Target Classification

After adaptation, the same MNIST-trained classifier is used to classify USPS images.

This achieves ~95% USPS accuracy with a lightweight architecture.
<img width="1104" height="563" alt="minist-usps" src="https://github.com/user-attachments/assets/cb387611-4d98-463e-8ec8-3c4ab134b78a" />

