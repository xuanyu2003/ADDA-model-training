# ADDA: MNIST → USPS Domain Adaptation
*PyTorch · Google Colab GPU · Adversarial Learning*

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ADDA-orange.svg)
![Colab](https://img.shields.io/badge/Run%20on-Colab-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_GcIWExywUmpGBXLuWKvryhNJ_eAWdJX?usp=sharing)

This project implements **ADversarial Domain Adaptation (ADDA)** to transfer a digit classifier trained on **MNIST (source)** to **USPS (target)**.  
Using a lightweight CNN and adversarial feature alignment, the adapted model reaches **~95% accuracy** on USPS.  
Designed to run efficiently on **Google Colab GPU**.

---

## 🚀 Features
- Full ADDA pipeline implemented in PyTorch  
- MNIST → USPS domain adaptation  
- GPU-accelerated training on Google Colab  
- Latent feature extraction from encoders  
- t-SNE visualizations: before & after adaptation  
- Checkpoint saving:
  - `src_encoder.pth`
  - `tgt_encoder_adapted.pth`
  - `classifier.pth`

---

## 📌 Method Overview

### 1. Source Pretraining
Train a CNN encoder + classifier on MNIST to learn clean, discriminative features.

### 2. Adversarial Adaptation
Freeze the source encoder and train:
- a **domain discriminator** (source vs. target)  
- a **target encoder** to fool the discriminator  

This aligns USPS features with the MNIST feature space.

### 3. Target Classification
Use the MNIST-trained classifier directly on adapted USPS features.

---

<img width="1104" height="563" alt="minist-usps" src="https://github.com/user-attachments/assets/cb387611-4d98-463e-8ec8-3c4ab134b78a" />

