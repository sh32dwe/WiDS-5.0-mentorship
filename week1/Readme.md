# **Visual Question Answering (VQA) Project**

## 📌 Introduction

The **Virtual Question Answering (VQA)** project explores an exciting area where **Computer Vision** meets **Natural Language Processing (NLP)**. The goal is to build an intelligent system capable of looking at an image, understanding its visual content, and answering natural-language questions about it. For example:

> *“What is the man holding?”*  
> *“How many people are in the image?”*  
> *“Is the dog sitting or standing?”*

This project provides a hands-on journey into **multimodal AI**, where models process both images and text together.

---

## 📘 Project Description

This project builds a complete **end-to-end Visual Question Answering pipeline** using state-of-the-art **Vision–Language models**, such as:

- **BLIP (Bootstrapped Language-Image Pretraining)**  
- **Flamingo-style multimodal transformers**  
- **CLIP-based encoders**  

The system integrates image features and question embeddings to generate accurate and context-aware answers.

A major focus of the project is **explainability** — understanding *why* the model gives a certain answer. To accomplish this, participants will design and implement **attention-based heatmaps and visualizations** that show which parts of the image influenced the model’s decision. This helps create an **interpretable and trustworthy AI system**.

---

## 🎯 Project Outcomes

By the end of the project, mentees will:

- Build a **working VQA system** that answers questions about real-world images  
- Implement **visual attention heatmaps** for explainability  
- Evaluate model performance using:
  - Accuracy  
  - Plausibility  
  - Interpretability metrics  
- Gain hands-on experience in:
  - Multimodal learning  
  - Explainable AI (XAI)  
  - Vision-language reasoning  
  - Working with pretrained models and transformers  

---

# 🗓️ Week 1: Learning Resources & Setup

Week 1 focuses on environment setup, understanding tools, and building foundational knowledge for Vision & Language modeling.

---

## ⚙️ Option 1: Working on Google Colab

Google Colab is ideal for beginners and requires no installation. It provides free GPU access (a maximum of 12 hours per session).

### **Resources**
- Introduction to Colab  
  https://colab.research.google.com/notebooks/intro.ipynb  
- Using GPU in Colab:
 
  Change Hardware Accelerator to **T4 GPU** in Runtime settings :)
  
---

## ⚙️ Option 2: Setting Up a GPU-Enabled Conda Environment and Adding It to Jupyter Notebook

For People with an NVIDIA GPU, a local environment offers faster training and more control.

### **Installation Steps**

#### **1️⃣ Install Anaconda or Miniconda**

Download and install:

- **Anaconda:** https://www.anaconda.com/products/distribution  
- **Miniconda:** https://docs.conda.io/en/latest/miniconda.html  

#### **2️⃣ Create a Conda Environment (Recommended: Python 3.10)**
Open Anaconda Prompt and then give commands as follow:

```bash
conda create -n env_name python=3.10
conda activate env_name
```
#### **3️⃣ Install PyTorch with CUDA Support**
Use the official PyTorch CUDA-enabled wheels (example: CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Expected Output: True

#### **4️⃣ Install Additional Required Libraries**

```bash
pip install numpy matplotlib pillow opencv-python tqdm transformers jupyter ipykernel
```
#### **5️⃣ Add the Conda Environment to Jupyter Notebook**

```bash
python -m ipykernel install --user --name vqa --display-name "VQA (Python 3.10 GPU)"
```
#### **6️⃣ Launch Jupyter Notebook**
```bash
jupyter notebook
```

When Jupyter opens:

Go to
**Kernel → Change Kernel → VQA (Python 3.10 GPU)**

#### **Test GPU Access Inside Notebook**
In a notebook cell:

```bash
import torch
torch.cuda.is_available(), torch.cuda.device_name(0)
```

Expected example output:
```
(True, 'NVIDIA GeForce RTX 3060')
```

If ***False*** appears, your CUDA or PyTorch installation needs adjustment.

[Learning Resources](https://www.geeksforgeeks.org/applications-of-computer-vision/)
