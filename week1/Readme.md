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


---

## PyTorch for Deep Learning

PyTorch is a popular open-source deep learning framework known for its flexibility and ease of use. It supports dynamic computation graphs, making it ideal for research and experimentation in NLP, computer vision, and reinforcement learning. Below are some recommended resources to get started — use as many as you need

 * [[Codemy.com – Deep Learning With PyTorch]](https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1) : First 17 videos are sufficient :)
 * [[PyTorch – PyTorch Beginner Series]](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN)
 * [[Official PyTorch Documentation]](https://docs.pytorch.org/docs/stable/index.html) : Always go through the documentation to understand and fix your errors, you can use ChatGPT/Microsoft Bing/Copilot for Code Error Correction too, however, it gets too easy then :)


## Neural Network and CNN (Convolutional Neural Network) in Computer Vision

### Neural Network and Deep Neural Network
Inspired by the human brain, neural networks use interconnected artificial “neurons” to process inputs and generate outputs. They power applications like image recognition, language processing, and decision-making. Explore these resources to understand how they enable intelligent systems.

1. [[Neural Networks--3Blue1Brown]](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=9lcJKb1AlcvvPxwZ) : First 4 videos should suffice.
2. [[The spelled-out intro to neural networks and backpropagation: Building micrograd--
Andrej Karpathy]](https://youtu.be/VMj-3S1tku0?si=GvBjrYHb4wPrZX5M) : This is the most step-by-step spelled-out explanation of backpropagation and training of neural networks. It only assumes basic knowledge of Python and a vague recollection of calculus.
3. [[Implementation of  Basic Neural Network with single hidden layer]](https://www.youtube.com/watch?v=mlk0rddP3L4&list=PLuhqtP7jdD8CftMk831qdE8BlIteSaNzD) : First 7 videos should suffice

### Convolutional Neural Networks
A Convolutional Neural Network (ConvNet) is a class of deep learning models designed to process data with a grid-like topology, such as images. It is a foundational technology for most modern computer vision applications and is inspired by the human visual cortex.

They are used in a wide range of fields:
* Image classification, Segmentation and object detection (e.g., in self-driving cars, social media tagging)
* Medical image analysis (e.g., detecting tumors in X-rays or MRIs)
* Natural language processing and speech recognition (e.g., virtual assistants)

**Learning Resources**
Go 
1. [Convolutional Neural Networks: A Comprehensive Guide (Medium)](https://medium.com/thedeephub/convolutional-neural-networks-a-comprehensive-guide-5cc0b5eae175)
2. [CNN Explainer: Visual Explaination of Convolutional Neural Networks](https://poloclub.github.io/cnn-explainer/)
3. [CNN From Scratch](https://youtu.be/Lakz2MoHy6o?si=aVOlYgqkc3PqUtk0)
4. [MIT 6.S191: Convolutional Neural Networks](https://youtu.be/oGpzWAlP5p0?si=SkiLrZonrfxkBppr)







