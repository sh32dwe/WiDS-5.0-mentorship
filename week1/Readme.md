# 📚 Week 1 — Introduction to Computer Vision

## Welcome to Week 1! 🚀

Welcome to the first week of your Image Captioning journey! This week, we go back to the roots—not of magic or AI hype, but to the fundamentals of how machines interpret the world visually.

The science of enabling machines to comprehend visual input, such as pictures and videos, so they can interpret what they "see," is known as computer vision. However, in order to understand this, we must first address two fundamental questions:

-------------------------------------------

## 📷 What is Computer Vision?
Computer vision is **an AI domain enabling computers to interpret and understand the visual world from images and videos**, mimicking human sight by identifying objects, detecting patterns, and making decisions based on visual input, powering applications like self-driving cars, facial recognition, and automated quality control in manufacturing. It uses deep learning, neural networks, and algorithms to process pixel data, allowing machines to "see" and react to their environment, transforming various industries.

From recognizing faces in your phone to detecting objects in self-driving cars, CV powers some of the most exciting technologies today.

At its core, Computer Vision teaches machines to answer questions like:
* What is in this image?
* Where is it located?
* What action is happening?
* How does the appearance change?

**Applications of Computer Vision**
* **Medical Imagening**: helps in MRI reconstruction, automatic pathology, diagnosis, and computer aided surgeries and more.
* **AR/VR**: Object occlusion, outside-in tracking, and inside-out tracking for virtual and augmented reality.
* **Smartphones** : All the photo filters (including animation filters on social media), QR code scanners, panorama construction, Computational photography, face detectors, image detectors like (Google Lens, Night Sight) that we use are computer vision applications.
* **Internet**: Image search, Mapping, photo captioning, Ariel imaging for maps, video categorization and more....

To solve these problems, CV uses powerful models such as:
* **Convolutional Neural Networks (CNNs)** — specialized neural networks that extract rich features from images.
* **Vision Transformers (ViT)** — transformer-based architectures adapted for image understanding.
* **Transfer Learning** — using pretrained models like ResNet or EfficientNet to leverage prior knowledge.

This week, you’ll explore the foundations that make these models work.  

---------------------------------------------------------
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

---

## OpenCV for Computer Vision
OpenCV (Open Source Computer Vision), a cross-platform and free to use library of functions is based on real-time Computer Vision which supports Deep Learning frameworks that aids in image and video processing.  In Computer Vision, the principal element is to extract the pixels from the image to study the objects and thus understand what it contains.

Below are a few key aspects that Computer Vision seeks to recognize in the photographs:
* Object Detection: The location of the object.
* Object Recognition: The objects in the image, and their positions.
* Object Classification: The broad category that the object lies in.
* Object Segmentation: The pixels belonging to that object.

To have this library make sure the latest version python and pip (python package installer) is already installed on your device.

To install OpenCV, just go to the command-line and type the following commands:
1. On Windows/MacOS: **pip install opencv-python**
2. On Linux: **pip3 install opencv-python**
3. [Set up Opencv with anaconda environment](https://www.geeksforgeeks.org/set-opencv-anaconda-environment/)

[OpenCV Tutorial](https://www.youtube.com/watch?v=oXlwWbU8l2o&t=00s)

[OpenCV: Python Documentation](https://docs.opencv.org/4.12.0/d6/d00/tutorial_py_root.html)

## PyTorch for Deep Learning

PyTorch is a popular open-source deep learning framework known for its flexibility and ease of use. It supports dynamic computation graphs, making it ideal for research and experimentation in NLP, computer vision, and reinforcement learning. Below are some recommended resources to get started — use as many as you need

 * [[Codemy.com – Deep Learning With PyTorch]](https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1) : First 17 videos are sufficient :)
 * [[PyTorch – PyTorch Beginner Series]](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN)
 * [[Official PyTorch Documentation]](https://docs.pytorch.org/docs/stable/index.html) : Always go through the documentation to understand and fix your errors, you can use ChatGPT/Microsoft Bing/Copilot for Code Error Correction too, however, it gets too easy then :)

## Image Preprocessing and Transformation
Image Preprocessing is one of the most important parts of the Computer Vision pipeline.
Preprocessing prepares raw images into forms suitable for training.

### 🔧 Key Elements of Image Preprocessing
1. **Image Resizing**
    * Make all images the same shape.
    * Ex: Resize to ***224×224** for CNNs (ResNet, VGG)
2. **Normalization**
    * Scale pixel range (0–255 → 0–1 or standardized)
    *  Example of PyTorch:
       ```
       mean = [0.485, 0.456, 0.406]
       std  = [0.229, 0.224, 0.225]
       ```
3. **Data Augmentation (for generalization)**
    * Random Crop
    * Random Flip
    * Rotation
    * Color Jitter
    * Gaussian Noise
    * Cutout/ Cutmix/ Mixup

    This prevents model from overfitting.
4. **Denoising/ Filtering**
   * Gaussian Blur
   * Median Filter
   * Bilateral Filter
  
    Used when Image contain Noise.
5. **Image Transformation**
    * Grayscale conversion
    * Convert between BGR ↔ RGB (OpenCV images are BGR!)
  
6. **Edge / Feature Enhancements:**
Useful in early CV tasks or classical ML pipelines (In modern Architectures we allow our model to learn these filter values from the data itself as a learnable parameter instead of manually implementing them.)
    * Sobel filter
    * Canny edge detector
    * Laplacian

All these techniques can be implemented with help of **OpenCV and PyTorch**.

[The Complete Guide to Image Preprocessing Techniques in Python](https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c)

[Image Processing with OpenCV and Python](https://www.youtube.com/watch?v=kSqxn6zGE0c&t=00s)

[Image Transformations using OpenCV in Python](https://www.geeksforgeeks.org/python/image-transformations-using-opencv-in-python/)

#### 📌 Important Note on Image Preprocessing and Image Transformation
Image preprocessing and image transformation are important components of a computer vision pipeline, but it is essential to understand that **not all techniques are applied together**. The choice of preprocessing steps depends on several factors, including the **model architecture, task requirements, size and variability of the dataset, and the model’s capacity (number of parameters).**

**Image preprocessing**—such as resizing, normalization, or noise reduction—ensures that input images are consistent and suitable for training. I**mage transformations**—often used as data augmentation—introduce variations through operations like flipping, rotation, cropping, or color adjustments to improve generalization.

However, the selection of these steps must be deliberate.

For example:
* Models like Vision Transformers may require fixed-size inputs and benefit from stronger augmentations, while CNNs may perform adequately with simpler preprocessing.
* Tasks sensitive to spatial or color information may restrict the types of transformations allowed.
* Small datasets often require more augmentation to avoid overfitting, whereas large datasets typically need only minimal preprocessing.
* Large models with many parameters may demand stronger augmentation to prevent overfitting, while smaller models may require simpler inputs for stable training.

In summary, **image preprocessing and transformation are not one-size-fits-all**. They should be carefully selected to align with the model, the dataset, and the task to ensure effective and reliable training.


## Neural Network and CNN (Convolutional Neural Network) in Computer Vision

### Neural Network and Deep Neural Network
Inspired by the human brain, neural networks use interconnected artificial “neurons” to process inputs and generate outputs. They power applications like image recognition, language processing, and decision-making. Explore these resources to understand how they enable intelligent systems.

1. [[Neural Networks--3Blue1Brown]](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=9lcJKb1AlcvvPxwZ) : First 4 videos should suffice.
2. [[The spelled-out intro to neural networks and backpropagation: Building micrograd--
Andrej Karpathy]](https://www.youtube.com/watch?v=VMj-3S1tku0&t=00s) : This is the most step-by-step spelled-out explanation of backpropagation and training of neural networks. It only assumes basic knowledge of Python and a vague recollection of calculus.
3. [Implementation of  Basic Neural Network with single hidden layer](https://www.youtube.com/watch?v=mlk0rddP3L4&list=PLuhqtP7jdD8CftMk831qdE8BlIteSaNzD) : First 7 videos should suffice

### Convolutional Neural Networks
A Convolutional Neural Network (ConvNet) is a class of deep learning models designed to process data with a grid-like topology, such as images. It is a foundational technology for most modern computer vision applications and is inspired by the human visual cortex.

They are used in a wide range of fields:
* Image classification, Segmentation and object detection (e.g. in self-driving cars, social media tagging)
* Medical image analysis (e.g. detecting tumors in X-rays or MRIs)
* Natural language processing and speech recognition (e.g. virtual assistants)

**Learning Resources**
Follow as many as you can:

1. [Convolutional Neural Networks: A Comprehensive Guide (Medium)](https://medium.com/thedeephub/convolutional-neural-networks-a-comprehensive-guide-5cc0b5eae175)
2. [CNN Explainer: Visual Explaination of Convolutional Neural Networks](https://poloclub.github.io/cnn-explainer/)
3. [CNN From Scratch](https://youtu.be/Lakz2MoHy6o?si=aVOlYgqkc3PqUtk0)
4. [MIT 6.S191: Convolutional Neural Networks](https://youtu.be/oGpzWAlP5p0?si=SkiLrZonrfxkBppr)

Happy learning! 🚀
