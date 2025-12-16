# Transformers
Transformers are a type of deep learning model introduced in the 2017 paper ["Attention is All You Need" by Vaswani et al](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) . They revolutionized natural language processing (NLP) and are now used in many domains like vision, speech, and more.

Applications:
- NLP: BERT, GPT, T5, etc.

- Vision: Vision Transformers (ViT)

- Speech, Recommendation Systems, Genomics, etc.

## Recurrent Neural network (RNN):
In order to understand the Transformer Architecture, It is Important to understand the RNN Stucture

[Introduction to RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)

[More Indepth Intution](https://www.geeksforgeeks.org/machine-learning/recurrent-neural-networks-explanation/)

[Video Resource 1](https://www.youtube.com/watch?v=Y2wfIKQyd1I)

[Video Resource 2](https://www.youtube.com/watch?v=EzsXi4WzelI)

[Bi-directional RNN](https://www.geeksforgeeks.org/bidirectional-recurrent-neural-network/)

[Video Resource](https://youtu.be/atYPhweJ7ao?si=VlHMh2zwwqPWo5AF)

## LSTM RNN
Traditional RNNs struggle with vanishing gradients, making them bad at remembering information over long sequences.
LSTM solves this using a memory cell and gates that control information flow.

[Introduction to LSTM](https://www.geeksforgeeks.org/machine-learning/long-short-term-memory-networks-explanation/)

[More Indepth Intution](https://www.geeksforgeeks.org/machine-learning/long-short-term-memory-networks-explanation/)

[Video Resources](https://youtu.be/LfnrRPFhkuY?si=Q7tAaOkHZUwokVAV)

[GRU-RNN (optional)](https://youtu.be/tOuXgORsXJ4?si=Cv2JxQQip_lmi40h)

## Encoder-Decoder
The Encoder-Decoder is a neural network architecture designed to handle sequence-to-sequence (seq2seq) tasks — where input and output are both sequences, possibly of different lengths.

[Encoder-Decoder (Seq2Seq)](https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b)

[Video Resources](https://www.youtube.com/watch?v=L8HKweZIOmg)

## Attention Mechanism
The attention mechanism allows a model to focus on relevant parts of the input sequence when generating each part of the output.

It was introduced to improve performance in sequence-to-sequence tasks (like translation), especially in long sequences.

[Attention Mechanism Explained](https://erdem.pl/2021/05/introduction-to-attention-mechanism)

[Video Resources](https://youtu.be/PSs6nxngL6k?si=mcrMj7QjLftvYXz4)


## Transformers
### Key Concepts:
- [Self-Attention](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) :
  Allows the model to weigh the importance of different words in a sentence when encoding a specific word.

- [Positional Encoding](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) :
Since transformers don't process sequences in order (like RNNs), positional encodings are added to retain word order.

### Core Components:
- Multi-Head Attention:
Captures relationships from different subspaces in the data.

- Feed-Forward Neural Networks:
Applies transformations to each position independently.

- Layer Normalization & Residual Connections:
Helps with training stability and deeper networks.

[Transformer Architecture](https://jalammar.github.io/illustrated-transformer/)

[Research Paper (Must Read)](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)

[Video Resources](https://youtu.be/ZhAz268Hdpw?si=FCy2wMu-hOBIxIIu)

_______________________________________________________________________________________________
## What is Hugging Face?

Hugging Face is an open-source platform offering pre-trained models, datasets, and tools for NLP, computer vision, and audio tasks. Its **Transformers** library simplifies ML, and the **Model Hub** hosts community-contributed resources.

**Why Use It?**

- Beginner-friendly APIs
- Supports multiple domains
- Active community

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- Basic Python knowledge
- Optional: GPU with CUDA

### Steps

1. **Virtual environment**:

   ```bash
   python -m venv hf_env
   source hf_env/bin/activate  # Windows: hf_env\Scripts\activate
   ```

2. **Install libraries**:

   ```bash
   pip install transformers datasets tokenizers torch
   ```

3. **Verify**:

   ```python
   import transformers
   print(transformers.__version__)
   ```

**Troubleshooting**:

- **Dependency issues**: Use a fresh virtual environment.
- **GPU errors**: Ensure CUDA compatibility.

---

## Core Libraries

### Transformers

Provides pre-trained models and pipelines for tasks like text classification.

- **Example**: Sentiment analysis

  ```python
  from transformers import pipeline
  classifier = pipeline("sentiment-analysis")
  print(classifier("I love Hugging Face!"))  # [{'label': 'POSITIVE', 'score': 0.999}]
  ```

### Datasets

Access and preprocess datasets efficiently.

- **Example**: Load IMDB dataset

  ```python
  from datasets import load_dataset
  dataset = load_dataset("imdb")
  print(dataset["train"][0])
  ```

### Tokenizers

Converts text to model inputs.

- **Example**: Tokenize text

  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  tokens = tokenizer("Hello, Hugging Face!", return_tensors="pt")
  print(tokens)
  ```

---

## Model Loading and Inference

### Basic Loading

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Inference with Pipeline

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model=model_name)
print(classifier("Great tutorial!"))  # [{'label': 'POSITIVE', 'score': 0.999}]
```

### Cross-Domain Examples

- **NLP (Text Classification)**:

  ```python
  classifier = pipeline("text-classification")
  print(classifier("Fun movie!"))
  ```

- **Vision (Image Classification)**:

  ```python
  vision_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
  print(vision_classifier("https://example.com/cat.jpg"))
  ```

**Warning**: Match model to task (e.g., text model for text tasks).

---

## Fine-Tuning

Fine-tune a model on a custom dataset.

### Steps

1. **Load dataset**:

   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb")
   ```

2. **Load model**:

   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   model_name = "distilbert-base-uncased"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
   ```

3. **Preprocess**:

   ```python
   def tokenize(examples):
       return tokenizer(examples["text"], padding="max_length", truncation=True)
   tokenized_dataset = dataset.map(tokenize, batched=True)
   ```

4. **Train**:

   ```python
   from transformers import Trainer, TrainingArguments
   training_args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       learning_rate=2e-5,
       per_device_train_batch_size=8,
       num_train_epochs=3,
   )
   trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"])
   trainer.train()
   ```

5. **Save**:

   ```python
   model.save_pretrained("./fine_tuned_model")
   ```

**Best Practices**:

- Use small learning rates (e.g., 2e-5).
- Monitor validation loss to avoid overfitting.

---

## Model Sharing

Share models on the **Model Hub**.

1. **Log in**:

   ```python
   from huggingface_hub import login
   login()  # Use your Hugging Face token
   ```

2. **Push model**:

   ```python
   model.push_to_hub("my-model")
   tokenizer.push_to_hub("my-model")
   ```

---

## Use Cases

- **NLP**: Sentiment analysis

  ```python
  classifier = pipeline("sentiment-analysis")
  print(classifier(["Great book!", "Boring plot."]))
  ```

- **Vision**: Object detection

  ```python
  classifier = pipeline("image-classification")
  print(classifier("https://example.com/dog.jpg"))
  ```

---

## Helpful Resources 

- Hugging Face in 10 Minutes - https://www.youtube.com/watch?v=9gBC9R-msAk
- Official Hugging Face Course - https://huggingface.co/learn/nlp-course
- CodeBasics Transformers & HF playlist (very beginner friendly) - https://www.youtube.com/playlist?list=PLKnIA16_Rmvb7F5cnA6WhgZfz3BlvkxLx
- Crash Course for Hugging face (in 1 hour) - https://www.youtube.com/watch?v=b665B04CWkI 
- Hugging Face GitHub Repo - https://github.com/huggingface/awesome-huggingface

---

## Learning Path

1. **Beginner**: Install libraries, try pipelines, explore Model Hub.
2. **Intermediate**: Fine-tune models, experiment with vision/audio tasks.
3. **Advanced**: Build ML pipelines, contribute to the community.

This guide equips you to leverage Hugging Face for ML projects.

________________________________________________________________________________________________
# Overview of Generative Adversarial Networks (GANs) (Advanced and Optional)

Generative Adversarial Networks (GANs) represent a revolutionary approach to generative machine learning that has transformed how we think about creating synthetic data. While primarily known for image generation, GANs have important applications in Natural Language Processing and serve as a foundation for understanding modern generative models.

In this guide, we explore key learning materials and structured tutorials to help you understand GANs and their relevance to NLP applications.

---

## GANs in NLP Context
While GANs are most famous for image generation, understanding their principles is valuable for NLP practitioners working with modern generative models.

### Relevance to Text Generation
GANs have influenced the development of text generation models and provide important insights into:
- Adversarial Training: The concept of training competing networks has influenced techniques like adversarial training for robust NLP models
- Generative Modeling: Understanding how GANs generate data helps in comprehending more advanced text generation approaches
- Quality Assessment: The discriminator concept parallels techniques used to evaluate generated text quality

### Connection to Modern NLP
While transformer-based models like GPT have largely superseded GANs for text generation, the adversarial training principles remain relevant in:
- Training robust language models
- Generating synthetic training data
- Understanding generative model evaluation
- Preparing for advanced topics like diffusion models in NLP

---

## Understanding the Fundamentals of GANs

Before diving into complex implementations, it's essential to grasp the core concepts behind how GANs work, including the adversarial training process and the interplay between generator and discriminator networks.

- [A Friendly Introduction to GANs - YouTube](https://www.youtube.com/watch?v=8L11aMN5KY8) <br>
This video provides an intuitive explanation of GANs using simple examples, perfect for beginners. It demonstrates how to build basic GANs from scratch using minimal code, making the concepts accessible and easy to understand. **Please refer to the github repo given in the video description for the implementation.**

---

## Practical Implementations

- [Build a FashionGAN – YouTube](https://www.youtube.com/watch?v=AALBGpLbj6Q) <br>
A comprehensive hands-on tutorial that walks through building a complete GAN implementation using TensorFlow. This video covers environment setup, data visualization, neural network architecture, custom training loops, and image generation.
- [O'Rielly Tutorial on GANs](https://github.com/jonbruner/generative-adversarial-networks) <br>
A hands-on tutorial with clean TensorFlow/Keras implementations of GANs, including MLP and CNN-based models. Ideal for understanding adversarial training dynamics, even if examples are image-focused.

---

## Additional Reading

For a more in-depth theoretical explanation of GANs, refer to the following academic resources:
- [Original 2014 GAN Paper by Ian J. Goodfellow](https://arxiv.org/pdf/1701.00160)
- [NIPS 2016 Tutorial: Generative Adversarial Networks](http://arxiv.org/pdf/1701.00160)

---

This module introduces GANs from a conceptual and practical lens, giving you both intuition and code resources to explore further. Happy learning!


