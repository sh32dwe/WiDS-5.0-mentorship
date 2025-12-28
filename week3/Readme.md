# Week 3 — Bridging CV + NLP for Image Captioning and Explainability

Welcome to Week 3! 🚀

Welcome to the most exciting week of your Image Captioning journey! This is where everything comes together. We've built strong foundations in Computer Vision (Week 1) and Natural Language Processing (Week 2). Now, we bridge the two worlds to create models that not only "see" images but can *describe* them in natural language.

Image Captioning is a multimodal task: it combines visual understanding (what's in the image?) with language generation (how to describe it coherently?). You'll learn classic and modern architectures, attention mechanisms that align visual regions with words, and even how to *explain* what your model is "looking at" when generating captions.

By the end of this week, you'll have the complete toolkit to build, train, and interpret your own image captioning model.

## 🖼️ What is Image Captioning?

Image Captioning is the task of automatically generating a natural language description of an image. It requires the model to:

- Understand visual content (objects, scenes, actions, attributes)
- Reason about relationships (e.g., "a dog chasing a ball" vs. just "dog" and "ball")
- Generate fluent, grammatically correct sentences

**Real-world applications:**

- Accessibility: Helping visually impaired users understand images
- Content moderation & search: Automatic tagging and retrieval
- Social media: Auto-suggestions for photo descriptions
- Robotics: Enabling agents to describe their environment

**Classic datasets:** Flickr8k, Flickr30k, MS COCO (we'll use COCO later for hands-on!)

## Topics We'll Cover This Week

- Vision Transformers (ViT) introduction
- Encoder–Decoder architectures for seq2seq multimodal tasks
- Using CNN or ViT as image encoder
- Using LSTM or Transformer as caption decoder
- Attention mechanisms in caption generation
- Explainability (XAI): Understanding model decisions

## 🗓️ Week 3: Learning Resources & Hands-On Prep

This week involves more code and visualization. Make sure your environment (Colab or local GPU) is ready with PyTorch, Transformers, and OpenCV from previous weeks.

**Additional libraries you might need:**

```bash
pip install timm einops matplotlib seaborn
```

Let's dive in!

### Vision Transformers (ViT) - Introduction

Traditional CNNs use convolutions to capture local patterns. Vision Transformers treat images as sequences of patches and apply the same self-attention mechanism that revolutionized NLP.

**How ViT Works:**

1. Split image into fixed-size patches (e.g., 16×16)
2. Flatten & linearly embed patches
3. Add positional encodings
4. Add a [CLS] token for global representation
5. Process through Transformer encoder layers (multi-head self-attention + FFN)
6. Use [CLS] token or pooled output for classification/feature extraction

**Advantages over CNNs:**

- Better scaling with data
- Global receptive field from the start
- Often superior performance on large datasets

**In captioning:** ViT replaces (or complements) CNN as a powerful image encoder.

**Learning Resources:**

- Original Paper: ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (Must read summary)
- Illustrated ViT: Detailed visual explanation
- [ViT Explained - YouTube](https://www.youtube.com/watch?v=4PDkBx8z4dM) (Great intuition with animations)
- [How Vision Transformers Work - YouTube](https://www.youtube.com/watch?v=1hG8oC4mpG8) (Code walkthrough)
- PyTorch Implementation with Timm: Easy pretrained ViT loading

**Quick Code Snippet (Pretrained ViT):**

```python
import torch
from timm import create_model

vit = create_model('vit_base_patch16_224', pretrained=True)
vit.eval()

# Dummy image (3, 224, 224)
img = torch.randn(1, 3, 224, 224)
features = vit(img)  # [1, 768] global feature + patch features depending on return type
print(features.shape)
```

### Encoder–Decoder Architecture

The backbone of image captioning: A visual encoder extracts image features, a language decoder generates words autoregressively.

**Two main paradigms:**

1. CNN + RNN (Classic: ResNet encoder + LSTM decoder)
2. Transformer-based (Modern: ViT encoder + Transformer decoder)

The decoder attends to encoder features while generating each word.

**Why Encoder-Decoder?**

- Handles variable-length inputs (fixed image → variable caption length)
- Allows cross-modal attention (vision ↔ language)

**Resources:**

- ["Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"](https://arxiv.org/abs/1502.03044) (Seminal paper)
- Encoder-Decoder Seq2Seq for Captioning - YouTube
- From CNN+RNN to Transformer Captioning - Medium overview

### CNN/ViT as the Image Encoder

The encoder produces rich spatial (or patch-wise) features.

**For CNN (e.g., ResNet50):**

- Remove final classification layer
- Extract feature maps from last conv layer (e.g., 7×7×2048)
- Flatten spatially → sequence of region features

**For ViT:**

- Use patch embeddings directly as sequence
- Optionally use [CLS] token for global feature

Pretrained encoders (ImageNet) give strong zero-shot features via transfer learning!

**Code Example (CNN Encoder):**

```python
from torchvision.models import resnet50
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove FC & avgpool
        self.resnet = nn.Sequential(*modules)
        self.adapt = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, images):
        features = self.resnet(images)          # [B, 2048, H, W]
        features = self.adapt(features)
        features = self.relu(features)
        features = features.permute(0, 2, 3, 1)  # [B, H, W, embed]
        return features.view(features.size(0), -1, features.size(3))  # [B, L, embed]
```

### LSTM/Transformer as the Caption Decoder

Decoder generates words one-by-one, conditioned on previous words and image features.

**LSTM Decoder (Classic):**

- Takes previous word embedding + attention-weighted image features
- Hidden state carries context

**Transformer Decoder (Modern):**

- Uses masked self-attention on partial caption
- Cross-attention to image patch features
- Often better coherence and fluency

**Resources:**

- LSTM Captioning Tutorial - PyTorch Official
- Transformer Captioning from Scratch - YouTube
- BLIP/BLIP-2 models (State-of-the-art multimodal)

### Attention Mechanism for Caption Generation

The magic glue! Attention lets the decoder "look" at relevant image regions when generating each word.

**Types:**

- Soft Attention: Weighted average of all region features
- Hard Attention: Sample one region (stochastic, harder to train)
- Multi-head: Different attention patterns

Bahdanau vs. Luong attention styles.

Visualizes word-to-region alignment perfectly for explainability.

**Resources:**

- Visual Attention in Captioning - Original "Show, Attend and Tell"
- Attention Mechanism Animation - Great visuals
- How Attention Works in Captioning - YouTube

### Explainability (XAI) in Image Captioning

Why does the model generate certain words? XAI helps debug, build trust, and improve models.

**Key Techniques:**

1. **Grad-CAM for CNNs**
   - Gradient-weighted Class Activation Mapping
   - Highlights image regions important for a specific output (word/class)
   - Works on CNN encoders

   **Resources:**
   - Grad-CAM Paper
   - Grad-CAM Tutorial - PyTorch
   - Visualizing CNN Decisions - YouTube

2. **Attention Heatmaps**
   - Directly visualize attention weights from decoder → encoder
   - Shows which image regions influenced each generated word

   **Code Snippet:**

   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   # attention_weights: [seq_len, num_regions]
   sns.heatmap(attention_weights, cmap="viridis")
   plt.xlabel("Image Regions")
   plt.ylabel("Generated Words")
   plt.show()
   ```

3. **Visualizing Word-to-Region Alignment**
   - Overlay attention maps on original image
   - Tools: Use OpenCV to draw heatmaps

   **Resources:**
   - Attention Rollout & Relevance - Advanced methods
   - XAI for Multimodal Models - Survey

**Hands-On Project Idea:**

Build a simple CNN+LSTM+Attention captioner on Flickr8k, then visualize:

- Grad-CAM for object words
- Attention maps for spatial words ("left", "behind")
- Compare CNN vs ViT encoders

**Helpful Repos & Tutorials:**

- [PyTorch Image Captioning Tutorial (Official)](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) (Adapt for images)
- [Bottom-Up and Top-Down Attention](https://arxiv.org/abs/1707.07998) (Classic paper implementation)
- [Hugging Face BLIP Model Demo](https://huggingface.co/Salesforce/blip-image-captioning-base) (Modern, easy to use)

Next steps: Start implementing a baseline model and play with visualizations. This is where the real fun begins!

**Happy Learning :)**
