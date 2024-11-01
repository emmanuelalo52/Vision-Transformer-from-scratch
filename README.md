
# Vision Transformer (ViT) Implementation in PyTorch

This project is a PyTorch implementation of a Vision Transformer (ViT) model, based on the original ViT architecture. ViT applies the transformer model, traditionally used in NLP, to images, achieving state-of-the-art performance on various vision tasks.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Model Overview](#model-overview)
7. [Training and Evaluation](#training-and-evaluation)
8. [License](#license)

---

### Project Structure

- **ViT.py**: Contains the full implementation of the Vision Transformer model, including configurations, model architecture, and helper functions.

### Features

- **Configurable Model**: Define the transformer parameters, image dimensions, patch sizes, and number of classes.
- **Self-Attention Mechanism**: Implements multi-head self-attention to process image patches.
- **Training on Multiple Devices**: Supports training on CPU, CUDA (GPU), and MPS (Mac GPU).

### Installation

Ensure you have Python 3.7+ and PyTorch installed.

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd vision-transformer-pytorch
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision transformers numpy datasets
   ```

### Usage

1. **Configuration**: Modify the `VITConfig` class to set parameters such as image size, patch size, embedding size, number of heads, layers, and classes.

2. **Device Selection**: The script automatically detects the available device (CPU, CUDA, or MPS).

3. **Training**: Use the provided dataset loader to train the model on your image dataset.

### Configuration

All configurable parameters are in the `VITConfig` class:

- `n_emb`: Embedding size for each patch.
- `image_size`: Input image size (height and width should match).
- `patch_size`: Size of each image patch.
- `n_heads`: Number of attention heads.
- `n_layers`: Number of transformer encoder layers.
- `num_classes`: Number of classes for classification.

Example:
```python
config = VITConfig(
    n_emb=768,
    image_size=224,
    n_heads=12,
    patch_size=16,
    n_layers=12,
    num_classes=10
)
```

### Model Overview

The Vision Transformer consists of the following components:

1. **Patch Embedding**: Splits the image into patches and embeds them into the transformer input dimension.
2. **Transformer Encoder**: Applies multiple self-attention layers with residual connections.
3. **Classification Head**: Maps the transformer output to the number of classes.

### Training and Evaluation

1. **Load Data**: Use the `torchvision.datasets` and `torch.utils.data.DataLoader` modules to load your image dataset.
2. **Train**: Define a training loop with an optimizer and a loss function, such as cross-entropy.
3. **Evaluate**: After training, evaluate the model on a validation/test set to assess performance.

### License

This project is open-source and available under the MIT License.
