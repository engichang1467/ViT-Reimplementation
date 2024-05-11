# Vision Transformer Architecture Reimplementation

This project implements the Vision Transformer (ViT) architecture from the paper [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al. The Vision Transformer is a transformer-based model that applies the transformer architecture, originally developed for natural language processing tasks, to image recognition tasks.

![figure](img/vit_figure.png)

### Key Features

- **Transformer Architecture**: The ViT model uses a standard transformer encoder architecture, treating an image as a sequence of patches and encoding the patches using a transformer encoder.

- **Image Patch Embedding**: The input image is split into fixed-size patches, which are then linearly embedded and serve as the input sequence for the transformer encoder.

- **Position Embeddings**: To retain positional information, learnable position embeddings are added to the patch embeddings.

- **Pre-training on Large Datasets**: The ViT model can be pre-trained on large datasets like ImageNet and then fine-tuned on downstream tasks. (I'm using [CIFAR dataset](https://huggingface.co/datasets/cifar10) for training)

## Installation

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Training the Model

To train the model, execute the following command:

```
python train.py
```

## Dataset

This implementation uses the [CIFAR10 dataset](https://huggingface.co/datasets/cifar10), a collection of images consist of 60000 32x32 colour images in 10 classes, with 6000 images per class.