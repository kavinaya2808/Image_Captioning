# Image_Captioning with Deep Learning
This project explores the task of image captioning, where a model generates natural language descriptions for input images. Several deep learning architectures are implemented and evaluated, ranging from simple CNN + RNN models to Transformer-based caption generators.


# To compare different image captioning architectures and evaluate their performance using:

Training/validation loss curves
BLEU scores
Qualitative caption examples

# Models Implemented

# Model	Description
Model 1	Baseline CNN + RNN
Model 2	Improved CNN + RNN with better regularization
Model 3	Bidirectional RNN
Model 4	Attention-based encoder-decoder
Model 5	Transformer-based model
Model 6	Vision Transformer (ViT) + Transformer Decoder

# Evaluation Highlights:
Loss Curves: Transformer-based models (Models 5 and 6) achieved faster convergence and more stable validation losses.
BLEU Scores: Model 6 outperformed all others on BLEU-1 to BLEU-4 metrics.
Generated Captions: Qualitative results show more coherent and descriptive captions from attention-based and Transformer models.

# Dataset:
flickr8k Dataset
Preprocessed using standard techniques (resizing, tokenization, padding)

# Report Contents:

Training/validation performance curves
Comparison of BLEU scores across all models
Sample generated captions on test images
Key observations and insights

# Requirements:
  torch
  torchvision
  numpy
  matplotlib
  nltk
  transformers
  scikit-learn

# Future Improvements:
Use larger Transformer models (e.g., ViT-B, GPT-2 decoder)
Integrate image features from CLIP or DINO
Train on a larger dataset (full MS-COCO)
Apply beam search for caption generation
