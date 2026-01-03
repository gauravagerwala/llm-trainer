# Language Model Trainer

A Python package for training and using small transformer-based language models. This package provides a clean, modular interface for data preparation, model training, and text generation.

## Features

- **Simple Word Tokenization**: Tokenize text datasets into words (seprated by spaces) with support for padding and unknown tokens
- **Data Preparation**: Load, tokenize, pad, and batch datasets for training
- **Model Training**: Train transformer-based language models with configurable hyperparameters
- **Text Generation**: Generate text from trained models using greedy or random sampling

## Quick Start

### Training a Model

Train a language model on a dataset:

```bash
python -m slm_trainer.main train \
    --dataset "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json" \
    --output-dir ./models \
    --epochs 200 \
    --batch-size 32 \
    --max-length 300
```

### Generating Text

Generate text from a trained model:

```bash
python -m slm_trainer.main generate \
    --model ./models/model_weights.weights.h5 \
    --tokenizer ./models/tokenizer_vocab.txt \
    --prompt "Abeni, a bright-eyed" \
    --num-tokens 100 \
    --sampling-mode greedy
```

## Configuration

### Training Parameters

- `max_length`: Maximum sequence length (default: 300)
- `batch_size`: Number of examples per batch (default: 32)
- `epochs`: Number of training epochs (default: 200)
- `learning_rate`: Learning rate for optimizer (default: 1e-4)
- `embedding_dim`: Embedding dimension (default: 256)
- `mlp_dim`: MLP dimension (default: 256)
- `num_heads`: Number of attention heads (default: 2)
- `num_blocks`: Number of transformer blocks (default: 1)

### Generation Parameters

- `sampling_mode`: Either "greedy" (always pick most likely token) or "random" (sample from distribution)
- `num_tokens`: Number of tokens to generate

## Dependencies

- **tensorflow**: For data handling and batching
- **keras**: For model definition and training
- **pandas**: For dataset loading
- **numpy**: For numerical operations
- **jax**: Backend for Keras

All necessary transformer layers, training utilities, and generation functions are included in this package.

## Notes

- The model uses JAX as the Keras backend by default
- Training can be time-consuming; consider using a GPU if available
- The model architecture is a small transformer (~3.5M parameters)
- For best results, train for at least 200 epochs