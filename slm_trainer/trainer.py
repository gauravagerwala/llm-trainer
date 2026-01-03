"""Training utilities for small language models.

This module provides a high-level interface for training transformer-based
language models.
"""

import os
from typing import Any, Dict, List, Optional

import keras
import tensorflow as tf

# Set Keras backend to JAX if not already set.
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

from .tokenizer import SimpleWordTokenizer
from .training import create_model, TextGenerator as TrainingTextGenerator


class ModelTrainer:
    """Trainer for small language models.

    This class provides a high-level interface for creating and training
    transformer-based language models.

    Attributes:
        model: The compiled Keras model.
        tokenizer: The tokenizer used for encoding text.
        max_length: Maximum sequence length.
        vocabulary_size: Size of the vocabulary.
    """

    def __init__(
        self,
        tokenizer: SimpleWordTokenizer,
        max_length: int,
        vocabulary_size: Optional[int] = None,
        learning_rate: float = 1e-4,
        embedding_dim: int = 256,
        mlp_dim: int = 256,
        num_heads: int = 2,
        num_blocks: int = 1,
        dropout_rate: float = 0.0,
        activation_function: str = "relu",
        optimizer: str = "adamw",
        random_seed: int = 812,
    ):
        """Initialize the ModelTrainer.

        Args:
            tokenizer: The tokenizer used for encoding text.
            max_length: Maximum sequence length for the model.
            vocabulary_size: Size of the vocabulary. If None, uses
                tokenizer.vocabulary_size.
            learning_rate: Learning rate for the optimizer.
            embedding_dim: Dimensionality of the embedding space.
            mlp_dim: Number of units in the feed-forward network.
            num_heads: Number of attention heads.
            num_blocks: Number of transformer blocks.
            dropout_rate: Dropout rate to prevent overfitting.
            activation_function: Activation function for the feed-forward network.
            optimizer: Optimizer to use ('adamw' or 'sgd').
            random_seed: Random seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size or tokenizer.vocabulary_size

        # Set random seed for reproducibility.
        keras.utils.set_random_seed(random_seed)

        # Create the model.
        self.model = create_model(
            vocabulary_size=self.vocabulary_size,
            max_length=self.max_length,
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            optimizer=optimizer,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            pad_token_id=tokenizer.pad_token_id,
        )

    def train(
        self,
        batches: tf.data.Dataset,
        epochs: int = 200,
        verbose: int = 2,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
    ) -> keras.callbacks.History:
        """Train the model on the provided batches.

        Args:
            batches: Batched TensorFlow dataset for training.
            epochs: Number of training epochs.
            verbose: Verbosity mode (0, 1, or 2).
            callbacks: Optional list of Keras callbacks.

        Returns:
            Training history object.
        """
        if callbacks is None:
            callbacks = []

        history = self.model.fit(
            x=batches,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

        return history

    def create_text_generator_callback(
        self,
        prompt: str,
        max_tokens: int = 10,
        print_every: int = 10,
    ) -> keras.callbacks.Callback:
        """Create a callback that generates text during training.

        Args:
            prompt: Initial prompt for text generation.
            max_tokens: Maximum number of tokens to generate.
            print_every: Print generated text every N epochs.

        Returns:
            TextGenerator callback instance.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        return TrainingTextGenerator(
            max_tokens=max_tokens,
            start_tokens=prompt_ids,
            tokenizer=self.tokenizer,
            print_every=print_every,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: Path where to save the model.
        """
        # Ensure filepath ends with .weights.h5 for Keras compatibility
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')

        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk.

        Args:
            filepath: Path to the saved model weights.
        """
        self.model.load_weights(filepath)
        print(f"Model loaded from {filepath}")

    def get_model(self) -> keras.Model:
        """Get the underlying Keras model.

        Returns:
            The compiled Keras model.
        """
        return self.model

    def get_model_summary(self) -> None:
        """Print a summary of the model architecture."""
        self.model.summary()

