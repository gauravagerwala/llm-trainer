"""Data preparation utilities for training language models.

This module provides functions to load, tokenize, pad, and batch datasets
for training transformer language models.
"""

from typing import Optional, Tuple

import keras
import pandas as pd
import tensorflow as tf

from .tokenizer import SimpleWordTokenizer


class DataPreparator:
    """Prepares datasets for training language models.

    This class handles loading datasets, tokenization, padding, and batching
    for transformer model training.

    Attributes:
        tokenizer: The tokenizer used for encoding text.
        max_length: Maximum sequence length after padding/truncation.
        batch_size: Number of examples per batch.
    """

    def __init__(
        self,
        tokenizer: Optional[SimpleWordTokenizer] = None,
        max_length: int = 300,
        batch_size: int = 32,
    ):
        """Initialize the DataPreparator.

        Args:
            tokenizer: Optional tokenizer. If None, one will be created from
                the dataset.
            max_length: Maximum sequence length for padding/truncation.
            batch_size: Number of examples per batch.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def load_dataset(
        self, dataset_path: str, text_column: str = "description"
    ) -> list[str]:
        """Load a dataset from a JSON file or URL.

        Args:
            dataset_path: Path or URL to the JSON dataset file.
            text_column: Name of the column containing text data.

        Returns:
            List of text strings from the dataset.
        """
        df = pd.read_json(dataset_path)
        dataset = df[text_column].values.tolist()
        return dataset

    def tokenize_dataset(self, dataset: list[str]) -> Tuple[SimpleWordTokenizer, list[list[int]]]:
        """Tokenize a dataset and create a tokenizer if needed.

        Args:
            dataset: List of text strings to tokenize.

        Returns:
            Tuple of (tokenizer, encoded_tokens) where encoded_tokens is a
            list of lists of token IDs.
        """
        # Create tokenizer if not provided.
        if self.tokenizer is None:
            self.tokenizer = SimpleWordTokenizer(dataset)

        # Translate all tokens to their corresponding IDs.
        encoded_tokens = []
        for text in dataset:
            token_ids = self.tokenizer.encode(text)
            encoded_tokens.append(token_ids)

        return self.tokenizer, encoded_tokens

    def compute_length_statistics(
        self, encoded_tokens: list[list[int]]
    ) -> Tuple[int, int]:
        """Compute minimum and maximum sequence lengths in the dataset.

        Args:
            encoded_tokens: List of token ID sequences.

        Returns:
            Tuple of (shortest_length, longest_length).
        """
        if not encoded_tokens:
            return 0, 0

        lengths = [len(seq) for seq in encoded_tokens]
        return min(lengths), max(lengths)

    def pad_sequences(
        self, encoded_tokens: list[list[int]]
    ) -> Tuple[tf.Tensor, int]:
        """Pad and truncate sequences to a fixed length.

        Args:
            encoded_tokens: List of token ID sequences.

        Returns:
            Tuple of (padded_sequences, actual_max_length) where
            actual_max_length accounts for the shift in input/target sequences.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before padding.")

        padded_sequences = keras.preprocessing.sequence.pad_sequences(
            encoded_tokens,
            maxlen=self.max_length,
            padding="post",
            truncating="post",
            value=self.tokenizer.pad_token_id,
        )

        # Convert to TensorFlow tensor.
        padded_sequences = tf.constant(padded_sequences, dtype=tf.int32)

        return padded_sequences, self.max_length

    def prepare_input_target(
        self, padded_sequences: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Prepare input and target sequences for training.

        For language modeling, the target is the input shifted by one token.
        Input: all tokens except the last.
        Target: all tokens except the first.

        Args:
            padded_sequences: Padded sequences of shape (num_examples, max_length).

        Returns:
            Tuple of (input_sequences, target_sequences) where both have shape
            (num_examples, max_length - 1).
        """
        # For each example, extract all tokens except the last one.
        input_sequences = padded_sequences[:, :-1]
        # For each example, extract all tokens except the first one.
        target_sequences = padded_sequences[:, 1:]

        return input_sequences, target_sequences

    def create_batches(
        self,
        input_sequences: tf.Tensor,
        target_sequences: tf.Tensor,
        shuffle: bool = True,
        buffer_size: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Create batched dataset for training.

        Args:
            input_sequences: Input sequences tensor.
            target_sequences: Target sequences tensor.
            shuffle: Whether to shuffle the dataset before batching.
            buffer_size: Buffer size for shuffling. If None, uses the full
                dataset size.

        Returns:
            Batched TensorFlow dataset.
        """
        # Create TensorFlow dataset.
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )

        # Randomly shuffle the dataset if requested.
        if shuffle:
            if buffer_size is None:
                buffer_size = len(input_sequences)
            tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size)

        # Create batches.
        batches = tf_dataset.batch(self.batch_size)

        return batches

    def prepare_full_pipeline(
        self,
        dataset_path: str,
        text_column: str = "description",
        shuffle: bool = True,
    ) -> Tuple[
        SimpleWordTokenizer,
        tf.data.Dataset,
        int,
        Tuple[int, int],
    ]:
        """Run the complete data preparation pipeline.

        This method loads, tokenizes, pads, and batches the dataset in one call.

        Args:
            dataset_path: Path or URL to the JSON dataset file.
            text_column: Name of the column containing text data.
            shuffle: Whether to shuffle the dataset before batching.

        Returns:
            Tuple of (tokenizer, batches, max_length, (shortest, longest)) where:
            - tokenizer: The tokenizer used for encoding
            - batches: Batched TensorFlow dataset ready for training
            - max_length: Maximum sequence length (after input/target shift)
            - (shortest, longest): Original sequence length statistics
        """
        # Load dataset.
        dataset = self.load_dataset(dataset_path, text_column)

        # Tokenize dataset.
        tokenizer, encoded_tokens = self.tokenize_dataset(dataset)

        # Compute statistics.
        shortest, longest = self.compute_length_statistics(encoded_tokens)

        # Pad sequences.
        padded_sequences, _ = self.pad_sequences(encoded_tokens)

        # Prepare input and target.
        input_sequences, target_sequences = self.prepare_input_target(
            padded_sequences
        )

        # Update max_length to account for the shift.
        actual_max_length = input_sequences.shape[1]

        # Create batches.
        batches = self.create_batches(
            input_sequences, target_sequences, shuffle=shuffle
        )

        return tokenizer, batches, actual_max_length, (shortest, longest)

