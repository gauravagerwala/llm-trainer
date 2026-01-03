"""Tests for the DataPreparator class."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from slm_trainer.data_preparation import DataPreparator
from slm_trainer.tokenizer import SimpleWordTokenizer


class TestDataPreparator(unittest.TestCase):
    """Test cases for DataPreparator."""

    def setUp(self):
        """Set up test fixtures."""
        self.corpus = [
            "Hello world",
            "How are you",
            "Hello there",
            "This is a longer sentence with more words",
        ]
        self.tokenizer = SimpleWordTokenizer(self.corpus)

    def test_initialization(self):
        """Test DataPreparator initialization."""
        preparator = DataPreparator(
            tokenizer=self.tokenizer, max_length=100, batch_size=32
        )

        self.assertEqual(preparator.tokenizer, self.tokenizer)
        self.assertEqual(preparator.max_length, 100)
        self.assertEqual(preparator.batch_size, 32)

    def test_initialization_without_tokenizer(self):
        """Test DataPreparator initialization without tokenizer."""
        preparator = DataPreparator(max_length=100, batch_size=32)

        self.assertIsNone(preparator.tokenizer)

    def test_tokenize_dataset(self):
        """Test dataset tokenization."""
        preparator = DataPreparator(max_length=100, batch_size=32)
        tokenizer, encoded_tokens = preparator.tokenize_dataset(self.corpus)

        # Should return a tokenizer.
        self.assertIsInstance(tokenizer, SimpleWordTokenizer)

        # Should return encoded tokens.
        self.assertIsInstance(encoded_tokens, list)
        self.assertEqual(len(encoded_tokens), len(self.corpus))

        # Each encoded token should be a list of integers.
        for token_ids in encoded_tokens:
            self.assertIsInstance(token_ids, list)
            self.assertTrue(all(isinstance(id, int) for id in token_ids))

    def test_tokenize_dataset_with_existing_tokenizer(self):
        """Test tokenization with existing tokenizer."""
        preparator = DataPreparator(
            tokenizer=self.tokenizer, max_length=100, batch_size=32
        )
        tokenizer, encoded_tokens = preparator.tokenize_dataset(self.corpus)

        # Should use the existing tokenizer.
        self.assertEqual(tokenizer, self.tokenizer)

    def test_compute_length_statistics(self):
        """Test length statistics computation."""
        preparator = DataPreparator(
            tokenizer=self.tokenizer, max_length=100, batch_size=32
        )
        _, encoded_tokens = preparator.tokenize_dataset(self.corpus)

        shortest, longest = preparator.compute_length_statistics(encoded_tokens)

        lengths = [len(seq) for seq in encoded_tokens]
        self.assertEqual(shortest, min(lengths))
        self.assertEqual(longest, max(lengths))

    def test_pad_sequences(self):
        """Test sequence padding."""
        preparator = DataPreparator(
            tokenizer=self.tokenizer, max_length=10, batch_size=32
        )
        _, encoded_tokens = preparator.tokenize_dataset(self.corpus)

        padded_sequences, max_len = preparator.pad_sequences(encoded_tokens)

        # Should return a tensor.
        self.assertIsInstance(padded_sequences, tf.Tensor)

        # All sequences should have the same length.
        self.assertEqual(padded_sequences.shape[1], 10)

        # Check that sequences are padded correctly.
        for i in range(padded_sequences.shape[0]):
            seq = padded_sequences[i].numpy()
            # Check that padding tokens are at the end.
            if len(encoded_tokens[i]) < 10:
                # Find where padding starts.
                pad_start = len(encoded_tokens[i])
                # All tokens after pad_start should be pad_token_id.
                self.assertTrue(
                    all(
                        token == self.tokenizer.pad_token_id
                        for token in seq[pad_start:]
                    )
                )

    def test_prepare_input_target(self):
        """Test input and target sequence preparation."""
        preparator = DataPreparator(
            tokenizer=self.tokenizer, max_length=10, batch_size=32
        )
        _, encoded_tokens = preparator.tokenize_dataset(self.corpus)
        padded_sequences, _ = preparator.pad_sequences(encoded_tokens)

        input_sequences, target_sequences = preparator.prepare_input_target(
            padded_sequences
        )

        # Should return tensors.
        self.assertIsInstance(input_sequences, tf.Tensor)
        self.assertIsInstance(target_sequences, tf.Tensor)

        # Input and target should have the same shape.
        self.assertEqual(input_sequences.shape, target_sequences.shape)

        # Input should be target shifted by one position.
        # Check first example.
        input_seq = input_sequences[0].numpy()
        target_seq = target_sequences[0].numpy()

        # Target should be input shifted by one.
        np.testing.assert_array_equal(input_seq[1:], target_seq[:-1])

    def test_create_batches(self):
        """Test batch creation."""
        preparator = DataPreparator(
            tokenizer=self.tokenizer, max_length=10, batch_size=2
        )
        _, encoded_tokens = preparator.tokenize_dataset(self.corpus)
        padded_sequences, _ = preparator.pad_sequences(encoded_tokens)
        input_sequences, target_sequences = preparator.prepare_input_target(
            padded_sequences
        )

        batches = preparator.create_batches(
            input_sequences, target_sequences, shuffle=False
        )

        # Should return a dataset.
        self.assertIsInstance(batches, tf.data.Dataset)

        # Count batches.
        batch_count = 0
        for batch in batches:
            batch_count += 1
            inputs, targets = batch
            # Each batch should have at most batch_size examples.
            self.assertLessEqual(inputs.shape[0], preparator.batch_size)

        # Should have at least one batch.
        self.assertGreater(batch_count, 0)

    @patch("slm_trainer.data_preparation.pd.read_json")
    def test_load_dataset(self, mock_read_json):
        """Test dataset loading."""
        # Mock pandas read_json.
        mock_df = MagicMock()
        mock_df.__getitem__.return_value.values.tolist.return_value = self.corpus
        mock_read_json.return_value = mock_df

        preparator = DataPreparator(max_length=100, batch_size=32)
        dataset = preparator.load_dataset("test.json", text_column="description")

        # Should return a list of strings.
        self.assertIsInstance(dataset, list)
        self.assertEqual(len(dataset), len(self.corpus))

        # Verify pandas was called.
        mock_read_json.assert_called_once_with("test.json")


if __name__ == "__main__":
    unittest.main()

