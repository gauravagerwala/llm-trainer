"""Tests for the ModelTrainer class."""

import os
import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from slm_trainer.tokenizer import SimpleWordTokenizer
from slm_trainer.trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.corpus = ["Hello world", "How are you", "Hello there"]
        self.tokenizer = SimpleWordTokenizer(self.corpus)
        self.max_length = 10

    @patch("slm_trainer.trainer.training")
    def test_initialization(self, mock_training):
        """Test ModelTrainer initialization."""
        # Mock the create_model function.
        mock_model = MagicMock()
        mock_training.create_model.return_value = mock_model

        trainer = ModelTrainer(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            learning_rate=1e-4,
        )

        # Check that model was created.
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.tokenizer, self.tokenizer)
        self.assertEqual(trainer.max_length, self.max_length)
        self.assertEqual(trainer.vocabulary_size, self.tokenizer.vocabulary_size)

        # Verify create_model was called with correct parameters.
        mock_training.create_model.assert_called_once()

    @patch("slm_trainer.trainer.training")
    def test_train(self, mock_training):
        """Test model training."""
        # Mock the create_model function.
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_model.fit.return_value = mock_history
        mock_training.create_model.return_value = mock_model

        trainer = ModelTrainer(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Create dummy batches.
        dummy_data = tf.constant([[1, 2, 3], [4, 5, 6]])
        batches = tf.data.Dataset.from_tensor_slices(
            (dummy_data, dummy_data)
        ).batch(2)

        # Train the model.
        history = trainer.train(batches, epochs=2, verbose=0)

        # Verify fit was called.
        mock_model.fit.assert_called_once()
        self.assertEqual(history, mock_history)

    @patch("slm_trainer.trainer.training")
    def test_create_text_generator_callback(self, mock_training):
        """Test text generator callback creation."""
        # Mock the create_model function.
        mock_model = MagicMock()
        mock_training.create_model.return_value = mock_model
        mock_callback = MagicMock()
        mock_training.TextGenerator.return_value = mock_callback

        trainer = ModelTrainer(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        callback = trainer.create_text_generator_callback(
            prompt="Hello", max_tokens=5, print_every=1
        )

        # Verify TextGenerator was called with correct parameters.
        mock_training.TextGenerator.assert_called_once()
        self.assertEqual(callback, mock_callback)

    @patch("slm_trainer.trainer.training")
    def test_save_and_load_model(self, mock_training):
        """Test model saving and loading."""
        # Mock the create_model function.
        mock_model = MagicMock()
        mock_training.create_model.return_value = mock_model

        trainer = ModelTrainer(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Test save.
        trainer.save_model("test_model.h5")
        mock_model.save_weights.assert_called_once_with("test_model.h5")

        # Test load.
        trainer.load_model("test_model.h5")
        mock_model.load_weights.assert_called_once_with("test_model.h5")

    @patch("slm_trainer.trainer.training")
    def test_get_model(self, mock_training):
        """Test getting the underlying model."""
        # Mock the create_model function.
        mock_model = MagicMock()
        mock_training.create_model.return_value = mock_model

        trainer = ModelTrainer(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        model = trainer.get_model()

        self.assertEqual(model, mock_model)

    @patch("slm_trainer.trainer.training")
    def test_get_model_summary(self, mock_training):
        """Test getting model summary."""
        # Mock the create_model function.
        mock_model = MagicMock()
        mock_training.create_model.return_value = mock_model

        trainer = ModelTrainer(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        trainer.get_model_summary()

        # Verify summary was called.
        mock_model.summary.assert_called_once()


if __name__ == "__main__":
    unittest.main()

