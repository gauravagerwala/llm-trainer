"""Tests for the TextGenerator class."""

import unittest
from unittest.mock import MagicMock, patch

from slm_trainer.generator import TextGenerator
from slm_trainer.tokenizer import SimpleWordTokenizer


class TestTextGenerator(unittest.TestCase):
    """Test cases for TextGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.corpus = ["Hello world", "How are you", "Hello there"]
        self.tokenizer = SimpleWordTokenizer(self.corpus)
        self.mock_model = MagicMock()

    @patch("slm_trainer.generator.generation")
    def test_initialization(self, mock_generation):
        """Test TextGenerator initialization."""
        generator = TextGenerator(self.mock_model, self.tokenizer)

        self.assertEqual(generator.model, self.mock_model)
        self.assertEqual(generator.tokenizer, self.tokenizer)

    @patch("slm_trainer.generator.generation")
    def test_generate(self, mock_generation):
        """Test text generation."""
        mock_generation.generate_text.return_value = (
            "Hello world generated text",
            [MagicMock(), MagicMock()],
        )

        generator = TextGenerator(self.mock_model, self.tokenizer)
        generated_text, probs = generator.generate(
            prompt="Hello", num_tokens=10, sampling_mode="greedy"
        )

        # Verify generate_text was called.
        mock_generation.generate_text.assert_called_once()
        call_args = mock_generation.generate_text.call_args

        # Check arguments.
        self.assertEqual(call_args.kwargs["start_prompt"], "Hello")
        self.assertEqual(call_args.kwargs["n_tokens"], 10)
        self.assertEqual(call_args.kwargs["sampling_mode"], "greedy")
        self.assertEqual(call_args.kwargs["model"], self.mock_model)
        self.assertEqual(call_args.kwargs["tokenizer"], self.tokenizer)

        # Check return values.
        self.assertEqual(generated_text, "Hello world generated text")
        self.assertIsInstance(probs, list)

    @patch("slm_trainer.generator.generation")
    def test_generate_greedy(self, mock_generation):
        """Test greedy text generation."""
        mock_generation.generate_text.return_value = (
            "Hello world",
            [MagicMock()],
        )

        generator = TextGenerator(self.mock_model, self.tokenizer)
        generated_text, probs = generator.generate_greedy(
            prompt="Hello", num_tokens=5
        )

        # Verify generate_text was called with greedy mode.
        call_args = mock_generation.generate_text.call_args
        self.assertEqual(call_args.kwargs["sampling_mode"], "greedy")

    @patch("slm_trainer.generator.generation")
    def test_generate_random(self, mock_generation):
        """Test random text generation."""
        mock_generation.generate_text.return_value = (
            "Hello world",
            [MagicMock()],
        )

        generator = TextGenerator(self.mock_model, self.tokenizer)
        generated_text, probs = generator.generate_random(
            prompt="Hello", num_tokens=5
        )

        # Verify generate_text was called with random mode.
        call_args = mock_generation.generate_text.call_args
        self.assertEqual(call_args.kwargs["sampling_mode"], "random")


if __name__ == "__main__":
    unittest.main()

