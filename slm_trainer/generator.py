"""Text generation utilities for trained language models.

This module provides functions for generating text from trained language models
using various sampling strategies.
"""

from typing import Any, Literal, Optional

import keras

from .tokenizer import SimpleWordTokenizer
from .generation import generate_text as _generate_text


class TextGenerator:
    """Text generator for language models.

    This class provides a high-level interface for generating text from
    trained language models.

    Attributes:
        model: The trained Keras model.
        tokenizer: The tokenizer used for encoding/decoding text.
    """

    def __init__(self, model: keras.Model, tokenizer: SimpleWordTokenizer):
        """Initialize the TextGenerator.

        Args:
            model: The trained Keras model.
            tokenizer: The tokenizer used for encoding/decoding text.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        num_tokens: int = 100,
        sampling_mode: Literal["random", "greedy"] = "greedy",
    ) -> tuple[str, list]:
        """Generate text from a prompt.

        Args:
            prompt: Initial text prompt.
            num_tokens: Number of tokens to generate.
            sampling_mode: Sampling strategy ('random' or 'greedy').
                - 'greedy': Always selects the most probable next token.
                - 'random': Samples from the probability distribution.

        Returns:
            Tuple of (generated_text, probabilities) where:
            - generated_text: The complete generated text (prompt + generated tokens)
            - probabilities: List of probability distributions for each generated token.
        """
        generated_text, probs = _generate_text(
            start_prompt=prompt,
            n_tokens=num_tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
            sampling_mode=sampling_mode,
        )

        return generated_text, probs

    def generate_greedy(
        self, prompt: str, num_tokens: int = 100
    ) -> tuple[str, list]:
        """Generate text using greedy decoding (always picks most likely token).

        Args:
            prompt: Initial text prompt.
            num_tokens: Number of tokens to generate.

        Returns:
            Tuple of (generated_text, probabilities).
        """
        return self.generate(prompt, num_tokens, sampling_mode="greedy")

    def generate_random(
        self, prompt: str, num_tokens: int = 100
    ) -> tuple[str, list]:
        """Generate text using random sampling from probability distribution.

        Args:
            prompt: Initial text prompt.
            num_tokens: Number of tokens to generate.

        Returns:
            Tuple of (generated_text, probabilities).
        """
        return self.generate(prompt, num_tokens, sampling_mode="random")

