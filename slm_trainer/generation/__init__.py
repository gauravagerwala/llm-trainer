"""Text generation and inference utilities for language models."""

from .generate import generate_text
from .generate import greedy_decoding
from .generate import sampling as random_decoding

__all__ = [
    "generate_text",
    "greedy_decoding",
    "random_decoding",
]

