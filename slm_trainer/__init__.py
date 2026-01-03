"""Small Language Model Trainer Package.

This package provides utilities for training and using small language models
based on transformer architectures.
"""

__version__ = "0.1.0"

from .tokenizer import SimpleWordTokenizer
from .data_preparation import DataPreparator
from .trainer import ModelTrainer
from .generator import TextGenerator

__all__ = [
    "SimpleWordTokenizer",
    "DataPreparator",
    "ModelTrainer",
    "TextGenerator",
]

