"""Main entry point for training and using small language models.

This script provides a command-line interface for training language models
and generating text from trained models.
"""

import argparse
import os
import sys
from pathlib import Path

from .data_preparation import DataPreparator
from .generator import TextGenerator
from .trainer import ModelTrainer


def train_model(
    dataset_path: str,
    output_dir: str = "./models",
    max_length: int = 300,
    batch_size: int = 32,
    epochs: int = 200,
    learning_rate: float = 1e-4,
    embedding_dim: int = 256,
    mlp_dim: int = 256,
    num_heads: int = 2,
    num_blocks: int = 1,
    prompt: str = "Abeni,",
    print_every: int = 10,
    random_seed: int = 812,
):
    """Train a small language model.

    Args:
        dataset_path: Path or URL to the JSON dataset file.
        output_dir: Directory to save the trained model.
        max_length: Maximum sequence length.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        embedding_dim: Embedding dimension.
        mlp_dim: MLP dimension.
        num_heads: Number of attention heads.
        num_blocks: Number of transformer blocks.
        prompt: Prompt for text generation during training.
        print_every: Print generated text every N epochs.
        random_seed: Random seed for reproducibility.
    """
    print("=" * 60)
    print("Small Language Model Training")
    print("=" * 60)

    # Create output directory.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data.
    print("\n[1/4] Preparing data...")
    preparator = DataPreparator(max_length=max_length, batch_size=batch_size)
    tokenizer, batches, actual_max_length, (shortest, longest) = (
        preparator.prepare_full_pipeline(dataset_path)
    )

    print(f"  - Dataset loaded and tokenized")
    print(f"  - Vocabulary size: {tokenizer.vocabulary_size}")
    print(f"  - Sequence lengths: shortest={shortest}, longest={longest}")
    print(f"  - Max length (after padding): {actual_max_length}")

    # Count batches.
    total_batches = sum(1 for _ in batches)
    print(f"  - Total batches: {total_batches}")

    # Initialize trainer.
    print("\n[2/4] Initializing model...")
    trainer = ModelTrainer(
        tokenizer=tokenizer,
        max_length=actual_max_length,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        random_seed=random_seed,
    )

    print("  - Model created successfully")
    trainer.get_model_summary()

    # Create callback for text generation during training.
    print(f"\n[3/4] Setting up training callbacks...")
    text_callback = trainer.create_text_generator_callback(
        prompt=prompt, max_tokens=10, print_every=print_every
    )

    # Train model.
    print(f"\n[4/4] Training model for {epochs} epochs...")
    print("  - This may take a while. Generated text will be printed periodically.")
    print("-" * 60)

    history = trainer.train(
        batches=batches,
        epochs=epochs,
        verbose=2,
        callbacks=[text_callback],
    )

    # Save model.
    model_path = os.path.join(output_dir, "model_weights.weights.h5")
    trainer.save_model(model_path)

    # Save tokenizer info (vocabulary).
    tokenizer_path = os.path.join(output_dir, "tokenizer_vocab.txt")
    with open(tokenizer_path, "w") as f:
        for token in tokenizer.vocabulary:
            f.write(f"{token}\n")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"  - Model saved to: {model_path}")
    print(f"  - Tokenizer vocabulary saved to: {tokenizer_path}")
    print("=" * 60)

    return trainer, tokenizer


def generate_text(
    model_path: str,
    tokenizer_vocab_path: str,
    prompt: str,
    num_tokens: int = 100,
    sampling_mode: str = "greedy",
    max_length: int = 300,
    learning_rate: float = 1e-4,
):
    """Generate text from a trained model.

    Args:
        model_path: Path to the saved model weights.
        tokenizer_vocab_path: Path to the tokenizer vocabulary file.
        prompt: Initial text prompt.
        num_tokens: Number of tokens to generate.
        sampling_mode: Sampling mode ('greedy' or 'random').
        max_length: Maximum sequence length (must match training).
        learning_rate: Learning rate (must match training).
    """
    print("=" * 60)
    print("Text Generation")
    print("=" * 60)

    # Load tokenizer vocabulary.
    print("\n[1/2] Loading tokenizer...")
    from .tokenizer import SimpleWordTokenizer

    with open(tokenizer_vocab_path, "r") as f:
        vocabulary = [line.strip() for line in f.readlines()]

    # Create tokenizer from vocabulary.
    tokenizer = SimpleWordTokenizer(corpus=[], vocabulary=vocabulary)
    print(f"  - Vocabulary size: {tokenizer.vocabulary_size}")

    # Initialize trainer to recreate model architecture.
    print("\n[2/2] Loading model...")
    trainer = ModelTrainer(
        tokenizer=tokenizer,
        max_length=max_length,
        learning_rate=learning_rate,
    )
    trainer.load_model(model_path)

    # Generate text.
    print(f"\nGenerating {num_tokens} tokens with prompt: '{prompt}'")
    print("-" * 60)

    generator = TextGenerator(trainer.get_model(), tokenizer)
    generated_text, probs = generator.generate(
        prompt=prompt, num_tokens=num_tokens, sampling_mode=sampling_mode
    )

    print(f"\nGenerated text:\n{generated_text}")
    print("=" * 60)

    return generated_text


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Train and use small language models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command.
    train_parser = subparsers.add_parser("train", help="Train a language model")
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path or URL to the JSON dataset file",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save the trained model (default: ./models)",
    )
    train_parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum sequence length (default: 300)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension (default: 256)",
    )
    train_parser.add_argument(
        "--mlp-dim",
        type=int,
        default=256,
        help="MLP dimension (default: 256)",
    )
    train_parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    train_parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of transformer blocks (default: 1)",
    )
    train_parser.add_argument(
        "--prompt",
        type=str,
        default="Abeni,",
        help="Prompt for text generation during training (default: 'Abeni,')",
    )
    train_parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print generated text every N epochs (default: 10)",
    )
    train_parser.add_argument(
        "--random-seed",
        type=int,
        default=812,
        help="Random seed for reproducibility (default: 812)",
    )

    # Generate command.
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text from a trained model"
    )
    gen_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model weights (.weights.h5 file)",
    )
    gen_parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer vocabulary file",
    )
    gen_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial text prompt",
    )
    gen_parser.add_argument(
        "--num-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate (default: 100)",
    )
    gen_parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["greedy", "random"],
        default="greedy",
        help="Sampling mode: 'greedy' or 'random' (default: 'greedy')",
    )
    gen_parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum sequence length (must match training, default: 300)",
    )
    gen_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (must match training, default: 1e-4)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            embedding_dim=args.embedding_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            prompt=args.prompt,
            print_every=args.print_every,
            random_seed=args.random_seed,
        )
    elif args.command == "generate":
        generate_text(
            model_path=args.model,
            tokenizer_vocab_path=args.tokenizer,
            prompt=args.prompt,
            num_tokens=args.num_tokens,
            sampling_mode=args.sampling_mode,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

