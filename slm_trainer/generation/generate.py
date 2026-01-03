"""Text generation utilities for a trained Keras model.

This module provides functions for autoregressive text generation, supporting
both greedy decoding and random sampling methods.
"""

import random
from typing import Any, Literal

import jax
import jax.numpy as jnp
import keras
from keras import ops


def sampling(probs: jax.Array, key: jax.Array) -> int:
    """Sample a token index from the predicted next token probability.

    Args:
        probs: The probability distribution of predicted next token.
        key: The JAX random key.

    Returns:
        The index of the sampled token.
    """
    token_index = jax.random.choice(key, jnp.arange(probs.shape[0]), p=probs)
    return int(token_index)


def greedy_decoding(probs: jax.Array) -> int:
    """Select the token index from the predicted next token probability.

    Args:
        probs: The probability distribution of predicted next token.

    Returns:
        The index of the token with the highest probability.
    """
    return int(jnp.argmax(probs))


def generate_text(
    start_prompt: str,
    n_tokens: int,
    model: keras.Model,
    tokenizer: Any,
    pad_token_id: int = 0,
    sampling_mode: Literal["random", "greedy"] = "random",
) -> tuple[str, list[jax.Array]]:
    """Generate text based on a starting prompt using a trained model.

    Args:
        start_prompt: The initial prompt to start the generation.
        n_tokens: The number of tokens to generate after the prompt.
        model: The trained model to use for text generation.
        tokenizer: The tokenizer to encode and decode text.
        pad_token_id: The token ID used for padding.
        sampling_mode: Whether to use random or greedy sampling. Supported options
            are 'random' and 'greedy'.

    Returns:
        The generated text after the prompt.
    """

    if sampling_mode not in ["random", "greedy"]:
        raise ValueError(
            f"Sampling mode {sampling_mode} is not supported. Supported options are"
            " 'random' and 'greedy'."
        )

    # Introduce randomness by re-intializing JAX RNG with a different seed on
    # each call. While this harms reproducability, it avoids having to pass a JAX
    # key on every call, which would likely be confusing to learners.
    main_key = jax.random.PRNGKey(random.randint(0, 1000000))

    max_length = model.layers[0].output.shape[1]

    # Tokenize the starting prompt.
    start_tokens = tokenizer.encode(start_prompt)

    # Generate tokens.
    tokens_generated = start_tokens + []
    probs = []
    for _ in range(n_tokens):
        pad_len = max_length - len(start_tokens)
        sample_index = len(start_tokens) - 1
        if pad_len < 0:
            # Truncate the input sequence to fit the max context length.
            x = start_tokens[:max_length]
            sample_index = max_length - 1
        elif pad_len > 0:
            x = start_tokens + [pad_token_id] * pad_len  # Pad the input sequence.
        else:
            x = start_tokens

        x = jnp.array([x])

        # Get predictions from the model.
        y = model.predict(x, verbose="0")

        # Apply softmax to convert logits to probabilities.
        probabilities = ops.softmax(y, axis=-1)

        # Convert to JAX array if it's a TensorFlow tensor
        probs_array = probabilities[0][sample_index]
        if not isinstance(probs_array, jax.Array):
            probs_array = jnp.array(probs_array)

        probs.append(probs_array)

        # Use greedy decoding or sampling based on sampling_mode.
        if sampling_mode == "greedy":
            sample_token = greedy_decoding(probs_array)
        else:
            key, main_key = jax.random.split(main_key)
            sample_token = sampling(probs_array, key)

        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)

    # Convert tokens back to text.
    generated_text = tokenizer.decode(tokens_generated)
    generated_text = generated_text.replace(tokenizer.decode([pad_token_id]), "")

    return generated_text, probs

