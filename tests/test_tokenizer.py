"""Tests for the SimpleWordTokenizer class."""

import unittest

from slm_trainer.tokenizer import SimpleWordTokenizer


class TestSimpleWordTokenizer(unittest.TestCase):
    """Test cases for SimpleWordTokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.corpus = [
            "Hello world",
            "How are you",
            "Hello there",
        ]

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization from corpus."""
        tokenizer = SimpleWordTokenizer(self.corpus)

        # Check that special tokens are present.
        self.assertIn(SimpleWordTokenizer.PAD_TOKEN, tokenizer.vocabulary)
        self.assertIn(SimpleWordTokenizer.UNKNOWN_TOKEN, tokenizer.vocabulary)

        # Check that PAD is first and UNK is last.
        self.assertEqual(tokenizer.vocabulary[0], SimpleWordTokenizer.PAD_TOKEN)
        self.assertEqual(
            tokenizer.vocabulary[-1], SimpleWordTokenizer.UNKNOWN_TOKEN
        )

        # Check vocabulary size.
        unique_tokens = set()
        for text in self.corpus:
            unique_tokens.update(text.split())
        expected_vocab_size = len(unique_tokens) + 2  # +2 for PAD and UNK
        self.assertEqual(tokenizer.vocabulary_size, expected_vocab_size)

    def test_tokenizer_with_vocabulary(self):
        """Test tokenizer initialization with pre-defined vocabulary."""
        vocabulary = ["<PAD>", "Hello", "world", "How", "<UNK>"]
        tokenizer = SimpleWordTokenizer(corpus=[], vocabulary=vocabulary)

        self.assertEqual(tokenizer.vocabulary, vocabulary)
        self.assertEqual(tokenizer.vocabulary_size, len(vocabulary))

    def test_space_tokenize(self):
        """Test space tokenization."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        tokens = tokenizer.space_tokenize("Hello   world")
        self.assertEqual(tokens, ["Hello", "world"])

    def test_encode(self):
        """Test encoding text to token IDs."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        token_ids = tokenizer.encode("Hello world")

        # Should return a list of integers.
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(id, int) for id in token_ids))

        # Should have 2 tokens.
        self.assertEqual(len(token_ids), 2)

    def test_encode_unknown_token(self):
        """Test encoding with unknown tokens."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        token_ids = tokenizer.encode("Unknown word")

        # Unknown tokens should map to UNK token ID.
        unk_id = tokenizer.unknown_token_id
        self.assertIn(unk_id, token_ids)

    def test_decode(self):
        """Test decoding token IDs to text."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        token_ids = tokenizer.encode("Hello world")
        decoded = tokenizer.decode(token_ids)

        # Should return a string.
        self.assertIsInstance(decoded, str)
        # Should contain the original tokens.
        self.assertIn("Hello", decoded)
        self.assertIn("world", decoded)

    def test_decode_single_int(self):
        """Test decoding a single integer."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        token_ids = tokenizer.encode("Hello")
        decoded = tokenizer.decode(token_ids[0])

        self.assertIsInstance(decoded, str)

    def test_encode_decode_roundtrip(self):
        """Test that encode and decode are inverse operations."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        original_text = "Hello world"

        token_ids = tokenizer.encode(original_text)
        decoded_text = tokenizer.decode(token_ids)

        # Decoded text should contain the original tokens.
        # Note: exact match may not work due to spacing, but tokens should match.
        original_tokens = original_text.split()
        decoded_tokens = decoded_text.split()
        self.assertEqual(original_tokens, decoded_tokens)

    def test_pad_token_id(self):
        """Test that pad token ID is correctly set."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        self.assertEqual(tokenizer.pad_token_id, 0)  # PAD is first
        self.assertEqual(
            tokenizer.vocabulary[tokenizer.pad_token_id],
            SimpleWordTokenizer.PAD_TOKEN,
        )

    def test_unknown_token_id(self):
        """Test that unknown token ID is correctly set."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        # UNK should be the last token.
        self.assertEqual(
            tokenizer.unknown_token_id, tokenizer.vocabulary_size - 1
        )
        self.assertEqual(
            tokenizer.vocabulary[tokenizer.unknown_token_id],
            SimpleWordTokenizer.UNKNOWN_TOKEN,
        )

    def test_build_vocabulary(self):
        """Test vocabulary building."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        tokens = ["hello", "world", "hello", "test"]
        vocabulary = tokenizer.build_vocabulary(tokens)

        # Should be sorted and unique.
        self.assertEqual(vocabulary, sorted(set(tokens)))

    def test_join_text(self):
        """Test joining tokens into text."""
        tokenizer = SimpleWordTokenizer(self.corpus)
        tokens = ["Hello", "world"]
        joined = tokenizer.join_text(tokens)

        self.assertEqual(joined, "Hello world")


if __name__ == "__main__":
    unittest.main()

