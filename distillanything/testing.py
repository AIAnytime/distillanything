"""Tiny offline fixtures: a character-level tokenizer and small random Llama models.

Used by the test suite and `distill smoke` so the full pipeline can be exercised
without downloading anything — important for laptop development and CI.
"""

from __future__ import annotations

import string


def tiny_tokenizer():
    """A fast character-level tokenizer built entirely in memory."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = list(string.printable)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    return fast


def tiny_model(vocab_size: int, hidden: int = 64, layers: int = 2, heads: int = 4, seed: int = 0):
    """A random-weight Llama-architecture model small enough for CPU training."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(config)


def tiny_student_and_teacher(vocab_size: int):
    """A small student and a larger teacher sharing one vocabulary."""
    student = tiny_model(vocab_size, hidden=64, layers=2, seed=1)
    teacher = tiny_model(vocab_size, hidden=128, layers=4, seed=2)
    return student, teacher


def tiny_records(n: int = 64) -> list[dict]:
    """Deterministic toy instruction data (string reversal — learnable by a char model)."""
    words = ["apple", "banana", "cherry", "orange", "grape", "mango", "lemon", "peach"]
    records = []
    for i in range(n):
        word = words[i % len(words)]
        records.append({"prompt": f"reverse {word}{i}", "response": (word + str(i))[::-1]})
    return records
