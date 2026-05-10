"""Offline training utilities for the embedding-space classifier.

The runtime classifier (`safety_classifiers/embedding_space.py`) consumes a
trained classification head over engine hidden states. This package
ships the *bootstrap* — the script that uses the existing text-space
classifiers (Llama Guard, ShieldGemma) to label a corpus of prompts +
completions, producing JSONL that a downstream trainer turns into head
weights.

Slice 8 ships the bootstrap. The trainer itself (PyTorch script that
fits a BERT-tier head against the labeled corpus) is operator-chosen
and not in this package — the corpus shape is a stable contract that
multiple trainers can consume.
"""
