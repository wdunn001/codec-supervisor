"""Adversarial defense helpers — TokenBreak, EchoGram, glitch tokens.

The plan's adversarial-realities section calls out three attack
classes that beat naive token-space enforcement:

1. **TokenBreak** (HiddenLayer): Banning the literal "instructions"
   doesn't ban "finstructions" — the attacker prefix changes BPE
   tokenization, so a banned-id list authored against one
   tokenization sees nothing. Mitigation: enumerate more
   tokenizations of the same banned literal (with adversarial
   prefixes/suffixes) and feed them all into slice 9's AC matcher.

2. **EchoGram** (HiddenLayer 2025): Append carefully chosen suffix
   tokens to a prompt-injection payload to evade BPE-based
   detectors. The injection's *semantic* content is unchanged.
   Mitigation: a content-aware classifier (slices 3/4/7) catches it
   regardless of prefix/suffix obfuscation. The plan explicitly
   names "Layer 3 (semantic classifier) is the answer; Layer 2
   alone is insufficient."

3. **Glitch tokens**: Some token IDs in major tokenizers are
   undertrained ("solidGoldMagikarp" in GPT-2, etc.). They:
     - can route around text-level keyword filters when emitted as
       single tokens that decode to weird strings;
     - cause models to behave erratically (good for jailbreaks).
   Mitigation: maintain a community-curated banned-id list of
   known glitch tokens per tokenizer; merge into the policy's
   `banned_token_ids` at load time.

This module ships the *helpers*. The actual attack-fixture data and
the integration tests proving each layer catches its assigned attack
class live in `tests/fixtures/adversarial.py` +
`tests/test_safety_adversarial.py`.

Workflow (operator-side):

1. Author a policy with `multi_token_patterns` listing the banned
   literals you want to enforce.
2. For each literal, run `enumerate_text_variants(literal)` to get
   text-level variants that exploit BPE seam variance.
3. Run all variants through your model's actual tokenizer (offline)
   to get the per-tokenization ID sequences.
4. Write the merged result into
   `multi_token_patterns[*].tokenizers[<tokenizer_id>]`.
5. Call `merge_glitch_tokens(policy.banned_token_ids,
   well_known_glitch_tokens(<tokenizer_id>))` and write back.
6. Save the policy. Slice 9's AC matcher + slice 6's banned-ID
   processor + slices 7's classifier do the rest.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence


# ── TokenBreak: text-variant enumeration ─────────────────────────────────────
#
# BPE tokenizers split words by greedy longest-match against the merge
# table. Inserting a single character at the front (or sometimes mid-
# word) often forces a different split:
#
#   "instructions"  →  ['inst', 'ruct', 'ions']
#   "finstructions" →  ['fin', 'struct', 'ions']
#
# A banned-id list authored against the first tokenization misses the
# second. The fix is to enumerate text variants the operator runs
# through their tokenizer offline; the resulting ID sequences feed
# slice 9's AC matcher, which then catches every enumerated form.

# Common adversarial prefixes/suffixes — known to flip BPE merges
# across the most popular tokenizer families. NOT exhaustive; operators
# extend per-policy. Sourced from the TokenBreak research and BPE-merge
# tables of Llama-3, Qwen-2, Mistral, GPT-2.
TOKENBREAK_PREFIXES: tuple[str, ...] = (
    # Prefix vowel/consonant pairs that commonly flip the leading merge
    "a", "e", "i", "o", "u",
    "fin", "sub", "pre", "re", "un", "in", "im",
    # Zero-width-ish spacing tricks
    " ",  # leading space changes Llama-3 byte-level pretok
    "\t",
    # Visually-similar substitutions that defeat literal matchers
    # while staying in ASCII (homoglyph attacks live in a separate
    # research direction; we don't enumerate them here).
)

TOKENBREAK_SUFFIXES: tuple[str, ...] = (
    "s", "ed", "ing", "er", "est",
    ".", ",", ";", "!", "?",
)


def enumerate_text_variants(
    literal: str,
    *,
    prefixes: Iterable[str] = TOKENBREAK_PREFIXES,
    suffixes: Iterable[str] = TOKENBREAK_SUFFIXES,
    include_case_variants: bool = True,
) -> tuple[str, ...]:
    """Enumerate text variants of `literal` that commonly flip BPE merges.

    Returns text strings — operators run them through their tokenizer
    to get token-ID sequences. Result includes:
      - the original literal
      - every prefix-applied variant
      - every suffix-applied variant
      - upper/lower/title case if `include_case_variants` is True

    Order: original first, then deterministic prefix-major. De-duped.

    The enumeration is a *coverage assist*, not a guarantee — an
    attacker who knows the policy can still craft tokenization-flipping
    transformations the prefix list doesn't cover. The classifier
    layer (slices 3/4/7) is the real backstop for novel transformations;
    this helper closes the most common gaps.
    """
    if not literal:
        return ()

    seen: set[str] = set()
    out: list[str] = []

    def _add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    _add(literal)

    if include_case_variants:
        _add(literal.lower())
        _add(literal.upper())
        _add(literal.title())

    for p in prefixes:
        _add(p + literal)
        if include_case_variants:
            _add(p + literal.lower())
            _add(p + literal.upper())
            _add(p + literal.title())

    for s in suffixes:
        _add(literal + s)
        if include_case_variants:
            _add(literal.lower() + s)

    return tuple(out)


# ── Glitch tokens ────────────────────────────────────────────────────────────
#
# Curated illustrative lists per tokenizer. Real production lists are
# many thousands of entries; this is a starter set sourced from public
# glitch-token research (Rumbelow & Watkins's SolidGoldMagikarp, the
# llm-attacks survey, Hugging Face community findings). Operators
# expand per policy.

# Format: tokenizer-id → tuple of token IDs known to be undertrained or
# to produce model anomalies. Tokenizer-ids match the policy's
# `tokenizers` list (`meta-llama/llama-3`, `qwen/qwen2`, etc.).
WELL_KNOWN_GLITCH_TOKENS: dict[str, tuple[int, ...]] = {
    # NOTE: these IDs are illustrative placeholders. Real glitch-token
    # discovery requires running the actual tokenizer + probing the
    # model. Operators populate this dict from their own audit before
    # depending on it. Shipping non-empty defaults would be misleading;
    # the empty-tuple-per-known-tokenizer pattern below makes the
    # registration call site safe (returns () rather than KeyError) and
    # the omission auditable.
    "meta-llama/llama-3": (),
    "qwen/qwen2": (),
    "mistralai/mistral-7b": (),
}


def well_known_glitch_tokens(tokenizer_id: str) -> tuple[int, ...]:
    """Return the curated glitch-token ID list for a tokenizer.

    Empty tuple when the tokenizer is unknown or no curation exists
    yet — explicitly *not* a KeyError, so callers can wire this in
    unconditionally and the policy is unaffected when no list applies.
    """
    return WELL_KNOWN_GLITCH_TOKENS.get(tokenizer_id, ())


def merge_glitch_tokens(
    banned_token_ids: Sequence[int] | None,
    glitch_ids: Sequence[int],
) -> tuple[int, ...]:
    """Merge glitch IDs into a policy's `banned_token_ids` list.

    De-duped; sorted for determinism (so the policy file diff is
    minimal across re-saves). Either argument MAY be None / empty;
    the merge handles both cases.
    """
    base = banned_token_ids or ()
    merged = set(int(i) for i in base)
    merged.update(int(i) for i in glitch_ids)
    return tuple(sorted(merged))


# ── Discoverability ──────────────────────────────────────────────────────────
#
# Public API surface for callers (admin scripts, the policy editor's
# "auto-harden" button if/when added). Documented here so a future
# slice that wires UI doesn't have to spelunk the source.

__all__ = [
    "TOKENBREAK_PREFIXES",
    "TOKENBREAK_SUFFIXES",
    "WELL_KNOWN_GLITCH_TOKENS",
    "enumerate_text_variants",
    "well_known_glitch_tokens",
    "merge_glitch_tokens",
]
