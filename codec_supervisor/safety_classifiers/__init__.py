"""Concrete server-side `SafetyClassifier` implementations.

Slice 7 ships two implementations and registers them at module load:

  - LlamaGuard31B       — tier 1; 14-category MLCommons taxonomy
  - ShieldGemma2B       — tier 2; 4-category Google taxonomy

Importing this package side-effects-registers both. Hosts that prefer
explicit registration can import the classes directly from their
modules and skip this `__init__`.

Both implementations honor the same generator-DI pattern as the
browser side (`@codecai/web-safety`): construction takes an optional
generator callable so tests can stub model output without loading
real weights.
"""

from .embedding_space import EmbeddingSpaceClassifier, register_embedding_space
from .llamaguard_3_1b import LlamaGuard31B, register_llamaguard_3_1b
from .shieldgemma_2b import ShieldGemma2B, register_shieldgemma_2b

__all__ = [
    "EmbeddingSpaceClassifier",
    "LlamaGuard31B",
    "ShieldGemma2B",
    "register_embedding_space",
    "register_llamaguard_3_1b",
    "register_shieldgemma_2b",
    "register_default_classifiers",
]


def register_default_classifiers() -> None:
    """Register all shipped classifiers. Idempotent — safe to call
    multiple times (re-registration silently no-ops, matching the
    browser side).

    The embedding-space classifier registers in capability-fail mode by
    default (no scorer / no head weights wired). The registry's
    `resolve_classifier()` then transparently falls back to a text
    classifier. Operators who've wired up an engine hidden-state hook
    re-register with `scorer=` to enable Layer 3 enforcement.
    """
    register_embedding_space()
    register_llamaguard_3_1b()
    register_shieldgemma_2b()
