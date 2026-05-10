"""High-level wrapper over an internal policy for engine integration.

`SafetyEnforcement.from_policy_id(policies_dir, policy_id)` loads the
operator's full-detail policy from disk (the format slice 5 persists)
and exposes everything an engine integration needs in one place:

  - `banned_token_ids: tuple[int, ...]`   — raw IDs to mask in logits
  - `grammar_specs: tuple[dict, ...]`     — passthrough for engine guided-
                                            decoding (vLLM, llama.cpp)
  - `multi_token_patterns: tuple[dict, ...]` — passthrough for slice 9
  - `categories: dict[name → action]`     — what to do when a category fires
  - `tokenizer_ids: tuple[str, ...]`      — must match engine's loaded model

`make_logits_processor()` returns a ready-to-go vLLM-compatible callable.

This wrapper is the single point of contact between the static config
(slice 5) and runtime enforcement (this slice + slices 7-9). The
proxy / engine integration plumbing (passing the processor to vLLM
via `extra_body.logits_processors` or registering it as a plugin) is
deliberately not in this module — it depends on which engine fork the
operator is running and is parameterized by the deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .safety import InternalPolicy, load_policy
from .safety_logits import BannedTokenLogitsProcessor

LogitsProcessor = Callable[[Sequence[int], object], object]


class TokenizerMismatch(ValueError):
    """Raised when an operator tries to enforce a policy bound to a
    tokenizer the loaded engine isn't using.

    Banning token ID 4218 against the wrong vocab is silently wrong —
    that ID resolves to a different sub-word — so we reject the load
    rather than enforce a misaligned policy. Operators authoring a
    cross-tokenizer policy publish per-tokenizer descriptors with a
    shared id-prefix (e.g. `acme/strict-v3-llama3` + `acme/strict-v3-qwen2`).
    """


@dataclass(frozen=True)
class SafetyEnforcement:
    policy_id: str
    tokenizer_ids: tuple[str, ...]
    banned_token_ids: tuple[int, ...]
    grammar_specs: tuple[dict, ...]
    multi_token_patterns: tuple[dict, ...]
    categories: dict[str, str]

    @staticmethod
    def from_policy(policy: InternalPolicy) -> "SafetyEnforcement":
        return SafetyEnforcement(
            policy_id=policy.id,
            tokenizer_ids=tuple(policy.tokenizers),
            banned_token_ids=tuple(policy.banned_token_ids or ()),
            grammar_specs=tuple(policy.grammar_constraints or ()),
            multi_token_patterns=tuple(policy.multi_token_patterns or ()),
            categories={c.name: c.action for c in policy.categories},
        )

    @staticmethod
    def from_policy_id(policies_dir: Path, policy_id: str) -> "SafetyEnforcement":
        return SafetyEnforcement.from_policy(load_policy(policies_dir, policy_id))

    def assert_tokenizer_match(self, engine_tokenizer_id: str) -> None:
        """Raise `TokenizerMismatch` if the engine's tokenizer isn't bound
        to this policy. Call before installing the logits processor on
        an engine — it's the only check that catches a misalignment that
        would otherwise pass silently.
        """
        if engine_tokenizer_id not in self.tokenizer_ids:
            raise TokenizerMismatch(
                f"policy {self.policy_id!r} is bound to tokenizers "
                f"{list(self.tokenizer_ids)!r}; engine loaded "
                f"{engine_tokenizer_id!r}",
            )

    def make_logits_processor(self) -> LogitsProcessor:
        """Build the per-request logits processor.

        Slice 6 returns a single `BannedTokenLogitsProcessor`. Future
        slices (Aho-Corasick token-trie, embedding-space scorer) extend
        this to compose multiple processors via
        `safety_logits.compose(...)`.
        """
        return BannedTokenLogitsProcessor(self.banned_token_ids)
