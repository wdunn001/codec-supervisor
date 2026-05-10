"""`ShieldGemma2B` — server-side tier-2 classifier (Google ShieldGemma 2B).

Per-category Yes/No classifier. ShieldGemma is built differently than
Llama Guard: instead of one generation that names violated categories,
it answers a separate "Does this content violate this single category?"
prompt per category, and the score is read off the next-token
probability of "Yes" vs "No". To produce a full ClassificationResult
we run the model `len(categories)` times — slower than Llama Guard but
finer-grained per-category control.

Categories (Google's harm taxonomy, distinct from Llama Guard's):
  - dangerous_content       — harmful "how to" advice
  - harassment              — targeted abuse
  - hate_speech             — protected-class slurs / dehumanization
  - sexually_explicit       — sexual content involving humans

The default factory lazy-imports transformers + torch (declared under
the `classifiers` extra) and reads the next-token "Yes" probability
directly from logits. Tests inject a `category_scorer` callable that
returns a Yes-probability per (prompt, category) pair without loading
weights.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from ..safety_classifier import (
    ClassificationInput,
    ClassificationResult,
    ClassifierInputForm,
    RegistryEntry,
    maybe_await,
    register,
)


SHIELDGEMMA_CATEGORIES: tuple[str, ...] = (
    "dangerous_content",
    "harassment",
    "hate_speech",
    "sexually_explicit",
)

# Human-readable guideline strings ShieldGemma's prompt template expects.
# Wording from the official model card.
_GUIDELINES: dict[str, str] = {
    "dangerous_content": (
        "\"No Dangerous Content\": The chatbot shall not generate content that "
        "harming oneself and/or others (e.g., accessing or building firearms "
        "and explosive devices, promotion of terrorism, instructions for suicide)."
    ),
    "harassment": (
        "\"No Harassment\": The chatbot shall not generate content that is "
        "malicious, intimidating, bullying, or abusive content targeting another "
        "individual (e.g., physical threats, denial of tragic events, "
        "disparaging victims of violence)."
    ),
    "hate_speech": (
        "\"No Hate Speech\": The chatbot shall not generate content that "
        "targeting identity and/or protected attributes (e.g., racial slurs, "
        "promotion of discrimination, calls to violence against protected "
        "groups) or dehumanizing/belittling/vilifying based on race, ethnicity, "
        "religion, disability, age, nationality, veteran status, sexual "
        "orientation, gender, gender identity, caste, or any other protected "
        "status."
    ),
    "sexually_explicit": (
        "\"No Sexually Explicit Information\": The chatbot shall not generate "
        "content that contains references to sexual acts or other lewd content "
        "(e.g., sexually graphic descriptions, content aimed at causing arousal). "
        "Medical or scientific terms related to human anatomy or sex education "
        "are permitted."
    ),
}


def build_shieldgemma_prompt(user_message: str, category: str) -> str:
    """Build the per-category ShieldGemma prompt.

    Format from the model card. The expected next token is `Yes` (violates
    the policy) or `No` (does not).
    """
    if category not in _GUIDELINES:
        raise ValueError(
            f"unknown ShieldGemma category {category!r}; "
            f"valid: {sorted(_GUIDELINES.keys())}",
        )
    return (
        "You are a policy expert trying to help determine whether a user "
        "prompt is in violation of the defined safety policies.\n\n"
        f"<start_of_turn>\nHuman Question: {user_message.strip()}\n<end_of_turn>\n\n"
        "Our safety principle is defined in the below:\n\n"
        f"* {_GUIDELINES[category]}\n\n"
        "===\n\n"
        "Does the human question violate the above principle? Your answer must "
        "start with 'Yes' or 'No'. And then walk through step by step to be sure "
        "we answer correctly.\n\n"
    )


# Generator signature: given (prompt, category), return Yes-probability ∈ [0, 1].
# MAY be sync or async.
ShieldGemmaCategoryScorer = Callable[
    [str, str],
    Awaitable[float] | float,
]


class ShieldGemma2B:
    """Server-side ShieldGemma 2B classifier.

    Construct with an injected `category_scorer` for tests; default
    factory lazy-imports transformers + torch and reads the
    Yes-probability from logits over the model's vocabulary.

    Loading is idempotent and async-safe.
    """

    MODEL_ID: str = "ShieldGemma-2B"
    HF_MODEL_ID_DEFAULT: str = "google/shieldgemma-2b"

    def __init__(
        self,
        *,
        category_scorer: ShieldGemmaCategoryScorer | None = None,
        hf_model_id: str | None = None,
        categories: tuple[str, ...] = SHIELDGEMMA_CATEGORIES,
    ) -> None:
        self._scorer: ShieldGemmaCategoryScorer | None = category_scorer
        self._hf_model_id = hf_model_id or self.HF_MODEL_ID_DEFAULT
        self._categories = categories
        self._loaded = category_scorer is not None
        self._load_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def requires(self) -> ClassifierInputForm:
        return "text"

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    async def load(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            self._scorer = await self._build_default_scorer()
            self._loaded = True

    async def unload(self) -> None:
        self._scorer = None
        self._loaded = False

    async def capability(self) -> str | None:
        if self._scorer is not None:
            return None
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            return f"transformers not installed: {e}"
        try:
            import torch  # noqa: F401
        except ImportError as e:
            return f"torch not installed: {e}"
        return None

    async def score(self, input: ClassificationInput) -> ClassificationResult:
        if input.form != "text":
            raise ValueError(
                f"ShieldGemma2B only accepts form='text'; got form={input.form!r}",
            )
        if not isinstance(input.payload, str):
            raise ValueError(
                "ShieldGemma2B: text input payload must be a string",
            )
        await self.load()
        assert self._scorer is not None

        # ShieldGemma is per-category. Run them concurrently so the
        # 4-call latency is one-call wall time.
        tasks: list[Awaitable[float] | float] = []
        for cat in self._categories:
            prompt = build_shieldgemma_prompt(input.payload, cat)
            tasks.append(maybe_await(self._scorer(prompt, cat)))  # type: ignore[arg-type]

        # `tasks` may contain awaitables and/or sync return values. Run any
        # awaitables in parallel; pass through plain floats.
        async def _ensure_float(v: Awaitable[float] | float) -> float:
            if hasattr(v, "__await__"):
                v = await v  # type: ignore[assignment]
            v_float = float(v)  # type: ignore[arg-type]
            if not 0.0 <= v_float <= 1.0:
                raise ValueError(
                    f"ShieldGemma category scorer returned {v_float!r}; "
                    "expected a probability in [0, 1]",
                )
            return v_float

        results = await asyncio.gather(*[_ensure_float(t) for t in tasks])
        scores: dict[str, float] = dict(zip(self._categories, results))
        return ClassificationResult(scores=scores, raw={"per_category": scores})

    # -------------------------------------------------------------- internals

    async def _build_default_scorer(self) -> ShieldGemmaCategoryScorer:
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            import torch  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "ShieldGemma2B's default scorer needs the 'classifiers' extra "
                f"installed: pip install 'codec-supervisor[classifiers]'. "
                f"Underlying error: {e}",
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(self._hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self._hf_model_id,
            torch_dtype=torch.float16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Pre-resolve the "Yes" / "No" token IDs once.
        yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

        async def _score(prompt: str, _category: str) -> float:
            def _sync() -> float:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                # Last-token logits; softmax over the (Yes, No) pair.
                last = logits[0, -1, :]
                yes_logit = last[yes_id].item()
                no_logit = last[no_id].item()
                # Two-way softmax — equivalent to sigmoid(yes - no) for
                # numerical stability across mixed precisions.
                import math
                exp_yes = math.exp(yes_logit - max(yes_logit, no_logit))
                exp_no = math.exp(no_logit - max(yes_logit, no_logit))
                return exp_yes / (exp_yes + exp_no)

            return await asyncio.to_thread(_sync)

        return _score


# ── Registration ─────────────────────────────────────────────────────────────


def register_shieldgemma_2b(
    *,
    category_scorer: ShieldGemmaCategoryScorer | None = None,
    hf_model_id: str | None = None,
) -> None:
    """Add ShieldGemma 2B to the registry. Idempotent."""
    from ..safety_classifier import has_classifier

    if has_classifier(ShieldGemma2B.MODEL_ID):
        return
    register(
        RegistryEntry(
            model_id=ShieldGemma2B.MODEL_ID,
            factory=lambda: ShieldGemma2B(
                category_scorer=category_scorer,
                hf_model_id=hf_model_id,
            ),
            tier=2,
            description=(
                "Google ShieldGemma 2B — per-category Yes/No classifier; "
                "4-category Google harm taxonomy. Tier-2 opt-in (slower, "
                "finer-grained than Llama Guard)."
            ),
            advertised_categories=SHIELDGEMMA_CATEGORIES,
            advertised_requires="text",
        ),
    )
