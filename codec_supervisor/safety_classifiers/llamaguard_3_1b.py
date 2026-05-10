"""`LlamaGuard31B` — server-side tier-1 classifier (Meta Llama Guard 3 1B).

Generative classifier. Builds a structured prompt with the 14-category
MLCommons taxonomy declared, runs a short generation, parses the
two-line "safe / unsafe + S-codes" response. Mirrors the browser side
(`@codecai/web-safety/classifiers/llama-guard-3-1b.ts`) so both report
the same canonical category names — admin policies map identically
across server and browser.

Weights come from `meta-llama/Llama-Guard-3-1B` on Hugging Face. The
default factory lazy-imports `transformers` + `torch` (optional deps,
declared under the `classifiers` extra in pyproject.toml) and spins
up an async-friendly generation pipeline. Tests inject their own
generator callable via the `generator=` constructor argument and never
need the real weights.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..safety_classifier import (
    ClassificationInput,
    ClassificationResult,
    ClassifierInputForm,
    RegistryEntry,
    maybe_await,
    register,
)


# ── Llama Guard 3 taxonomy ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _LGCategory:
    code: str
    name: str
    title: str


_LG_CATEGORIES: tuple[_LGCategory, ...] = (
    _LGCategory("S1",  "violent_crimes",         "Violent Crimes"),
    _LGCategory("S2",  "non_violent_crimes",     "Non-Violent Crimes"),
    _LGCategory("S3",  "sex_crimes",             "Sex Crimes"),
    _LGCategory("S4",  "child_exploitation",     "Child Exploitation"),
    _LGCategory("S5",  "defamation",             "Defamation"),
    _LGCategory("S6",  "specialized_advice",     "Specialized Advice"),
    _LGCategory("S7",  "privacy",                "Privacy"),
    _LGCategory("S8",  "intellectual_property",  "Intellectual Property"),
    _LGCategory("S9",  "indiscriminate_weapons", "Indiscriminate Weapons"),
    _LGCategory("S10", "hate",                   "Hate"),
    _LGCategory("S11", "self_harm",              "Self-Harm"),
    _LGCategory("S12", "sexual_content",         "Sexual Content"),
    _LGCategory("S13", "elections",              "Elections"),
    _LGCategory("S14", "code_interpreter_abuse", "Code Interpreter Abuse"),
)

_BY_CODE = {c.code: c for c in _LG_CATEGORIES}

LLAMA_GUARD_CATEGORIES: tuple[str, ...] = tuple(c.name for c in _LG_CATEGORIES)


# ── Prompt + parser ──────────────────────────────────────────────────────────


def build_llama_guard_prompt(user_message: str) -> str:
    """Build the Llama Guard 3 prompt for a single user message.

    Format matches Meta's reference exactly — the official chat
    template wraps user + assistant turns with the same structure.
    Exposed as a top-level function so out-of-band hosts (eval scripts,
    direct vLLM calls) can reuse the bit-exact prompt.
    """
    cats = "\n".join(f"{c.code}: {c.title}." for c in _LG_CATEGORIES)
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Task: Check if there is unsafe content in 'User' messages in conversations "
        "according our safety policy with the below categories.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        f"{cats}\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "<BEGIN CONVERSATION>\n\n"
        f"User: {user_message}\n\n"
        "<END CONVERSATION>\n\n"
        "Provide your safety assessment for ONLY THE LAST User message in the above "
        "conversation:\n"
        " - First line must read 'safe' or 'unsafe'.\n"
        " - If unsafe, a second line must include a comma-separated list of violated "
        "categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_llama_guard_output(text: str) -> ClassificationResult:
    """Parse Llama Guard's two-line completion into per-category scores.

    `safe` → all-zero scores. `unsafe\\nS5,S10` → those S-codes mapped to
    canonical category names with score 1.0. Out-of-range or unparseable
    category lists fall back to a generic `unsafe=1` so policies can
    still act when the model deviates from the prescribed format.
    """
    trimmed = text.strip()
    lines = [ln.strip() for ln in trimmed.split("\n")]
    first = (lines[0] if lines else "").lower()

    scores: dict[str, float] = {c.name: 0.0 for c in _LG_CATEGORIES}

    if first == "safe":
        return ClassificationResult(scores=scores, raw={"first_line": first, "body": trimmed})

    second = lines[1] if len(lines) > 1 else ""
    raw_codes = [tok.strip().upper() for tok in second.replace(",", " ").split()]
    valid_codes = [c for c in raw_codes if c in _BY_CODE]

    if not valid_codes:
        scores["unsafe"] = 1.0
        return ClassificationResult(
            scores=scores,
            raw={
                "first_line": first,
                "body": trimmed,
                "unparseable_categories": second,
            },
        )

    for code in valid_codes:
        cat = _BY_CODE[code]
        scores[cat.name] = 1.0
    return ClassificationResult(
        scores=scores,
        raw={"first_line": first, "body": trimmed, "codes": valid_codes},
    )


# ── Implementation ───────────────────────────────────────────────────────────


# Generator signature (DI for tests). MAY be sync or async; the class
# normalizes via `maybe_await`.
LlamaGuardGenerator = Callable[[str], Awaitable[str] | str]


class LlamaGuard31B:
    """Server-side Llama Guard 3 1B classifier.

    Construct with an injected `generator` for tests; default factory
    lazy-imports transformers + torch and spins up a small generation
    pipeline. Loading is idempotent and async-safe (concurrent
    `score()` calls before `load()` completes share one load future).
    """

    MODEL_ID: str = "Llama-Guard-3-1B"
    HF_MODEL_ID_DEFAULT: str = "meta-llama/Llama-Guard-3-1B"

    def __init__(
        self,
        *,
        generator: LlamaGuardGenerator | None = None,
        hf_model_id: str | None = None,
    ) -> None:
        self._generator: LlamaGuardGenerator | None = generator
        self._hf_model_id = hf_model_id or self.HF_MODEL_ID_DEFAULT
        self._loaded = generator is not None
        self._load_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def requires(self) -> ClassifierInputForm:
        return "text"

    @property
    def categories(self) -> tuple[str, ...]:
        return LLAMA_GUARD_CATEGORIES

    async def load(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            self._generator = await self._build_default_generator()
            self._loaded = True

    async def unload(self) -> None:
        # Drop the reference; transformers + torch GC the underlying
        # weights when the last strong ref is gone.
        self._generator = None
        self._loaded = False

    async def capability(self) -> str | None:
        # If a generator was injected, capability is always OK — tests
        # and out-of-process generators don't need transformers locally.
        if self._generator is not None:
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
                f"LlamaGuard31B only accepts form='text'; got form={input.form!r}",
            )
        if not isinstance(input.payload, str):
            raise ValueError(
                "LlamaGuard31B: text input payload must be a string",
            )
        await self.load()
        assert self._generator is not None
        prompt = build_llama_guard_prompt(input.payload)
        out = await maybe_await(self._generator(prompt))
        if not isinstance(out, str):
            raise TypeError(
                f"LlamaGuard31B generator returned {type(out).__name__}, expected str",
            )
        return parse_llama_guard_output(out)

    # -------------------------------------------------------------- internals

    async def _build_default_generator(self) -> LlamaGuardGenerator:
        # Lazy-imported here so the optional dep is touched only when
        # the default generator is actually constructed. Tests that pass
        # `generator=` skip this branch entirely.
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            import torch  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "LlamaGuard31B's default generator needs the 'classifiers' extra "
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

        async def _generate(prompt: str) -> str:
            # Run the synchronous transformers call in a worker thread
            # so we don't block the FastAPI event loop.
            def _sync() -> str:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    temperature=0.0,
                )
                # The model includes the prompt tokens in `output_ids`;
                # slice to just the new generation.
                new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
                return tokenizer.decode(new_tokens, skip_special_tokens=True)

            return await asyncio.to_thread(_sync)

        return _generate


# ── Registration ─────────────────────────────────────────────────────────────


def register_llamaguard_3_1b(
    *,
    generator: LlamaGuardGenerator | None = None,
    hf_model_id: str | None = None,
) -> None:
    """Add Llama Guard 3 1B to the registry. Idempotent: a second call
    is a no-op rather than raising, so module-load auto-registration
    survives re-imports / hot-reload.
    """
    from ..safety_classifier import has_classifier

    if has_classifier(LlamaGuard31B.MODEL_ID):
        return
    register(
        RegistryEntry(
            model_id=LlamaGuard31B.MODEL_ID,
            factory=lambda: LlamaGuard31B(
                generator=generator,
                hf_model_id=hf_model_id,
            ),
            tier=1,
            description=(
                "Meta Llama Guard 3 1B — generative; 14-category MLCommons taxonomy. "
                "Default tier-1 server classifier."
            ),
            advertised_categories=LLAMA_GUARD_CATEGORIES,
            advertised_requires="text",
        ),
    )
