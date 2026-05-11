# codec-supervisor — coverage

Last measured: 2026-05-11 (v0.4 release-cut)

## How

```
pip install pytest-cov
python -m pytest tests/ --cov=codec_supervisor --cov-report=term
```

## Result (v0.4 baseline)

```
TOTAL    1577    438    72%
159 passed
```

| Module                                            | Cov%  | Notes |
|---------------------------------------------------|------:|-------|
| `safety_logits.py`                                | 100%  | BannedTokenLogitsProcessor — pure-function masking |
| `safety_enforcement.py`                           | 100%  | wrapper that wires policy + tokenizer to logits + matcher |
| `safety_streaming.py`                             | 100%  | delay-k decisioning state machine |
| `schemas.py`                                      | 100%  | Pydantic models for internal + published descriptor |
| `safety_token_matcher.py`                         |  98%  | Aho-Corasick over int alphabets |
| `safety_classifier.py`                            |  94%  | Protocol + registry |
| `safety.py`                                       |  ~85% | sanitize() + load/save |
| `admin_safety.py`                                 |  ~80% | REST routes |
| `safety_classifiers/embedding_space.py`           |  76%  | engine-hidden-state path; tests use generator-DI |
| `safety_classifiers/llamaguard_3_1b.py`           |  66%  | 14-category — tests cover prompt/parse path with DI |
| `safety_classifiers/shieldgemma_2b.py`            |  54%  | 4-category — same DI pattern, less corpus diversity |
| `training/bootstrap_corpus.py`                    |  44%  | distillation pipeline — runs against real classifiers in integration runs |

## Intentionally uncovered

- The downloadable-weight code paths in each classifier (~30 % of
  each `safety_classifiers/*.py`) — tests use the generator-injection
  pattern (DI a `(text) → labels` callable) so the classifier-pipeline
  logic runs without loading Llama Guard 3 1B / ShieldGemma 2B
  weights. The actual `transformers` loading paths are exercised
  by the lab integration runs.
- `training/bootstrap_corpus.py` (44 %) — distillation pipeline runs
  end-to-end only with real classifier weights; covered in the lab
  integration suite, not the per-package pytest.

## v0.5 follow-up

- Coverage gap in ShieldGemma's category-specific branch logic.
  Add fixtures covering each of the 4 categories with positive
  + negative examples.
- Train + commit a tiny replay corpus for `bootstrap_corpus.py` so
  the distillation path runs in `pytest` without GPU.
- Wire CI to compute % per module + fail on regression.
