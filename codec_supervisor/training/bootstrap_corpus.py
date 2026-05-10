"""Bootstrap a labeled corpus for embedding-space classifier training.

Reads an input JSONL of `{"text": "..."}` rows, runs each through one
or more registered text-space classifiers (Llama Guard, ShieldGemma —
or any classifier with `requires="text"`), and writes a labeled JSONL:

    {
      "text":   "<original text>",
      "labels": { "<model_id>": { "<category>": <score>, ... }, ... },
      "ts":     "2026-05-10T12:34:56Z"
    }

A downstream trainer ingests this corpus, computes engine hidden
states for each row's text (via a frozen encoder of the operator's
choice), and fits a BERT-tier classification head whose targets are
the labels above. The trained head plugs back into
`EmbeddingSpaceClassifier` via `scorer=` or `head_weights_path=`.

Distillation rationale: the embedding-space classifier doesn't need
to be smarter than its labeler — it only needs to be *cheaper*. By
distilling a 1B-param Llama Guard's judgments into a small head over
engine hidden states, we trade a few percent of accuracy for ~100×
inference cost (no text-space pass; just one MatMul per token over the
hidden state).

Usage:

    python -m codec_supervisor.training.bootstrap_corpus \\
        --input prompts.jsonl \\
        --output labeled.jsonl \\
        --classifiers Llama-Guard-3-1B,ShieldGemma-2B

Both classifiers are registered with their default in-process loaders
(needs the `classifiers` extra installed). Pass `--classifiers` to
restrict to a subset, or register custom classifiers before invoking.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable

from ..safety_classifier import (
    ClassificationInput,
    SafetyClassifier,
    list_registered,
    resolve_classifier,
)


logger = logging.getLogger(__name__)


# ── I/O ──────────────────────────────────────────────────────────────────────


def iter_input_rows(path: Path) -> Iterable[dict]:
    """Stream JSONL rows from disk, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {e}",
                ) from e


# ── Labeling ─────────────────────────────────────────────────────────────────


async def _resolve_classifiers(
    requested_ids: list[str],
) -> list[tuple[str, SafetyClassifier]]:
    """Resolve each requested classifier id; skip ones that aren't
    capable on this host (with a warning), so operators on a CPU box
    can still bootstrap a corpus using whatever classifiers are
    installed.
    """
    resolved: list[tuple[str, SafetyClassifier]] = []
    for cid in requested_ids:
        try:
            r = await resolve_classifier(cid, allow_fallback=False)
        except Exception as e:  # noqa: BLE001 — see docstring
            logger.warning("skipping %r: %s", cid, e)
            continue
        if r.classifier.requires != "text":
            logger.warning(
                "skipping %r: requires=%r (only text-space classifiers "
                "are useful for the bootstrap)",
                cid,
                r.classifier.requires,
            )
            continue
        resolved.append((cid, r.classifier))
    return resolved


async def label_row(
    text: str,
    classifiers: list[tuple[str, SafetyClassifier]],
) -> dict:
    """Run every classifier on `text` and return the merged labels
    block. Classifiers run concurrently — total wall time is the
    slowest single classifier.
    """
    inp = ClassificationInput(form="text", payload=text)

    async def _one(model_id: str, c: SafetyClassifier) -> tuple[str, dict]:
        result = await c.score(inp)
        return model_id, dict(result.scores)

    pairs = await asyncio.gather(*[_one(mid, c) for mid, c in classifiers])
    return {model_id: scores for model_id, scores in pairs}


async def label_corpus(
    rows: Iterable[dict],
    classifiers: list[tuple[str, SafetyClassifier]],
    *,
    text_key: str = "text",
) -> AsyncIterator[dict]:
    """Yield labeled rows in input order. Carries forward any extra
    fields present on the input row (so a corpus that already has
    `prompt_id` etc. survives the labeling pass)."""
    for row in rows:
        text = row.get(text_key)
        if not isinstance(text, str):
            logger.warning("skipping row: missing or non-string %r field", text_key)
            continue
        labels = await label_row(text, classifiers)
        yield {
            **row,
            "labels": labels,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bootstrap_corpus",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="JSONL of {'text': '...'} rows",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSONL with labels[<classifier>][<category>] = score",
    )
    p.add_argument(
        "--classifiers",
        type=str,
        default=None,
        help="Comma-separated model ids. Default: every registered "
             "text-space classifier.",
    )
    p.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="JSON key holding the text field (default: 'text')",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N rows (debugging)",
    )
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    if args.classifiers:
        requested = [s.strip() for s in args.classifiers.split(",") if s.strip()]
    else:
        requested = [
            e.model_id
            for e in list_registered()
            if e.advertised_requires == "text"
        ]
    if not requested:
        print(
            "no text-space classifiers registered; "
            "import codec_supervisor.safety_classifiers and call "
            "register_default_classifiers() before invoking",
            file=sys.stderr,
        )
        return 2

    classifiers = await _resolve_classifiers(requested)
    if not classifiers:
        print("no classifiers resolved successfully", file=sys.stderr)
        return 2

    logger.info("labeling with: %s", [mid for mid, _ in classifiers])

    rows = iter_input_rows(args.input)
    if args.limit is not None:
        rows = (r for i, r in enumerate(rows) if i < args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        async for labeled in label_corpus(rows, classifiers, text_key=args.text_key):
            out.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            written += 1
    logger.info("wrote %d labeled rows to %s", written, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_amain(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
