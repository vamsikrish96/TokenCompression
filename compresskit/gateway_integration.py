"""How to use compresskit in a model gateway.

The pattern in one line: **compress the payload, never the instructions.**

A prompt is usually your framing plus something bulky pasted into it -- a tool
result, a retrieved document, a log dump, an API response. The bulky part is
where the tokens are and where the redundancy is. Your instructions and the
user's question are small and load-bearing: compressing them is how "Do not
invent account numbers" becomes "Do not inven ...", leaving the model bound by
a rule it can no longer read.

So keep them separate. That is the whole integration.

Run this file directly to see it work with no API calls:

    python examples/gateway_integration.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Only needed to run this file from the repo. In your gateway, `compresskit`
# sits in your source tree and imports normally.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compresskit import Compressor, CompressorConfig, ReducerConfig

# Build the compressor once at module scope and reuse it. The handlers hold
# compiled patterns and, where tree-sitter is installed, cached parsers.
COMPRESSOR = Compressor(
    config=CompressorConfig(
        # Leave small payloads alone. The library default of 100 is far too low
        # for production: below a few thousand characters the marker text costs
        # more than the compression saves.
        min_content_length=2_000,
        # Keep ~40% of each reducible span. Lower is more aggressive; check the
        # answers before going below 0.3 on content you care about.
        reducer=ReducerConfig(target_ratio=0.4),
    )
)


@dataclass
class CompressionStats:
    """What one compression bought, for logging and dashboards."""

    content_type: str
    handler: str
    chars_before: int
    chars_after: int
    lossless_chars_saved: int

    @property
    def saved_percentage(self) -> float:
        """Share of characters removed."""
        if not self.chars_before:
            return 0.0
        return (1 - self.chars_after / self.chars_before) * 100


def compress_payload(payload: str) -> tuple[str, CompressionStats]:
    """Compress one bulky payload before it goes into a prompt.

    Args:
        payload: Tool output, a retrieved document, an API response -- the
            bulk. Not your instructions and not the user's question.

    Returns:
        ``(compressed_payload, stats)``. On any unexpected failure the original
        payload is returned unchanged: a compression bug must degrade your
        token bill, never your answers.
    """
    try:
        result = COMPRESSOR.compress(payload)
    except Exception:  # noqa: BLE001 - never let compression break a request
        return payload, CompressionStats("unknown", "none", len(payload), len(payload), 0)

    return result.compressed, CompressionStats(
        content_type=result.content_type.value,
        handler=result.handler_used,
        chars_before=len(result.original),
        chars_after=len(result.compressed),
        lossless_chars_saved=result.lossless_chars_saved,
    )


def build_messages(system_prompt: str, question: str, payload: str) -> list[dict]:
    """Assemble the request, compressing only the payload.

    Args:
        system_prompt: Your instructions. Sent verbatim.
        question: The user's question. Sent verbatim.
        payload: The bulk to compress.

    Returns:
        Messages ready for your provider's chat endpoint.
    """
    compressed, stats = compress_payload(payload)

    # Log it. You want this in your metrics: which handler ran, what it saved,
    # and how often compression was a no-op.
    print(
        f"[compresskit] {stats.content_type}/{stats.handler} "
        f"{stats.chars_before:,} -> {stats.chars_after:,} chars "
        f"({stats.saved_percentage:.1f}% saved, {stats.lossless_chars_saved:,} of it reversible)"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\n---\n{compressed}\n---"},
    ]


def main() -> None:
    """Demonstrate the pattern on a realistic tool result."""
    system_prompt = (
        "You answer questions about the data provided. Some of it may have been "
        "compressed, with sections replaced by '...' or collapsed with a repeat "
        "count. Answer from what is present, and say so if something needed is "
        "missing rather than guessing."
    )
    question = "How many accounts are dormant, and what is the request_id?"

    # Stand-in for a tool result: 60 records with verbose repeated boilerplate.
    payload = json.dumps(
        {
            "request_id": "req_9f2a4c8e1b7d",
            "accounts": [
                {
                    "account_id": f"ACC-{i:06d}",
                    "status": "dormant" if i % 3 == 0 else "active",
                    "review_note": (
                        "Reviewed during the quarterly compliance cycle. No exceptions "
                        "were raised by the reviewing officer and standard monitoring "
                        "continues to apply under the existing risk rating. "
                    ),
                }
                for i in range(60)
            ],
        },
        indent=2,
    )

    messages = build_messages(system_prompt, question, payload)

    # Hand `messages` to your provider exactly as you do today. Nothing about
    # the call changes -- the model never knows the payload was processed.
    #
    #     response = client.chat.completions.create(model=..., messages=messages)

    user_message = messages[1]["content"]
    print(f"\nsystem prompt : {len(system_prompt):,} chars, sent verbatim")
    print(f"question      : {len(question):,} chars, sent verbatim")
    print(f"user message  : {len(user_message):,} chars total")
    print("\nstill present in the compressed payload:")
    for label, needle in (
        ("request_id", "req_9f2a4c8e1b7d"),
        ("account ids", '"account_id"'),
        ("status field", '"status"'),
        ("the question", "How many accounts are dormant"),
        ("the instruction", "say so if something needed is missing"),
    ):
        found = needle in user_message or needle in messages[0]["content"]
        print(f"   {'yes' if found else 'NO '}  {label}")


if __name__ == "__main__":
    main()
