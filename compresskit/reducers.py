"""Rule-based reducers for the non-structural parts of content.

Upstream Headroom hands compressible spans to Kompress, a learned compressor.
This package cannot, so the spans are reduced by deterministic rules instead:

1. Collapse runs of consecutive identical lines into one line plus a count.
2. Collapse runs of whitespace.
3. Truncate to a character budget, keeping a head and a tail.

Two invariants hold for every reducer here, because a reduced span is spliced
back into content that may still have to parse:

* **No new newlines.** The newlines in the output are a subset of the ones in
  the input. A span taken from inside a JSON string value contains no raw
  newlines, so the reduced span cannot introduce one -- RFC 8259 section 7
  forbids unescaped control characters in strings.
* **No dangling backslash.** Truncation never leaves a span ending in an odd
  run of backslashes, which would escape the closing quote of a JSON string.

Reduction is lossy and one-way. Keep ``CompressionResult.original`` if you need
the input back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Runs of spaces/tabs, and runs of blank lines. Kept as two patterns so that
# horizontal whitespace can be collapsed without touching line structure.
_HORIZONTAL_WS_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")


@dataclass
class ReducerConfig:
    """Configuration for :func:`reduce_text`.

    Attributes:
        target_ratio: Fraction of the input length to aim for (0.0-1.0).
            0.3 means "cut to roughly 30% of the characters".
        dedupe_lines: Collapse runs of consecutive identical lines.
        collapse_whitespace: Collapse runs of spaces, tabs and blank lines.
        min_length: Spans shorter than this are returned unchanged. Reducing a
            short span costs more in marker characters than it saves.
        marker: Inserted where truncation removed the middle. Must contain no
            control characters -- see the module docstring.
        head_fraction: Share of the budget spent on the head of the span. The
            remainder goes to the tail. Front-loaded because the start of a
            span is usually the more informative end.
        snap_to_lines: Move truncation cuts to the nearest line boundary when
            the span has lines, so whole lines survive instead of fragments of
            identifiers. Costs a little ratio, buys output that still reads as
            the thing it came from.
    """

    target_ratio: float = 0.3
    dedupe_lines: bool = True
    collapse_whitespace: bool = True
    min_length: int = 50
    marker: str = " ... "
    head_fraction: float = 2 / 3
    snap_to_lines: bool = True


def collapse_repeated_lines(text: str, min_run: int = 2) -> str:
    """Collapse runs of consecutive identical lines into one line plus a count.

    This is where most of the win comes from on logs and on machine-generated
    output, where the same line repeats hundreds of times. Only *consecutive*
    runs are collapsed, so the result stays a faithful ordering of the input.

    Comparison ignores leading and trailing whitespace, but the first line of
    the run is emitted verbatim, so indentation is preserved.

    Args:
        text: Text to process.
        min_run: Minimum run length before a run is collapsed.

    Returns:
        Text with repeated-line runs collapsed.

    Example:
        >>> collapse_repeated_lines("a\\na\\na\\nb\\n")
        'a  [x3]\\nb\\n'
    """
    if min_run < 2 or "\n" not in text:
        return text

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i + 1
        key = lines[i].strip()
        # Never collapse blank lines here; _BLANK_LINES_RE handles those and
        # would otherwise produce a meaningless "[x12]" marker.
        if key:
            while j < len(lines) and lines[j].strip() == key:
                j += 1
        run = j - i
        if run >= min_run:
            out.append(f"{lines[i]}  [x{run}]")
        else:
            out.extend(lines[i:j])
        i = j
    return "\n".join(out)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace and runs of blank lines.

    Never introduces a newline: blank-line runs are replaced by a single
    newline, which the input already contained.

    Args:
        text: Text to process.

    Returns:
        Text with whitespace runs collapsed.
    """
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    return _BLANK_LINES_RE.sub("\n", text)


def trim_dangling_escape(text: str) -> str:
    """Drop a trailing backslash run that would escape the next character.

    Truncating in the middle of a JSON string can leave ``...\\`` or a partial
    ``\\uXXXX`` escape, either of which escapes the closing quote and breaks the
    document. Removing the odd backslash costs one character and keeps the
    result parseable.

    Args:
        text: Text that may end mid-escape.

    Returns:
        Text with any odd trailing backslash removed.
    """
    trailing = len(text) - len(text.rstrip("\\"))
    if trailing % 2 == 1:
        return text[:-1]
    return text


def escape_interior_positions(text: str) -> set[int]:
    """Indices that fall *inside* a backslash escape sequence.

    Cutting at one of these splits an escape: ``\\uXXXX`` becomes ``\\u65`` and
    ``\\"`` becomes a lone backslash. Either breaks a JSON document, and
    ``json.dumps`` emits ``\\uXXXX`` for every non-ASCII character by default,
    so any content with accents or CJK is full of them.

    Args:
        text: The span being cut.

    Returns:
        Set of indices no cut may land on. Index of the backslash itself is
        safe -- a cut there drops the whole escape.
    """
    if "\\" not in text:
        return set()

    interior: set[int] = set()
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "\\":
            width = 6 if text[index : index + 2] == "\\u" else 2
            interior.update(range(index + 1, min(index + width, length)))
            index += width
        else:
            index += 1
    return interior


def _safe_cut(index: int, interior: set[int]) -> int:
    """Move a cut point back until it no longer splits an escape."""
    while index > 0 and index in interior:
        index -= 1
    return index


def truncate_middle(text: str, budget: int, config: ReducerConfig) -> str:
    """Keep the head and tail of ``text``, dropping the middle.

    When the span contains newlines and ``snap_to_lines`` is set, the cut is
    moved to the nearest line boundary. Cutting mid-token turns
    ``self.product_id = product_id`` into ``self.product_id = pro`` -- text that
    is no longer valid code and reads as a typo rather than an omission.
    Whole lines survive as whole lines, and the marker sits on its own.

    Args:
        text: Text to truncate.
        budget: Target character count for the kept content, excluding the
            marker.
        config: Reducer configuration, for the marker and head/tail split.

    Returns:
        ``head + marker + tail``, or ``text`` unchanged when it already fits or
        when the budget is too small to say anything useful.
    """
    if budget <= 0 or len(text) <= budget:
        return text

    keep_head = int(budget * config.head_fraction)
    keep_tail = budget - keep_head
    head_end = keep_head
    tail_start = len(text) - keep_tail if keep_tail > 0 else len(text)

    if config.snap_to_lines and "\n" in text:
        # Snap the head back to the end of the last complete line, and the
        # tail forward to the start of the next one. Fall back to the raw
        # offsets when a line is longer than the whole budget.
        # rfind returning 0 is a real newline at index 0, not "not found";
        # -1 is the miss. Testing `> 0` silently skipped the snap on any span
        # that begins with a newline, which is most function bodies.
        snapped_head = text.rfind("\n", 0, head_end)
        if snapped_head >= 0:
            head_end = snapped_head + 1
        # Snap the tail BACK to the start of the line it lands in, not forward
        # to the next one: forward-snapping onto a trailing newline yields an
        # empty tail, and the fallback then cuts mid-identifier anyway. Going
        # back keeps one whole line and slightly overshoots the budget, which
        # is the cheaper mistake.
        snapped_tail = text.rfind("\n", 0, tail_start)
        if snapped_tail >= 0:
            tail_start = snapped_tail + 1

        if tail_start <= head_end:
            return text

    # Never cut through a backslash escape. Both cut points move backwards:
    # the head sheds a partial escape, and the tail moves back to the escape's
    # own backslash so it survives whole.
    interior = escape_interior_positions(text)
    if interior:
        head_end = _safe_cut(head_end, interior)
        tail_start = _safe_cut(tail_start, interior)
        if tail_start <= head_end:
            return text

    head = trim_dangling_escape(text[:head_end])
    tail = text[tail_start:]

    candidate = head + config.marker + tail
    # Truncation that grows the span is not truncation. Short spans with a
    # long marker hit this.
    return candidate if len(candidate) < len(text) else text


def reduce_text(text: str, config: ReducerConfig | None = None) -> str:
    """Reduce a compressible span by rules alone.

    Applies, in order: line dedup, whitespace collapse, then truncation to the
    remaining budget. The cheap structural passes run first so that truncation
    only has to remove what dedup and collapsing could not.

    Args:
        text: The span to reduce.
        config: Reducer configuration. Defaults to :class:`ReducerConfig`.

    Returns:
        The reduced span. Never longer than ``text``.

    Example:
        >>> cfg = ReducerConfig(target_ratio=0.5)
        >>> reduce_text("x" * 200, cfg).count("...")
        1
    """
    config = config or ReducerConfig()

    if not text or len(text) < config.min_length:
        return text

    budget = int(len(text) * config.target_ratio)
    result = text

    if config.dedupe_lines:
        result = collapse_repeated_lines(result)
    if config.collapse_whitespace:
        result = collapse_whitespace(result)

    if len(result) > budget:
        result = truncate_middle(result, budget, config)

    # The passes above are independently safe, but a pathological marker or
    # dedup annotation could still add characters; never return a longer span.
    return result if len(result) <= len(text) else text
