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
  forbids unescaped control characters in strings. The single exception is
  ``ReducerConfig.comment_prefix``, which the compressor sets only for source
  code; see :func:`_omission_marker`.
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
#
# The lookbehind restricts the collapse to *interior* runs. Without it, leading
# indentation collapsed too: "        self._cache = {}" came back as
# " self._cache = {}". In Python that is not a formatting nicety, it is a
# change of meaning, and it left the reduced span unable to parse.
_HORIZONTAL_WS_RE = re.compile(r"(?<=\S)[ \t]{2,}")
# Only whitespace that sits on an otherwise BLANK line is part of the run. The
# previous form ended with a trailing "[ \t]*" that reached past the final
# newline and swallowed the indentation of the next real line, so
# "…= {}\n\n    def place_order(" came back as "…= {}\ndef place_order(" --
# dedented out of its class.
_BLANK_LINES_RE = re.compile(r"\n(?:[ \t]*\n)+")


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
        comment_prefix: Line-comment token of the surrounding language ("#",
            "//"). When set, and when the cut landed on line boundaries, the
            removed middle is replaced by a comment line stating how many lines
            went -- ``# ... 12 lines omitted ...`` -- instead of an inline
            ``" ... "``. The inline marker leaves output that no longer parses
            and gives no clue whether two lines or two hundred were dropped.
            This is the one case that adds a newline, so it is opt-in and the
            compressor only sets it for code.
    """

    target_ratio: float = 0.3
    dedupe_lines: bool = True
    collapse_whitespace: bool = True
    min_length: int = 50
    marker: str = " ... "
    head_fraction: float = 2 / 3
    snap_to_lines: bool = True
    comment_prefix: str | None = None


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


def _line_indent(line: str) -> int:
    """Width of the leading whitespace run."""
    return len(line) - len(line.lstrip(" \t"))


def _skip_literal(text: str, index: int) -> int:
    """Index just past the string literal starting at ``index``."""
    quote = text[index]
    if text.startswith(quote * 3, index):
        end = text.find(quote * 3, index + 3)
        return len(text) if end == -1 else end + 3
    i = index + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote or text[i] == "\n":
            return i + 1
        i += 1
    return len(text)


def _bracket_balance(text: str, comment_prefix: str | None) -> tuple[int, int]:
    """``(net depth change, lowest depth reached)``, ignoring strings.

    A removed chunk is safe to drop only when it is self-contained: it must
    not close a bracket that was opened before it (lowest < 0), and must not
    leave one open behind it (net != 0).
    """
    depth = 0
    lowest = 0
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in "\"'":
            i = _skip_literal(text, i)
            continue
        if comment_prefix and text.startswith(comment_prefix, i):
            newline = text.find("\n", i)
            i = n if newline == -1 else newline
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            lowest = min(lowest, depth)
        i += 1
    return depth, lowest


def _block_safe_cuts(
    text: str,
    head_end: int,
    tail_start: int,
    comment_prefix: str | None,
) -> tuple[int, int]:
    """Move the cut points so the surviving lines still nest legally.

    Keeping whole lines is not enough where indentation carries meaning.
    Dropping ``if flag:`` but keeping the ``total *= 2`` beneath it leaves a
    line indented under nothing; keeping ``for item in items:`` but dropping
    everything under it leaves a header with no body. Both are syntax errors,
    and both look like corruption rather than omission.

    So the head sheds any trailing line that opens a block, and the tail
    advances to the next line no deeper than the head's last kept line. Costs
    a few more lines than the budget asked for; buys output that still parses.
    """
    line_starts = [0] + [i + 1 for i, char in enumerate(text) if char == "\n"]

    def start_of_line(offset: int) -> int:
        best = 0
        for start in line_starts:
            if start > offset:
                break
            best = start
        return best

    def line_at(start: int) -> str:
        end = text.find("\n", start)
        return text[start : end if end != -1 else len(text)]

    def last_nonblank(before: int) -> str | None:
        """The last line with content that ends at or before ``before``."""
        cursor = before
        while cursor > 0:
            start = start_of_line(cursor - 1)
            line = line_at(start)
            if line.strip():
                return line
            cursor = start
        return None

    while head_end > 0:
        previous = last_nonblank(head_end)
        if previous is not None and previous.rstrip().endswith((":", "{")):
            head_end = start_of_line(head_end - 1)
        else:
            break

    # The head of a body span is often just the span's leading newline, whose
    # "line" is empty. Reading an indent of 0 off it makes every real body
    # line look too deep, and the tail search then walks off the end and drops
    # the body entirely -- leaving a def with no statements under it.
    anchor = last_nonblank(head_end)
    if anchor is not None:
        limit = _line_indent(anchor)
    else:
        limit = min(
            (_line_indent(line) for line in text.split("\n") if line.strip()),
            default=0,
        )

    def droppable(stop: int) -> bool:
        """Can text[head_end:stop] be removed without breaking the code?"""
        delta, lowest = _bracket_balance(text[head_end:stop], comment_prefix)
        return delta == 0 and lowest == 0

    while tail_start < len(text):
        candidate = line_at(tail_start)
        if candidate.strip() and _line_indent(candidate) <= limit and droppable(tail_start):
            break
        newline = text.find("\n", tail_start)
        if newline == -1:
            tail_start = len(text)
            break
        tail_start = newline + 1

    # No cut in this span is safe -- collapsing it back onto the head tells
    # the caller to leave the span alone. Losing the compression on one span
    # is cheaper than emitting a file with an unmatched bracket in it.
    if not droppable(tail_start):
        return head_end, head_end

    return head_end, tail_start


def _omission_marker(
    text: str,
    head: str,
    head_end: int,
    tail_start: int,
    config: ReducerConfig,
) -> str:
    """What to splice in where the middle was removed.

    Defaults to the inline ``config.marker``. When the caller supplied a
    ``comment_prefix`` and the cut fell on line boundaries, returns a comment
    line naming the number of lines dropped instead, indented to match the
    code it sits between. That is the difference between output the model can
    still read as code and output that merely looks corrupted.
    """
    removed = text[head_end:tail_start]
    if not config.comment_prefix or "\n" not in removed:
        return config.marker
    # An empty head is a clean boundary too -- it means the whole head was
    # dropped rather than cut mid-line.
    if head and not head.endswith("\n"):
        return config.marker

    # Indent the marker like the code it sits between. The tail is the better
    # guide; when the cut ran to the end of the span there is no tail, so the
    # head's last line answers instead -- otherwise the comment lands at
    # column 0 inside an indented body.
    line_end = text.find("\n", tail_start)
    tail_line = text[tail_start : line_end if line_end != -1 else len(text)]
    if not tail_line.strip():
        tail_line = next(
            (line for line in reversed(head.split("\n")) if line.strip()),
            "",
        )
    indent = tail_line[: len(tail_line) - len(tail_line.lstrip(" \t"))]
    count = removed.count("\n")
    unit = "line" if count == 1 else "lines"
    return f"{indent}{config.comment_prefix} ... {count} {unit} omitted ...\n"


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

    # A single physical code line has no *interior* newline for the
    # snap-to-lines logic below to snap to -- a lone trailing "\n" does not
    # count, since it sits beyond both the head and tail cut points for any
    # budget short enough to need cutting at all, so the snap collapses to
    # the same failure by a different path: head_end lands at 0 (no earlier
    # break to snap back to) and tail_start is never snapped forward either,
    # left exactly where the raw offset put it -- mid-word. That turned
    # "before tax" into "ore tax": a fragment that reads as a typo, not a
    # recognisable omission. There is no multi-line "# ... N lines omitted
    # ..." convention to fall back on for a single line either, so refuse
    # the cut entirely rather than splice mid-word. This can only happen for
    # code (comment_prefix set): everywhere else, a single unbroken line --
    # a long paragraph, a JSON string value -- is exactly the case
    # truncate_middle exists to shorten, and already has its own tested
    # behaviour for it.
    if config.comment_prefix and "\n" not in text.rstrip("\n"):
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
        #
        # But a snap to exactly 0 means the "line" it kept is empty -- a
        # blank separator before the real content, not a boundary worth
        # stopping at. For code (comment_prefix set) that is fine and
        # already relied on: it produces "def foo():\n    # ... N lines
        # omitted ...\n", a fully-explained omission. Plain text has no such
        # comment to say what happened, so keeping literally nothing as head
        # silently discarded a whole first line the budget had room for --
        # measured on a five-line paragraph with real budget to spare, this
        # is what turned a paragraph naming the bug, the ticket and the
        # exact failure into just its last clause. Treat it like no
        # backward boundary was found at all, and look forward for a real
        # line instead.
        snapped_head = text.rfind("\n", 0, head_end)
        if snapped_head > 0 or (snapped_head == 0 and config.comment_prefix):
            head_end = snapped_head + 1
        else:
            # The budget does not reach the first line break, so there is no
            # whole line behind the cut to fall back to. Take the first line
            # when it still fits the budget, otherwise keep no head at all.
            # Cutting here instead is the one option that yields neither valid
            # text nor a recognisable omission -- it produced
            # "self.repo = rep ..." mid-identifier.
            forward = text.find("\n", head_end)
            head_end = forward + 1 if 0 <= forward + 1 <= budget else 0
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

        # A head of zero means the first whole line did not fit the budget
        # at all -- correct to refuse rather than cut mid-word, per the
        # forward-search fallback above. For code that is still a good
        # outcome: paired with `comment_prefix` it reads as "def foo():\n
        # # ... N lines omitted ...\n", a fully-explained omission. Plain
        # text has no such comment, so a bare " ... " at the very start
        # says nothing about what vanished -- measured on a short list item
        # whose one wrapped line was longer than its own budget, this threw
        # away the entire item with only a trailing clause left to show for
        # it. Refusing the cut here costs this one span its savings, not
        # its content.
        if head_end == 0 and not config.comment_prefix:
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

    # Code only: whole lines are not enough where indentation carries meaning.
    if config.comment_prefix and config.snap_to_lines and "\n" in text:
        head_end, tail_start = _block_safe_cuts(
            text, head_end, tail_start, config.comment_prefix
        )
        if tail_start <= head_end:
            return text

    head = trim_dangling_escape(text[:head_end])
    tail = text[tail_start:]

    candidate = head + _omission_marker(text, head, head_end, tail_start, config) + tail
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
