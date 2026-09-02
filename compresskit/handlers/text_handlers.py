"""Structure handlers for line-oriented content: diffs, logs, Markdown.

Upstream Headroom routes these content types to learned or Rust-backed
compressors. Here they get rule-based handlers, for the same reason the
detector is rule-based: each of these formats carries its structure in
characters at the start of a line, which a regex reads perfectly well.

Like every handler in this package these only produce a mask. They decide what
must survive; :mod:`compresskit.reducers` decides what happens to the rest.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from ..masks import StructureMask
from .base import BaseStructureHandler, HandlerResult


def _iter_lines(content: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, line)`` for each line, including its newline.

    ``end`` is exclusive and includes the trailing newline when present, so the
    yielded spans tile ``content`` exactly and can be written into a
    character-level mask without gaps.
    """
    start = 0
    length = len(content)
    while start < length:
        newline = content.find("\n", start)
        end = length if newline == -1 else newline + 1
        yield start, end, content[start:end]
        start = end


class DiffStructureHandler(BaseStructureHandler):
    """Handler for unified diffs and git patches.

    Preserves:
    - File and hunk headers (``diff --git``, ``index``, ``---``, ``+++``,
      ``@@``, rename/mode/binary notices).
    - Every added (``+``) and removed (``-``) line. These *are* the diff; a
      patch whose changed lines were compressed would be worse than useless.

    Marks as compressible:
    - Context lines (leading space) and anything else. Context is recoverable
      from the file being patched, so it is the cheapest thing to lose.

    Example:
        >>> handler = DiffStructureHandler()
        >>> result = handler.get_mask("@@ -1 +1 @@\\n-old\\n+new\\n context\\n")
        >>> result.metadata["change_lines"]
        2
    """

    #: Line prefixes that mark diff metadata rather than content.
    _HEADER_PREFIXES = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "@@",
        "new file mode",
        "deleted file mode",
        "old mode",
        "new mode",
        "similarity index",
        "rename from",
        "rename to",
        "copy from",
        "copy to",
        "Binary files ",
        "GIT binary patch",
        "\\ No newline at end of file",
    )

    def __init__(self, preserve_context: bool = False):
        """Initialize the diff handler.

        Args:
            preserve_context: Preserve context lines too. Turns the handler
                into a near no-op on compression ratio; useful when the patch
                must stay applicable.
        """
        super().__init__(name="diff")
        self.preserve_context = preserve_context

    def can_handle(self, content: str) -> bool:
        """Check whether the content looks like a unified diff."""
        return any(
            line.startswith(("@@", "diff --git ", "--- ", "+++ "))
            for line in content.split("\n", 200)[:200]
        )

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Mark headers and changed lines as structural."""
        mask = [False] * len(content)
        header_lines = 0
        change_lines = 0

        for start, end, line in _iter_lines(content):
            stripped = line.rstrip("\n")
            is_header = stripped.startswith(self._HEADER_PREFIXES)
            # A changed line, but not a "---"/"+++" file header, which the
            # header check above has already claimed.
            is_change = not is_header and stripped[:1] in ("+", "-")

            if is_header:
                header_lines += 1
            if is_change:
                change_lines += 1

            if is_header or is_change or (self.preserve_context and stripped[:1] == " "):
                mask[start:end] = [True] * (end - start)

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.9,
            metadata={
                "header_lines": header_lines,
                "change_lines": change_lines,
            },
        )


class LogStructureHandler(BaseStructureHandler):
    """Handler for application logs.

    Preserves:
    - Whole lines at or above ``severity_floor`` (WARN by default). These are
      why anyone reads the log.
    - The prefix of every other line up to and including its level token, so
      timestamps, loggers and ordering survive even where the message does not.
    - Stack-trace continuation lines belonging to a preserved line, since a
      traceback split from its error is unreadable.

    Marks as compressible:
    - The message body of INFO/DEBUG/TRACE lines.

    Example:
        >>> handler = LogStructureHandler()
        >>> result = handler.get_mask("2024-01-01 INFO  started\\n2024-01-01 ERROR boom\\n")
        >>> result.metadata["preserved_lines"]
        1
    """

    _SEVERITY_ORDER = {
        "TRACE": 0,
        "DEBUG": 1,
        "INFO": 2,
        "NOTICE": 3,
        "WARN": 4,
        "WARNING": 4,
        "ERROR": 5,
        "FATAL": 6,
        "CRITICAL": 6,
    }

    _LEVEL_RE = re.compile(
        r"\[?\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL)\b\]?"
    )

    # A continuation line: stack-trace frames and wrapped messages, which are
    # indented or start with a Java/Python frame marker.
    _CONTINUATION_RE = re.compile(r"^(?:[ \t]+|\tat |Caused by:|\s*File \")")

    def __init__(
        self,
        severity_floor: str = "WARN",
        prefix_chars: int = 48,
    ):
        """Initialize the log handler.

        Args:
            severity_floor: Lines at this level or above are preserved whole.
            prefix_chars: How many characters of a non-preserved line to keep
                when it has no recognizable level token.
        """
        super().__init__(name="log")
        self.severity_floor = self._SEVERITY_ORDER.get(severity_floor.upper(), 4)
        self.prefix_chars = prefix_chars

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Mark high-severity lines and per-line prefixes as structural."""
        mask = [False] * len(content)
        # Characters that must stay compressible whatever entropy thinks.
        forced = [False] * len(content)
        preserved_lines = 0
        total_lines = 0
        repeated_lines = 0
        # Whether the previous line was preserved whole, so its indented
        # continuation lines (stack frames) are preserved too.
        in_preserved_block = False

        # Message text of the previous line, to spot repeats. A build that
        # emits the same DeprecationWarning on every import produces hundreds
        # of identical lines, and they are the single largest thing a log
        # compressor can remove.
        previous_message: str | None = None

        for start, end, line in _iter_lines(content):
            total_lines += 1
            stripped = line.rstrip("\n")

            match = self._LEVEL_RE.search(stripped)
            severity = self._SEVERITY_ORDER[match.group(1)] if match else None

            # A repeat of the line before it: claim nothing, not even the
            # timestamp prefix. Preserving prefixes here is what stopped the
            # reducer's line dedup from ever firing on a log -- it saw
            # alternating preserved and compressible fragments instead of a
            # run of identical lines, so 180 copies of a warning survived as
            # 180 copies. Leaving the whole run unclaimed lets dedup collapse
            # it to one line plus a count.
            # Severity is deliberately not consulted. The first occurrence is
            # preserved by whatever rule its level earns; a verbatim repeat of
            # it carries no further information, and 180 identical WARNINGs
            # sat exactly at the severity floor -- so the "preserve WARN and
            # above" rule kept every one of them.
            message = stripped[match.end():] if match else None
            if (
                message is not None
                and message.strip()
                and message == previous_message
            ):
                repeated_lines += 1
                forced[start:end] = [True] * (end - start)
                in_preserved_block = False
                continue
            previous_message = message

            if severity is not None and severity >= self.severity_floor:
                mask[start:end] = [True] * (end - start)
                preserved_lines += 1
                in_preserved_block = True
                continue

            # A continuation line belongs to the line above it. If that line
            # was preserved, keep the whole frame; if it was reduced, drop the
            # continuation with it rather than keeping an arbitrary prefix of
            # a stack frame that no longer has an error attached.
            if severity is None and self._CONTINUATION_RE.match(stripped):
                if in_preserved_block:
                    mask[start:end] = [True] * (end - start)
                    preserved_lines += 1
                continue

            in_preserved_block = False

            # Keep the prefix: through the level token if there is one, else a
            # fixed number of characters (usually enough for a timestamp).
            prefix_end = match.end() if match else min(len(stripped), self.prefix_chars)
            mask[start : start + prefix_end] = [True] * prefix_end

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.8,
            force_compressible=forced if repeated_lines else None,
            metadata={
                "total_lines": total_lines,
                "preserved_lines": preserved_lines,
                "repeated_lines": repeated_lines,
            },
        )


class MarkdownStructureHandler(BaseStructureHandler):
    """Handler for Markdown and other lightweight markup.

    Preserves:
    - Heading lines, which are the document's table of contents.
    - Code-fence delimiter lines, so fenced regions stay balanced and the
      language tag survives.
    - List bullets and blockquote markers -- the marker only, not the item text
      -- so the shape of the document is still legible.
    - Table header and separator rows.

    Marks as compressible:
    - Paragraph text and list item bodies.

    Example:
        >>> handler = MarkdownStructureHandler()
        >>> result = handler.get_mask("# Title\\n\\nSome prose here.\\n")
        >>> result.metadata["headings"]
        1
    """

    _HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+\S")
    _SETEXT_RE = re.compile(r"^[ \t]*(?:=+|-{2,})[ \t]*$")
    _FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~)")
    _LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
    _QUOTE_MARKER_RE = re.compile(r"^[ \t]*>+[ \t]*")
    _TABLE_SEPARATOR_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?")

    def __init__(self, preserve_fenced_code: bool = True):
        """Initialize the Markdown handler.

        Args:
            preserve_fenced_code: Preserve the contents of fenced code blocks,
                not just the fences. Code inside prose is usually the part a
                reader cannot reconstruct.
        """
        super().__init__(name="markdown")
        self.preserve_fenced_code = preserve_fenced_code

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Mark headings, fences, markers and table rows as structural."""
        mask = [False] * len(content)
        headings = 0
        in_fence = False

        for start, end, line in _iter_lines(content):
            stripped = line.rstrip("\n")

            if self._FENCE_RE.match(stripped):
                mask[start:end] = [True] * (end - start)
                in_fence = not in_fence
                continue

            if in_fence:
                if self.preserve_fenced_code:
                    mask[start:end] = [True] * (end - start)
                continue

            if self._HEADING_RE.match(stripped) or self._SETEXT_RE.match(stripped):
                mask[start:end] = [True] * (end - start)
                headings += 1
                continue

            if self._TABLE_SEPARATOR_RE.match(stripped) and "-" in stripped:
                mask[start:end] = [True] * (end - start)
                continue

            # Markers only: the bullet or quote character keeps the document's
            # shape without preserving the item text.
            for pattern in (self._QUOTE_MARKER_RE, self._LIST_MARKER_RE):
                marker = pattern.match(stripped)
                if marker:
                    mask[start : start + marker.end()] = [True] * marker.end()
                    break

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.85,
            metadata={"headings": headings, "unbalanced_fence": in_fence},
        )
