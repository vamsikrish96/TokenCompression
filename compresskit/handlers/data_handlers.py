"""Structure handlers for tabular data and configuration files.

Both formats share a shape: a small amount of text names things, and a much
larger amount of text is the values being named. A CSV's header row tells you
what every column means; a config file's keys tell you what every setting is.
Lose those and the rest is unreadable. Lose some values and you still know what
you are looking at.

So both handlers preserve the naming and reduce the named, which is the same
bargain :class:`~compresskit.handlers.json_handler.JSONStructureHandler` makes.
"""

from __future__ import annotations

import re
from typing import Any

from ..masks import EntropyScore, StructureMask
from .base import BaseStructureHandler, HandlerResult
from .text_handlers import _iter_lines


class TabularStructureHandler(BaseStructureHandler):
    """Handler for CSV, TSV and other delimiter-separated data.

    Preserves:
    - The header row in full. Every value below it is meaningless without it.
    - Every delimiter, so the column count stays readable and the rows stay
      alignable.
    - The first few data rows in full, so the model can see what a populated
      record looks like.
    - Short cells, numbers, and high-entropy cells (ids, references, hashes).

    Marks as compressible:
    - Long free-text cells in later rows, which is where the bulk lives.

    Example:
        >>> handler = TabularStructureHandler()
        >>> result = handler.get_mask("id,name\\n1,Alice\\n2,Bob\\n")
        >>> result.metadata["columns"]
        2
    """

    def __init__(
        self,
        preserve_header: bool = True,
        full_rows: int = 3,
        short_cell_threshold: int = 16,
        entropy_threshold: float = 0.85,
    ):
        """Initialize the tabular handler.

        Args:
            preserve_header: Keep the first row verbatim.
            full_rows: Data rows after the header to keep verbatim, so the
                model sees complete examples before the reduced ones.
            short_cell_threshold: Cells this length or shorter are kept.
                Reducing them costs more in marker text than it saves.
            entropy_threshold: Cells above this normalized entropy are kept,
                catching ids and reference codes that cannot be reconstructed.
        """
        super().__init__(name="tabular")
        self.preserve_header = preserve_header
        self.full_rows = full_rows
        self.short_cell_threshold = short_cell_threshold
        self.entropy_threshold = entropy_threshold

    @staticmethod
    def _detect_separator(lines: list[str]) -> str:
        """Pick the delimiter whose count is most consistent across rows."""
        best, best_score = ",", 0.0
        for candidate in (",", "\t", ";", "|"):
            counts = [line.count(candidate) for line in lines if line.strip()]
            populated = [c for c in counts if c >= 1]
            if len(populated) < 2:
                continue
            modal = max(set(populated), key=populated.count)
            score = sum(1 for c in counts if c == modal) * modal
            if score > best_score:
                best, best_score = candidate, score
        return best

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Mark the header, delimiters and short or high-entropy cells."""
        mask = [False] * len(content)
        lines = content.split("\n")
        separator = self._detect_separator(lines)

        row_index = 0
        columns = 0
        preserved_cells = 0

        for start, end, raw in _iter_lines(content):
            line = raw.rstrip("\n")
            if not line.strip():
                row_index += 1
                continue

            is_header = row_index == 0 and self.preserve_header
            keep_whole = is_header or row_index <= self.full_rows
            if is_header:
                columns = line.count(separator) + 1

            if keep_whole:
                mask[start:end] = [True] * (end - start)
                row_index += 1
                continue

            # Walk the cells, keeping the delimiters and deciding per cell.
            offset = start
            for cell in line.split(separator):
                cell_end = offset + len(cell)
                if self._keep_cell(cell):
                    mask[offset:cell_end] = [True] * len(cell)
                    preserved_cells += 1
                # The delimiter itself always survives, so the column count
                # stays legible even where the values do not.
                if cell_end < end:
                    mask[cell_end] = True
                offset = cell_end + len(separator)

            row_index += 1

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.9,
            metadata={
                "separator": separator,
                "columns": columns,
                "rows": row_index,
                "preserved_cells": preserved_cells,
            },
        )

    def _keep_cell(self, cell: str) -> bool:
        """Whether one cell survives verbatim."""
        value = cell.strip().strip('"')
        if not value:
            return True
        if len(value) <= self.short_cell_threshold:
            return True
        # An identifier has no spaces; prose does. Entropy alone rates a
        # diverse English word as highly as a reference code.
        if " " not in value and EntropyScore.compute(value, self.entropy_threshold).should_preserve:
            return True
        return False


class ConfigStructureHandler(BaseStructureHandler):
    """Handler for YAML, TOML, INI and .env configuration.

    Preserves:
    - Every key, its indentation, and its ``:`` or ``=``. The shape of a config
      file *is* its keys; a value without its key says nothing.
    - Section headers (``[section]``) and list markers.
    - Short values, which is most of them -- ports, booleans, levels, paths.
    - High-entropy values: tokens, secrets, connection strings.
    - Comments, optionally, since a config comment usually explains a choice
      that is not otherwise recoverable.

    Marks as compressible:
    - Long values only. In practice that is descriptions, embedded blobs and
      pasted certificates.

    Example:
        >>> handler = ConfigStructureHandler()
        >>> result = handler.get_mask("server:\\n  host: localhost\\n")
        >>> result.metadata["keys"]
        2
    """

    #: A key and its separator, keeping indentation. Group 1 is the whole
    #: prefix that must survive.
    _KEY_RE = re.compile(r"^([ \t]*[A-Za-z_][\w.\-]*[ \t]*[:=])")

    #: A TOML/INI section header, or a YAML document marker.
    _SECTION_RE = re.compile(r"^[ \t]*(\[[^\]\n]*\]|---|\.\.\.)[ \t]*$")

    #: A YAML sequence entry: the dash is structure, what follows is a value.
    _LIST_RE = re.compile(r"^([ \t]*-[ \t]+)")

    def __init__(
        self,
        preserve_comments: bool = True,
        short_value_threshold: int = 40,
        entropy_threshold: float = 0.85,
    ):
        """Initialize the config handler.

        Args:
            preserve_comments: Keep ``#`` comment lines. They usually explain
                a choice that the value alone does not.
            short_value_threshold: Values this length or shorter are kept.
                Most real settings are far below it.
            entropy_threshold: Values above this normalized entropy are kept,
                catching tokens and connection strings.
        """
        super().__init__(name="config")
        self.preserve_comments = preserve_comments
        self.short_value_threshold = short_value_threshold
        self.entropy_threshold = entropy_threshold

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Mark keys, sections and short or high-entropy values."""
        mask = [False] * len(content)
        keys = 0
        sections = 0

        for start, end, raw in _iter_lines(content):
            line = raw.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                if self.preserve_comments:
                    mask[start:end] = [True] * (end - start)
                continue

            if self._SECTION_RE.match(line):
                mask[start:end] = [True] * (end - start)
                sections += 1
                continue

            prefix = self._KEY_RE.match(line) or self._LIST_RE.match(line)
            if prefix is None:
                # A continuation of a multi-line value, or something we do not
                # recognise. Leave it compressible rather than guessing.
                continue

            width = prefix.end(1)
            mask[start : start + width] = [True] * width
            if self._KEY_RE.match(line):
                keys += 1

            value = line[width:]
            if self._keep_value(value):
                mask[start + width : end] = [True] * (end - start - width)

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.9,
            metadata={"keys": keys, "sections": sections},
        )

    def _keep_value(self, value: str) -> bool:
        """Whether a value survives verbatim alongside its key."""
        text = value.strip().strip("\"'")
        if not text:
            return True
        if len(text) <= self.short_value_threshold:
            return True
        if " " not in text and EntropyScore.compute(text, self.entropy_threshold).should_preserve:
            return True
        return False
