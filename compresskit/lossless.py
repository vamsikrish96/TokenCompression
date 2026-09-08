"""Reversible compaction: shorter text that means exactly the same thing.

Every other stage in this package is lossy. It shortens content and hopes the
answer survives, which is why a truncated ``"message"`` field or a shredded
instruction is always a possibility you have to check for.

This stage cannot lose anything. Each transform has an exact inverse, and
:func:`compact_lossless` verifies the round trip before returning: if the
inverse does not reproduce the input, or the result is not actually smaller,
you get your input back unchanged. It never raises.

What it does is rearrange, not encode. The output still looks like its own
format -- grep stays grep, logs stay logs, diffs stay diffs -- so a model reads
it exactly as it would read the original. Five tricks:

* Strip ANSI colour codes, which carry no meaning to a model.
* Collapse runs of identical consecutive lines, syslog style.
* Fold a repeated multi-line block into a back-reference.
* Regroup ``path:line:content`` grep rows under a single ``path:`` heading,
  instead of repeating the path on every row.
* Drop ``index <sha>..<sha>`` lines from a diff; the patch still applies.

Measured on realistic content: a colourised CI log with a repeated warning
burst compacts by 69%, grep output by 39%, a git diff by 25%. A log whose
messages are all distinct, or a JSON payload, compacts by 0% -- and that is
the correct answer, not a failure. Nothing repeats, so nothing folds.

Run this before the lossy pipeline, never after: folding repeats in text that
truncation has already mangled finds nothing.

One honest exception to "lossless": ANSI colour codes are dropped permanently.
The guarantee is about meaning, not bytes.

Ported from Headroom's ``headroom/transforms/lossless_compaction.py``
(Apache-2.0); see NOTICE. Unchanged apart from this docstring.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "strip_ansi",
    "collapse_runs",
    "expand_runs",
    "is_run_collapsed",
    "fold_repeated_blocks",
    "unfold_repeated_blocks",
    "search_heading",
    "search_unheading",
    "search_dir_heading",
    "search_dir_unheading",
    "diff_strip_index",
    "json_fold_constants",
    "json_unfold_constants",
    "tabular_fold_constants",
    "tabular_unfold_constants",
    "split_fold_note",
    "compact_lossless",
]

# ANSI CSI SGR (color/style) escape sequences: ESC [ ... m. Color is
# non-semantic, so stripping it is a safe (one-way) lossless-of-meaning op.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# syslog-style run-collapse marker. The count is captured for exact inversion.
_RUN_MARKER_RE = re.compile(r"^\.\.\. \(repeated (\d+) times\)$")

# multi-line block back-reference marker. Length and distance (both in lines,
# in ORIGINAL coordinates) are captured for exact inversion: everything before
# a marker expands to the exact original prefix, so `distance` lines back in
# the expanded output is the block's first occurrence.
_BLOCK_MARKER_RE = re.compile(r"^\.\.\. \(repeats (\d+) lines from (\d+) lines back\)$")

# fold_repeated_blocks search bounds: minimum/maximum block length worth a
# marker, candidate anchors per line, and an input size cap so the scan stays
# negligible on huge payloads.
_FOLD_MIN_BLOCK = 3
_FOLD_MAX_BLOCK = 64
_FOLD_MAX_CANDIDATES = 8
_FOLD_MAX_LINES = 20_000

# grep/ripgrep default row shape: ``path:line:content``. ``line`` is digits;
# ``path`` must not itself look like ``line:content`` (i.e. not start with a
# bare number) so we don't mis-split a heading-form ``line:content`` row.
_GREP_ROW_RE = re.compile(r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<content>.*)$")
# heading-form data row (``line:content``) produced by search_heading.
_HEADING_ROW_RE = re.compile(r"^(?P<line>\d+):(?P<content>.*)$")

# unified-diff ``index <sha>..<sha> <mode>`` line. The diff still applies
# without it (git only uses it for rename/blob bookkeeping).
_DIFF_INDEX_RE = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+( [0-7]+)?$")


def strip_ansi(text: str) -> str:
    """Remove ANSI CSI/SGR (color) escape sequences. Color is non-semantic."""
    return _ANSI_RE.sub("", text)


def _split_keep_trailing(text: str) -> tuple[list[str], bool]:
    """Split into lines, remembering whether a trailing newline was present.

    Returns (lines, had_trailing_newline). This lets the run helpers rejoin
    byte-exactly instead of always appending or always dropping a newline.
    """
    if text == "":
        return [], False
    had_trailing = text.endswith("\n")
    body = text[:-1] if had_trailing else text
    return body.split("\n"), had_trailing


def _join(lines: list[str], had_trailing: bool) -> str:
    out = "\n".join(lines)
    if had_trailing:
        out += "\n"
    return out


def collapse_runs(text: str) -> str:
    """Collapse runs of >=2 identical consecutive lines (syslog convention).

    A run of N (N>=2) identical lines becomes the line once followed by
    ``... (repeated N times)``. Exact inverse: :func:`expand_runs`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= 2:
            out.append(lines[i])
            out.append(f"... (repeated {run_len} times)")
        else:
            out.append(lines[i])
        i = j + 1
    return _join(out, had_trailing)


def expand_runs(text: str) -> str:
    """Exact inverse of :func:`collapse_runs`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if i + 1 < n:
            m = _RUN_MARKER_RE.match(lines[i + 1])
            if m:
                count = int(m.group(1))
                out.extend([line] * count)
                i += 2
                continue
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def is_run_collapsed(text: str) -> bool:
    """True if any run-collapse marker line is present."""
    for line in text.split("\n"):
        if _RUN_MARKER_RE.match(line):
            return True
    return False


def fold_repeated_blocks(text: str) -> str:
    """Collapse multi-line blocks that repeat earlier content into back-refs.

    The block-level generalization of :func:`collapse_runs`: a run of K
    consecutive lines (K >= 3) that exactly reproduces K lines seen D lines
    earlier becomes ``... (repeats K lines from D lines back)``. The repeats
    need not be adjacent, which is what config payloads actually look like —
    k8s container stanzas repeat with only the ``name:`` line differing, so
    their identical tails fold even though no two whole stanzas are
    consecutive. Coordinates are in original lines: the fold is only taken
    when the block does not overlap its anchor (K <= D), so on expansion the
    referenced region is always already reconstructed.
    Exact inverse: :func:`unfold_repeated_blocks`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    n = len(lines)
    if n < _FOLD_MIN_BLOCK * 2 or n > _FOLD_MAX_LINES:
        return text
    positions: dict[str, list[int]] = {}
    out: list[str] = []
    i = 0
    while i < n:
        best_len = 0
        best_dist = 0
        for q in reversed(positions.get(lines[i], ())):
            max_len = min(_FOLD_MAX_BLOCK, n - i, i - q)
            length = 0
            while length < max_len and lines[q + length] == lines[i + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_dist = i - q
        if best_len >= _FOLD_MIN_BLOCK:
            marker = f"... (repeats {best_len} lines from {best_dist} lines back)"
            block_chars = sum(len(lines[i + k]) + 1 for k in range(best_len))
            if block_chars > len(marker) + 1:
                out.append(marker)
                for k in range(best_len):
                    _remember(positions, lines[i + k], i + k)
                i += best_len
                continue
        _remember(positions, lines[i], i)
        out.append(lines[i])
        i += 1
    return _join(out, had_trailing)


def _remember(positions: dict[str, list[int]], line: str, index: int) -> None:
    """Track recent original positions of `line`, bounded per distinct line."""
    bucket = positions.setdefault(line, [])
    bucket.append(index)
    if len(bucket) > _FOLD_MAX_CANDIDATES:
        del bucket[0]


def unfold_repeated_blocks(text: str) -> str:
    """Exact inverse of :func:`fold_repeated_blocks`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    for line in lines:
        m = _BLOCK_MARKER_RE.match(line)
        if m:
            length, dist = int(m.group(1)), int(m.group(2))
            start = len(out) - dist
            if start >= 0 and length <= dist:
                out.extend(out[start : start + length])
                continue
        out.append(line)
    return _join(out, had_trailing)


def search_heading(text: str) -> str:
    """Convert grep ``path:line:content`` rows into ripgrep --heading form.

    Consecutive rows sharing a path collapse to the path once on its own line
    (a *header* line), then ``line:content`` rows beneath it. Lines that don't
    match the ``path:line:content`` shape are passed through untouched. No
    blank separators are inserted (they would be ambiguous with passthrough
    content), keeping the transform exactly reversible via
    :func:`search_unheading`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m:
            path = m.group("path")
            if path != current_path:
                out.append(path)
                current_path = path
            out.append(f"{m.group('line')}:{m.group('content')}")
        else:
            # Any non-grep-row line ends the current file grouping.
            out.append(line)
            current_path = None
    return _join(out, had_trailing)


def search_unheading(text: str) -> str:
    """Exact inverse of :func:`search_heading`.

    A *header* line is any line that is not itself a ``line:content`` data row
    and is immediately followed by at least one ``line:content`` data row; it
    is consumed (not re-emitted) and its text becomes the ``path`` prefix for
    the data rows that follow, until a non-data line appears.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _HEADING_ROW_RE.match(line)
        if current_path is not None and data:
            out.append(f"{current_path}:{data.group('line')}:{data.group('content')}")
            i += 1
            continue
        # Not a data row under an active header. Decide if THIS line is a new
        # header: it must not be a data row itself and must be followed by a
        # data row. If so, consume it as the path prefix (do not emit).
        if not data and i + 1 < n and _HEADING_ROW_RE.match(lines[i + 1]):
            current_path = line
            i += 1
            continue
        # Plain passthrough line (or a stray data row with no header): emit it
        # verbatim and clear any active grouping.
        current_path = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


# A dir-heading data row: ``<base>:<line>:<content>`` where base has no '/'.
_DIR_DATA_RE = re.compile(r"^(?P<base>[^/\n:]+):(?P<line>\d+):(?P<content>.*)$")


def search_dir_heading(text: str) -> str:
    """Fold grep ``path:line:content`` rows by DIRECTORY.

    Consecutive rows whose path shares a parent directory collapse to that
    directory once (a header ending in ``/``), then ``base:line:content`` rows
    beneath it. Complements :func:`search_heading` (which factors a repeated
    *file*): this factors a repeated *directory* across distinct files — the
    common ``grep -rn`` case where each file has a single match, so file-heading
    saves nothing but the shared directory repeats on every row. Rows whose path
    has no ``/`` pass through untouched. Exactly reversed by
    :func:`search_dir_unheading`; ``compact_lossless`` verifies the round-trip.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m and "/" in m.group("path"):
            path = m.group("path")
            cut = path.rindex("/") + 1
            dir_part, base = path[:cut], path[cut:]
            if dir_part != current_dir:
                out.append(dir_part)
                current_dir = dir_part
            out.append(f"{base}:{m.group('line')}:{m.group('content')}")
        else:
            out.append(line)
            current_dir = None
    return _join(out, had_trailing)


def search_dir_unheading(text: str) -> str:
    """Exact inverse of :func:`search_dir_heading`.

    A *header* is a line ending in ``/`` immediately followed by a
    ``base:line:content`` data row; it is consumed and re-prefixed onto each
    following data row until a non-data line appears.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _DIR_DATA_RE.match(line)
        if current_dir is not None and data:
            out.append(f"{current_dir}{line}")
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and _DIR_DATA_RE.match(lines[i + 1]):
            current_dir = line
            i += 1
            continue
        current_dir = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def diff_strip_index(text: str) -> str:
    """Drop ``index <sha>..<sha>`` lines from a unified diff (still applies)."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out = [line for line in lines if not _DIFF_INDEX_RE.match(line)]
    return _join(out, had_trailing)


# A whole-line file path: optional ``./``/``../`` root, >=1 directory segment,
# then a basename. No whitespace or ':' (so grep ``path:line:content`` rows —
# handled by search_heading — are excluded). Directory-only lines (trailing '/')
# don't match (empty basename), which keeps the fold unambiguous.
_PATH_ROW_RE = re.compile(r"^(?P<dir>(?:\.{0,2}/)?(?:[^/\s:]+/)+)(?P<base>[^/\s:]+)$")


def path_heading(text: str) -> str:
    """Fold a *pure* file-path listing (``find`` / ``ls -1`` / ``rg -l`` output)
    into ripgrep-heading form: each parent directory printed once on its own
    line (ending in ``/``), then the bare basenames beneath it.

    Reversibility is not assumed here — ``compact_lossless`` verifies the exact
    round-trip via :func:`path_unheading` and discards the fold on any mismatch
    (e.g. a stray no-slash line mistaken for a basename), so mixed content is
    always safe. Requires >=2 path rows or there is nothing to group.
    Complements ``search_heading``, which only handles the ``path:line:content``
    grep shape, not plain path lists.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if sum(1 for ln in lines if _PATH_ROW_RE.match(ln)) < 2:
        return text
    out: list[str] = []
    current: str | None = None
    for line in lines:
        m = _PATH_ROW_RE.match(line)
        if m:
            d = m.group("dir")
            if d != current:
                out.append(d)
                current = d
            out.append(m.group("base"))
        else:  # blank line inside/around the listing
            out.append(line)
            current = None
    return _join(out, had_trailing)


def path_unheading(text: str) -> str:
    """Exact inverse of :func:`path_heading`.

    A *header* is a line ending in ``/`` immediately followed by a basename row
    (a non-empty line with no ``/``); it is consumed and re-prefixed onto each
    following basename row until a blank line or another header.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        is_base = line != "" and "/" not in line
        if current is not None and is_base:
            out.append(current + line)
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and lines[i + 1] != "" and "/" not in lines[i + 1]:
            current = line
            i += 1
            continue
        current = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


# ---------------------------------------------------------------------------
# Cross-record constant folding
#
# Every fold above works on repeated *lines*. This one works on repeated
# *fields*: 60 records that each carry "role": "member" hold that fact 60
# times, and no line-level or span-level trick can see it -- the repeats are
# never adjacent and never long enough to truncate. Stating the fact once and
# removing it from every record is the only way to reach it.
#
# The removal changes the payload's shape, which is safe when a model reads the
# data to answer a question and unsafe when the payload is the template for the
# model's own output ("return me this JSON with X changed"). Nothing here can
# tell those apart, so the caller decides: see CompressorConfig.
# ---------------------------------------------------------------------------

#: Below this many records the note costs more than the fold saves.
_FOLD_MIN_RECORDS = 10

#: The machine-readable half of the note. Everything the inverse needs is on
#: this one line; the prose above it is for the model.
_JSON_RESTORE_PREFIX = "// compresskit-restore: "

_WHITESPACE = " \t\n\r"


class _JsonShapeError(ValueError):
    """The document is not a shape this fold can safely rewrite."""


def _json_ws(text: str, index: int) -> int:
    """Index of the next non-whitespace character at or after ``index``."""
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def _json_string_end(text: str, index: int) -> int:
    """Index just past the string literal starting at ``index``."""
    index += 1  # opening quote
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    raise _JsonShapeError("unterminated string")


def _json_value_end(text: str, index: int, arrays: list) -> int:
    """Index just past the value starting at ``index``.

    Arrays met along the way are appended to ``arrays`` as
    ``(start, end, elements)``, so one scan finds every candidate record set
    rather than re-walking the document once per array.
    """
    char = text[index]
    if char == '"':
        return _json_string_end(text, index)
    if char == "{":
        return _json_object(text, index, arrays)[1]
    if char == "[":
        return _json_array(text, index, arrays)
    scan = index
    while scan < len(text) and text[scan] not in ",]}" + _WHITESPACE:
        scan += 1
    if scan == index:
        raise _JsonShapeError("empty value")
    return scan


def _json_object(text: str, index: int, arrays: list) -> tuple[list, int]:
    """Members of the object at ``index``, and the index just past its ``}``.

    A member is ``(key_start, val_end)``; the raw slice between them is the
    member verbatim, which is what the fold splices. Working in raw offsets
    rather than parsed values is deliberate -- ``json.dumps`` will not
    reproduce the input's indentation, key order or spacing, and the round trip
    here is checked byte for byte.
    """
    members: list[tuple[int, int]] = []
    index = _json_ws(text, index + 1)
    if index < len(text) and text[index] == "}":
        return members, index + 1

    while True:
        index = _json_ws(text, index)
        if index >= len(text) or text[index] != '"':
            raise _JsonShapeError("object key is not a string")
        key_start = index
        index = _json_ws(text, _json_string_end(text, index))
        if index >= len(text) or text[index] != ":":
            raise _JsonShapeError("missing colon")
        index = _json_ws(text, index + 1)
        val_end = _json_value_end(text, index, arrays)
        members.append((key_start, val_end))

        index = _json_ws(text, val_end)
        if index >= len(text):
            raise _JsonShapeError("truncated object")
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            return members, index + 1
        raise _JsonShapeError("bad object separator")


def _json_array(text: str, index: int, arrays: list) -> int:
    """Index just past the array at ``index``; records it in ``arrays``."""
    start = index
    elements: list[tuple[int, int, list | None]] = []
    index = _json_ws(text, index + 1)
    if index < len(text) and text[index] == "]":
        arrays.append((start, index + 1, elements))
        return index + 1

    while True:
        index = _json_ws(text, index)
        element_start = index
        if text[index] == "{":
            members, element_end = _json_object(text, index, arrays)
            elements.append((element_start, element_end, members))
        else:
            element_end = _json_value_end(text, index, arrays)
            elements.append((element_start, element_end, None))

        index = _json_ws(text, element_end)
        if index >= len(text):
            raise _JsonShapeError("truncated array")
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "]":
            arrays.append((start, index + 1, elements))
            return index + 1
        raise _JsonShapeError("bad array separator")


def _json_pick_records(text: str) -> tuple[int, int, list] | None:
    """The largest array of objects in ``text``, or None if there is no fold.

    Chosen by element *count*, not text length: folding removes characters but
    never elements, so the same array wins before and after. The inverse relies
    on that -- it has to find the same array again in the folded text.
    """
    arrays: list = []
    start = _json_ws(text, 0)
    if start >= len(text):
        return None
    _json_value_end(text, start, arrays)

    best = None
    for array_start, array_end, elements in arrays:
        if len(elements) < _FOLD_MIN_RECORDS:
            continue
        if any(members is None for _, _, members in elements):
            continue  # not a record set
        if any(not members for _, _, members in elements):
            continue  # an empty object has nothing to fold
        if best is None or len(elements) > len(best[2]):
            best = (array_start, array_end, elements)
    return best


def _json_layout(text: str, elements: list) -> tuple[str, list[list[str]]]:
    """The one separator every record uses, and each record's raw members.

    Raises if records disagree on member count or on the exact text between
    members. Reconstruction joins with a single separator, so a document
    hand-formatted enough to vary is refused rather than silently reflowed.
    """
    separators: set[str] = set()
    records: list[list[str]] = []

    for _, _, members in elements:
        records.append([text[start:end] for start, end in members])
        for position in range(len(members) - 1):
            separators.add(text[members[position][1] : members[position + 1][0]])

    if len(separators) != 1:
        raise _JsonShapeError("records are not uniformly formatted")
    if len({len(record) for record in records}) != 1:
        raise _JsonShapeError("records have different member counts")
    return separators.pop(), records


def _json_member_key(member: str) -> str:
    """The raw key text of a ``"key": value`` member."""
    return member.split(":", 1)[0].strip()


def json_fold_constants(text: str) -> str:
    """Lift fields that never vary out of a homogeneous record array.

    The fields are stated once in a leading comment and removed from every
    record. Returns ``text`` unchanged whenever the document is not a shape
    this can rewrite exactly -- that is the common case, and it is not a
    failure.

    Example:
        >>> folded = json_fold_constants(payload)   # doctest: +SKIP
        >>> json_unfold_constants(folded) == payload
        True
    """
    if _JSON_RESTORE_PREFIX in text:
        return text  # already folded
    try:
        picked = _json_pick_records(text)
        if picked is None:
            return text
        array_start, _, elements = picked
        separator, records = _json_layout(text, elements)

        width = len(records[0])
        keys = [_json_member_key(member) for member in records[0]]
        for record in records:
            if [_json_member_key(member) for member in record] != keys:
                raise _JsonShapeError("records disagree on key order")

        constant = [
            position
            for position in range(width)
            if len({record[position] for record in records}) == 1
        ]
        # Folding every field would leave empty records with nothing to key the
        # restored values against, and nothing for the model to read.
        if not constant or len(constant) >= width:
            return text

        dropped = set(constant)
        keep = [position for position in range(width) if position not in dropped]

        pieces: list[str] = []
        cursor = array_start
        for element_start, element_end, members in elements:
            pieces.append(text[cursor:element_start])
            prefix = text[element_start : members[0][0]]
            tail = text[members[-1][1] : element_end]
            body = separator.join(
                text[members[position][0] : members[position][1]] for position in keep
            )
            pieces.append(prefix + body + tail)
            cursor = element_end
        pieces.append(text[cursor:])

        fields = [[position, records[0][position]] for position in constant]
        folded = text[:array_start] + "".join(pieces)
        return _json_note(len(elements), separator, fields) + folded
    except (_JsonShapeError, ValueError, IndexError):
        return text


def _json_note(count: int, separator: str, fields: list) -> str:
    """The comment block that carries the folded fields.

    Two audiences, one block. The prose tells the model to apply the fields to
    every record -- without that instruction it reads the note and then answers
    from the visible fields alone. The ``compresskit-restore`` line is what
    :func:`json_unfold_constants` reads: one line of JSON, so restoring never
    has to parse English.
    """
    listed = ", ".join(field[1] for field in fields)
    payload = json.dumps({"sep": separator, "fields": fields}, separators=(",", ":"))
    return (
        f"// compresskit: every object in the largest array below also has {listed}.\n"
        f"// Identical in all {count} entries, removed here to save space -- treat\n"
        f"// every entry as if it still contained them.\n"
        f"{_JSON_RESTORE_PREFIX}{payload}\n"
    )


def json_unfold_constants(text: str) -> str:
    """Exact inverse of :func:`json_fold_constants`."""
    if not text.startswith("// compresskit"):
        return text
    try:
        lines = text.split("\n")
        body = None
        for position, line in enumerate(lines):
            if line.startswith(_JSON_RESTORE_PREFIX):
                payload = json.loads(line[len(_JSON_RESTORE_PREFIX) :])
                body = "\n".join(lines[position + 1 :])
                break
        if body is None:
            return text

        separator = payload["sep"]
        fields = {int(index): raw for index, raw in payload["fields"]}

        picked = _json_pick_records(body)
        if picked is None:
            return text
        array_start, _, elements = picked

        pieces: list[str] = []
        cursor = array_start
        for element_start, element_end, members in elements:
            pieces.append(body[cursor:element_start])
            prefix = body[element_start : members[0][0]]
            tail = body[members[-1][1] : element_end]
            kept = iter(body[start:end] for start, end in members)
            restored = [
                fields[position] if position in fields else next(kept)
                for position in range(len(members) + len(fields))
            ]
            pieces.append(prefix + separator.join(restored) + tail)
            cursor = element_end
        pieces.append(body[cursor:])

        return body[:array_start] + "".join(pieces)
    except (_JsonShapeError, ValueError, KeyError, IndexError, StopIteration):
        return text


_TABULAR_RESTORE_PREFIX = "# compresskit-restore: "


def _tabular_rows(lines: list[str], separator: str) -> list[int] | None:
    """Indices of the lines that are rows, or None if the shape is irregular.

    Blank lines are not rows and are passed through untouched. Every row must
    carry the same field count as the header; a ragged file is refused rather
    than folded against a column that is not there.
    """
    rows = [index for index, line in enumerate(lines) if line.strip()]
    if len(rows) < _FOLD_MIN_RECORDS + 1:  # header plus the record floor
        return None
    width = lines[rows[0]].count(separator)
    if width < 1:
        return None
    if any(lines[index].count(separator) != width for index in rows):
        return None
    return rows


def tabular_fold_constants(text: str) -> str:
    """Lift columns whose value never varies out of delimited rows.

    The columns are stated once in a leading comment and removed from the
    header and every row. Returns ``text`` unchanged whenever the shape is
    irregular or nothing is constant.
    """
    if _TABULAR_RESTORE_PREFIX in text:
        return text  # already folded
    # A quoted field may contain the separator, which would make naive
    # splitting silently misalign columns -- and misalignment still round
    # trips, so verification would not catch it. Refuse the whole file.
    if '"' in text:
        return text
    try:
        # The tabular handler already sniffs delimiters by modal consistency;
        # imported here rather than at module scope to keep this layer free of
        # any dependency on the handler package.
        from .handlers.data_handlers import TabularStructureHandler

        lines, had_trailing = _split_keep_trailing(text)
        separator = TabularStructureHandler._detect_separator(lines)
        rows = _tabular_rows(lines, separator)
        if rows is None:
            return text

        header = lines[rows[0]].split(separator)
        data = [lines[index].split(separator) for index in rows[1:]]
        width = len(header)

        constant = [
            position
            for position in range(width)
            if len({row[position] for row in data}) == 1
        ]
        if not constant or len(constant) >= width:
            return text

        dropped = set(constant)
        keep = [position for position in range(width) if position not in dropped]
        fields = [[position, header[position], data[0][position]] for position in constant]

        folded = list(lines)
        for index in rows:
            cells = lines[index].split(separator)
            folded[index] = separator.join(cells[position] for position in keep)

        note = _tabular_note(len(data), separator, fields)
        return note + _join(folded, had_trailing)
    except (ValueError, IndexError):
        return text


def _tabular_note(count: int, separator: str, fields: list) -> str:
    """The comment block that carries the folded columns."""
    listed = ", ".join(f"{name}={value}" for _, name, value in fields)
    payload = json.dumps({"sep": separator, "fields": fields}, separators=(",", ":"))
    return (
        f"# compresskit: every row below also has {listed}. Identical in all\n"
        f"# {count} rows, removed here to save space -- treat every row as if it\n"
        f"# still contained them.\n"
        f"{_TABULAR_RESTORE_PREFIX}{payload}\n"
    )


def tabular_unfold_constants(text: str) -> str:
    """Exact inverse of :func:`tabular_fold_constants`."""
    if not text.startswith("# compresskit"):
        return text
    try:
        head, _, body = text.partition(_TABULAR_RESTORE_PREFIX)
        if not body:
            return text
        payload_line, _, remainder = body.partition("\n")
        payload = json.loads(payload_line)
        del head

        separator = payload["sep"]
        header_by_position = {int(p): name for p, name, _ in payload["fields"]}
        value_by_position = {int(p): value for p, _, value in payload["fields"]}

        lines, had_trailing = _split_keep_trailing(remainder)
        rows = [index for index, line in enumerate(lines) if line.strip()]
        if not rows:
            return text

        restored = list(lines)
        for order, index in enumerate(rows):
            kept = iter(lines[index].split(separator))
            source = header_by_position if order == 0 else value_by_position
            restored[index] = separator.join(
                source[position] if position in source else next(kept)
                for position in range(
                    len(lines[index].split(separator)) + len(source)
                )
            )
        return _join(restored, had_trailing)
    except (ValueError, KeyError, IndexError, StopIteration):
        return text


def split_fold_note(text: str) -> tuple[str, str]:
    """Separate a leading fold note from the payload it describes.

    The note is an instruction to the model, not data, and the lossy pass
    downstream cannot tell the difference -- left in place it gets truncated
    like any other run of characters, and ``treat every entry as if it still
    contained them`` becomes ``treat every ... them``. A fold whose note has
    been shredded is strictly worse than no fold at all, so the note is lifted
    out before reduction and put back afterwards.

    Args:
        text: Content that may begin with a fold note.

    Returns:
        ``(note, body)``. The note keeps its trailing newline and is ``""``
        when there is none, so ``note + body == text`` always holds.
    """
    for prefix in (_JSON_RESTORE_PREFIX, _TABULAR_RESTORE_PREFIX):
        marker = prefix.split(" ", 1)[0]  # "//" or "#"
        if not text.startswith(marker + " compresskit"):
            continue
        start = text.find(prefix)
        if start == -1:
            continue
        end = text.find("\n", start)
        if end == -1:
            continue
        return text[: end + 1], text[end + 1 :]
    return "", text


def _smaller(candidate: str, original: str) -> bool:
    return len(candidate) < len(original)


def compact_lossless(content: str, kind: str) -> str:
    """Dispatch format-native lossless compaction by ``kind``.

    ``kind`` in {'log', 'search', 'paths', 'diff', 'text', 'config',
    'json_records', 'tabular_columns'}. For reversible kinds the
    round-trip is verified internally (modulo the intentionally-dropped
    non-semantic bits, e.g. ANSI color for logs); if verification fails or the
    result is not smaller, the original content is returned unchanged. Never
    raises; unknown kinds pass through.
    """
    if not content:
        return content
    try:
        if kind == "log":
            # ANSI is non-semantic and dropped one-way; run-collapse must be
            # exactly reversible against the de-ANSI'd baseline.
            baseline = strip_ansi(content)
            candidate = collapse_runs(baseline)
            if expand_runs(candidate) != baseline:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "search":
            # Two independent folds; keep the smaller that round-trips exactly.
            # search_heading factors a repeated FILE (many matches in one file);
            # search_dir_heading factors a repeated DIRECTORY (one match each
            # across many files in a dir — the grep -rn case the file fold misses).
            best = content
            for candidate, inverse in (
                (search_heading(content), search_unheading),
                (search_dir_heading(content), search_dir_unheading),
            ):
                if inverse(candidate) == content and _smaller(candidate, best):
                    best = candidate
            return best

        if kind == "paths":
            # Pure path listings (find/ls -1/rg -l): fold repeated parent dirs.
            candidate = path_heading(content)
            if path_unheading(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "json_records":
            # Cross-record constant folding. Unlike the folds above this one
            # changes the payload's shape, so the caller opts in; see
            # CompressorConfig.fold_constant_fields.
            candidate = json_fold_constants(content)
            if json_unfold_constants(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "tabular_columns":
            candidate = tabular_fold_constants(content)
            if tabular_unfold_constants(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "diff":
            # Purely subtractive of non-semantic bookkeeping lines; the
            # remaining hunks still apply. No exact inverse needed.
            candidate = diff_strip_index(content)
            return candidate if _smaller(candidate, content) else content

        if kind == "text":
            # Collapse blank-line runs; reversible against itself.
            candidate = collapse_runs(content)
            if expand_runs(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "config":
            # Structured config (YAML/TOML/INI): single-line runs first, then
            # repeated multi-line stanzas. Inverse applies in reverse order.
            candidate = fold_repeated_blocks(collapse_runs(content))
            if expand_runs(unfold_repeated_blocks(candidate)) != content:
                return content
            return candidate if _smaller(candidate, content) else content
    except Exception:
        return content
    return content
