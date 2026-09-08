"""Embedding-driven relevance filtering for line- and paragraph-oriented content.

Every other reduction in this package decides what to keep from a span's
*shape* -- its role, its severity token, its position relative to a bracket.
None of that can tell a log line that happens to be short and unimportant
apart from a log line that happens to be short and load-bearing; shape alone
does not carry meaning. This module adds one more signal, orthogonal to all
of those: semantic similarity to an anchor.

Three filters, one shared mechanism, one difference between them -- where the
anchor comes from:

* :func:`filter_log_by_relevance` -- the anchor is built from the log's own
  severity-floor material (the error and its traceback). Self-contained,
  nothing to supply.
* :func:`filter_search_by_relevance` -- grep-style output has no equivalent
  internal signal; every match line looks the same shape. The anchor is the
  search query itself, supplied by the caller.
* :func:`filter_prose_by_relevance` -- a PR description or review comment has
  no internal signal either. The anchor is external context the caller
  supplies -- typically the diff the prose is about.

The design deliberately does not touch code logic. A dropped line or
paragraph costs a reader a fact; a dropped statement can cost a program its
syntax. Keeping this file scoped to whole, independent units means the
failure mode is always "omitted a possibly useful line," never "produced
something that looks valid and is not."

Two invariants, shared by all three filters:

* **Whole units only.** A line or paragraph is kept verbatim or it is not
  there at all -- never partially. There is no character-level cut in this
  module, so there is nothing here that can corrupt a token in place the way
  the generic reducer's ``truncate_middle`` can.
* **Fails open.** No anchor material, no embedder, or an embedder that
  raises: the content comes back unchanged. A missing or broken relevance
  signal must never cost more than the relevance feature itself would have
  saved -- it must never cost correctness.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .detector import _SEARCH_LINE_RE
from .handlers.text_handlers import (
    _CONTINUATION_RE,
    _FENCE_RE,
    _HEADING_RE,
    _LEVEL_RE,
    _SETEXT_RE,
    _SEVERITY_ORDER,
    _TABLE_SEPARATOR_RE,
    _iter_lines,
)

__all__ = [
    "EmbedFn",
    "RelevanceResult",
    "cosine_similarity",
    "filter_log_by_relevance",
    "filter_search_by_relevance",
    "filter_prose_by_relevance",
]

#: A batch embedder: many texts in, one vector per text out, same order.
#: Deliberately this narrow -- not a client class -- so any provider (Azure
#: `text-embedding-3`, a self-hosted model, a test double) plugs in as a
#: plain function with no dependency on how it talks to anything.
EmbedFn = Callable[[list[str]], list[Sequence[float]]]

_DEFAULT_THRESHOLD = 0.35


@dataclass
class RelevanceResult:
    """What a filter did, for a caller that wants to show its work.

    Attributes:
        text: The filtered content. Equal to the input when nothing was
            dropped, including every fail-open case.
        anchor: The text the anchor embedding was built from, or ``None``
            when there was no anchor material (log) or none was supplied
            (search, prose).
        candidates: How many units were scored.
        dropped: How many of those fell below the threshold.
        skipped_reason: Why nothing happened, when nothing happened.
        dropped_lines: The dropped units' own text, for a caller that wants
            to show what was removed (a log line, a match line, a paragraph).
        kept_spans: ``(start, end)`` in ``text`` of each unit that was scored
            and judged relevant -- not anchor material or passthrough
            structure, which were never candidates.
        kept_texts: The same units' own text, derived from ``kept_spans``.
            A caller downstream of further transforms (a reversible pre-pass
            that may shift offsets) can relocate these by substring search in
            whatever the content has become, rather than trusting stale
            positions -- and then protect them from whatever general-purpose
            compression runs after this filter, so a paragraph kept for
            being relevant is not then randomly truncated by an unrelated
            pass.
    """

    text: str
    anchor: str | None = None
    candidates: int = 0
    dropped: int = 0
    skipped_reason: str | None = None
    dropped_lines: list[str] = field(default_factory=list)
    kept_spans: list[tuple[int, int]] = field(default_factory=list)
    kept_texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kept_spans and not self.kept_texts:
            self.kept_texts = [self.text[start:end] for start, end in self.kept_spans]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors, in plain Python.

    No numpy dependency for two vectors of a few thousand floats -- this
    runs once per candidate unit, not in a hot loop.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        A value in roughly ``[-1, 1]``. ``0.0`` when either vector is all
        zeros, since there is no direction to compare.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _omission_line(count: int, unit_label: tuple[str, str], marker: str | None) -> str:
    """The text spliced in for one run of consecutive drops.

    ``marker`` is the caller's own reducer marker (the same text used
    everywhere else content gets shortened) -- when given, it replaces the
    whole run verbatim, with no count or noun attached. ``None`` (the
    default for every filter below) keeps the descriptive phrase, which
    names what happened for a caller with no marker convention of its own.
    """
    if marker is not None:
        return f"{marker}\n"
    singular, plural = unit_label
    return f"... {count} low-relevance {singular if count == 1 else plural} omitted ...\n"


def _score_candidates(
    anchor_text: str,
    candidate_texts: list[str],
    embed_fn: EmbedFn,
    threshold: float,
) -> tuple[list[bool] | None, str | None]:
    """Batch-embed the anchor plus every candidate, score each against it.

    One call to ``embed_fn``, always -- the anchor is index 0, candidates
    follow in order. Every failure mode returns ``(None, reason)`` rather
    than raising, so every caller here gets the same fail-open shape for
    free instead of re-implementing it.

    Args:
        anchor_text: What every candidate is compared to.
        candidate_texts: Units to score, in order.
        embed_fn: Batch embedder.
        threshold: Minimum cosine similarity to keep a candidate.

    Returns:
        ``(keep_flags, None)`` on success, same length and order as
        ``candidate_texts``; ``(None, reason)`` on any failure.
    """
    try:
        vectors = embed_fn([anchor_text, *candidate_texts])
    except Exception:
        return None, "embedder raised; kept content unchanged"
    if len(vectors) != len(candidate_texts) + 1:
        return None, "embedder returned the wrong number of vectors"

    anchor_vec = vectors[0]
    keep = [
        cosine_similarity(vectors[i + 1], anchor_vec) >= threshold
        for i in range(len(candidate_texts))
    ]
    return keep, None


def _rebuild_with_omissions(
    units: list[tuple[int, int, str]],
    keep: dict[int, bool],
    unit_label: tuple[str, str],
    marker: str | None = None,
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """Splice units back together, collapsing consecutive drops into one line.

    ``units`` must tile the original content exactly (as ``_iter_lines``
    does): every character of the input appears in exactly one unit, in
    order. An index absent from ``keep`` is never a drop candidate --
    anchor material, blank separators, protected structure -- and is always
    kept verbatim.

    Args:
        units: ``(start, end, text)`` for every unit, in document order.
        keep: Per-index keep/drop decision for scored units only.
        unit_label: ``(singular, plural)`` noun for the omission marker,
            e.g. ``("match", "matches")``.
        marker: The caller's own reducer marker, spliced in verbatim per
            dropped run instead of the descriptive phrase. ``None`` (the
            default) keeps that phrase -- see :func:`_omission_line`.

    Returns:
        ``(rebuilt text, dropped units' own text, kept-candidate spans)``.
        The third element is the ``(start, end)`` of each unit that *was*
        scored and judged relevant, in the coordinates of the rebuilt text --
        not anchor material or passthrough structure, which were never
        candidates in the first place. A caller can use these to protect
        what the filter deliberately chose to keep from whatever
        general-purpose compression runs after it.
    """
    parts: list[str] = []
    dropped: list[str] = []
    kept_spans: list[tuple[int, int]] = []
    run = 0
    offset = 0

    def flush_run() -> None:
        nonlocal run, offset
        if run:
            omission = _omission_line(run, unit_label, marker)
            parts.append(omission)
            offset += len(omission)
            run = 0

    for index, (_start, _end, text) in enumerate(units):
        if index in keep and not keep[index]:
            run += 1
            dropped.append(text.rstrip("\n"))
            continue
        flush_run()
        parts.append(text)
        if index in keep and keep[index]:
            kept_spans.append((offset, offset + len(text)))
        offset += len(text)
    flush_run()

    return "".join(parts), dropped, kept_spans


def filter_log_by_relevance(
    content: str,
    embed_fn: EmbedFn,
    severity_floor: str = "WARN",
    threshold: float = _DEFAULT_THRESHOLD,
    marker: str | None = None,
) -> RelevanceResult:
    """Drop whole low-relevance log lines, keeping error material as-is.

    The anchor is built from the content's own severity-floor-and-above
    lines plus their stack-trace continuations -- the error already present
    in the log, not a reference set maintained separately. Every other
    non-blank line is scored against that anchor in one batched call; lines
    below ``threshold`` are dropped, and a run of consecutive drops collapses
    into a single omission line so the reader knows something was removed
    rather than reading a shorter log as a complete one -- by default the
    descriptive ``... N low-relevance lines omitted ...``, or the caller's
    own ``marker`` verbatim when one is given.

    Blank lines are never candidates -- there is nothing to score -- and
    pass through untouched, so they cannot spuriously end an omission run.

    Args:
        content: The log text.
        embed_fn: Batch embedder. Called at most once.
        severity_floor: Same meaning as ``LogStructureHandler.severity_floor``
            -- the level at and above which a line is anchor material rather
            than a candidate for dropping.
        threshold: Minimum cosine similarity to the anchor for a line to
            survive. Lower keeps more.
        marker: Replace each run of dropped lines with this text verbatim
            instead of the descriptive phrase -- the same marker used
            everywhere else content is shortened, for a caller that wants
            one consistent convention rather than two. ``None`` (the
            default) keeps the phrase.

    Returns:
        A :class:`RelevanceResult`. ``result.text`` is always safe to use in
        place of ``content`` -- on any failure it equals ``content``.
    """
    floor = _SEVERITY_ORDER.get(severity_floor.upper(), 4)

    # Unlike LogStructureHandler's in_preserved_block, this does not require
    # unbroken continuity from the anchor line. A traceback's own header
    # ("Traceback (most recent call last):", "Caused by:") rarely matches
    # the continuation pattern, and gating on the line right before it would
    # have meant one unrecognised header line severs every real frame after
    # it from the anchor -- exactly the kind of miss the embedder is meant
    # to catch, not something the regex pass should be fooled by first. A
    # line shaped like a frame is anchor material regardless of what came
    # immediately before it.
    lines = list(_iter_lines(content))
    is_anchor = [False] * len(lines)
    for index, (_start, _end, line) in enumerate(lines):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        match = _LEVEL_RE.search(stripped)
        severity = _SEVERITY_ORDER[match.group(1)] if match else None
        if severity is not None and severity >= floor:
            is_anchor[index] = True
        elif severity is None and _CONTINUATION_RE.match(stripped):
            is_anchor[index] = True

    anchor_text = "".join(lines[i][2] for i in range(len(lines)) if is_anchor[i])
    if not anchor_text.strip():
        return RelevanceResult(text=content, skipped_reason="no severity-floor material to anchor to")

    candidate_indices = [
        i for i, (_s, _e, line) in enumerate(lines)
        if not is_anchor[i] and line.strip()
    ]
    if not candidate_indices:
        return RelevanceResult(text=content, anchor=anchor_text, skipped_reason="nothing to score")

    candidate_texts = [lines[i][2].rstrip("\n") for i in candidate_indices]
    keep, skipped = _score_candidates(anchor_text, candidate_texts, embed_fn, threshold)
    if keep is None:
        return RelevanceResult(
            text=content, anchor=anchor_text, candidates=len(candidate_indices),
            skipped_reason=skipped,
        )

    keep_map = dict(zip(candidate_indices, keep))
    text, dropped, kept_spans = _rebuild_with_omissions(
        lines, keep_map, unit_label=("line", "lines"), marker=marker
    )

    return RelevanceResult(
        text=text,
        anchor=anchor_text,
        candidates=len(candidate_indices),
        dropped=len(dropped),
        dropped_lines=dropped,
        kept_spans=kept_spans,
    )


def filter_search_by_relevance(
    content: str,
    query: str,
    embed_fn: EmbedFn,
    threshold: float = _DEFAULT_THRESHOLD,
    marker: str | None = None,
) -> RelevanceResult:
    """Drop whole low-relevance grep-style match lines.

    Unlike a log, grep output has no internal signal for what mattered --
    every match line is shaped the same way, ``path:line:text``. The query
    that produced the results is the anchor, and it does not live in the
    content, so it must be supplied here rather than inferred.

    A line that does not look like a match (a file header some tools print,
    a blank separator) is never a candidate and passes through untouched --
    it is structural, not content to score.

    Args:
        content: The grep-style output.
        query: The search query the results came from. This is the anchor;
            an empty query means there is nothing to compare against, and
            the filter is a no-op.
        embed_fn: Batch embedder. Called at most once.
        threshold: Minimum cosine similarity to the query for a match line
            to survive. Lower keeps more.
        marker: Replace each run of dropped lines with this text verbatim
            instead of the descriptive phrase. ``None`` (the default) keeps
            the phrase -- see :func:`filter_log_by_relevance`.

    Returns:
        A :class:`RelevanceResult`. ``result.text`` is always safe to use in
        place of ``content`` -- on any failure it equals ``content``.
    """
    if not query or not query.strip():
        return RelevanceResult(text=content, skipped_reason="no query given to anchor against")

    lines = list(_iter_lines(content))
    candidate_indices = [
        i for i, (_s, _e, line) in enumerate(lines)
        if _SEARCH_LINE_RE.match(line.rstrip("\n"))
    ]
    if not candidate_indices:
        return RelevanceResult(text=content, anchor=query, skipped_reason="no match lines found to score")

    candidate_texts = [lines[i][2].rstrip("\n") for i in candidate_indices]
    keep, skipped = _score_candidates(query, candidate_texts, embed_fn, threshold)
    if keep is None:
        return RelevanceResult(
            text=content, anchor=query, candidates=len(candidate_indices),
            skipped_reason=skipped,
        )

    keep_map = dict(zip(candidate_indices, keep))
    text, dropped, kept_spans = _rebuild_with_omissions(
        lines, keep_map, unit_label=("match", "matches"), marker=marker
    )

    return RelevanceResult(
        text=text,
        anchor=query,
        candidates=len(candidate_indices),
        dropped=len(dropped),
        dropped_lines=dropped,
        kept_spans=kept_spans,
    )


def filter_prose_by_relevance(
    content: str,
    anchor: str,
    embed_fn: EmbedFn,
    threshold: float = _DEFAULT_THRESHOLD,
    marker: str | None = None,
) -> RelevanceResult:
    """Drop whole low-relevance paragraphs from prose (PR text, review comments).

    Like search, prose has no internal signal for what matters -- a PR
    description does not carry an "error" the way a log does. The anchor is
    external context the caller supplies, typically the diff the prose is
    about, so relevance is measured against the actual change rather than
    against other paragraphs in the same document.

    The unit is the paragraph: a run of non-blank, non-structural lines
    bounded by blank lines. Markdown structure -- headings, fenced code
    blocks, table separator rows -- is never a candidate and always survives
    untouched, matching what ``MarkdownStructureHandler`` itself protects;
    this filter runs before that handler ever sees the content, so removing
    those here would remove something the rule-based layer would otherwise
    have kept. List and quote markers are not specially protected: a list
    item is ordinary paragraph text for this filter, the same as a sentence.

    Args:
        content: The prose, plain or Markdown.
        anchor: External context to score paragraphs against -- typically a
            diff. Required; an empty anchor means there is nothing to
            compare against, and the filter is a no-op.
        embed_fn: Batch embedder. Called at most once.
        threshold: Minimum cosine similarity to the anchor for a paragraph
            to survive. Lower keeps more.
        marker: Replace each run of dropped paragraphs with this text
            verbatim instead of the descriptive phrase. ``None`` (the
            default) keeps the phrase -- see :func:`filter_log_by_relevance`.

    Returns:
        A :class:`RelevanceResult`. ``result.text`` is always safe to use in
        place of ``content`` -- on any failure it equals ``content``.
    """
    if not anchor or not anchor.strip():
        return RelevanceResult(text=content, skipped_reason="no anchor text given")

    units: list[tuple[int, int, str]] = []
    candidate_flags: list[bool] = []
    in_fence = False
    para_start: int | None = None
    para_end: int | None = None

    def flush_paragraph() -> None:
        nonlocal para_start, para_end
        if para_start is not None:
            units.append((para_start, para_end, content[para_start:para_end]))
            candidate_flags.append(True)
            para_start = None

    def append_structural(start: int, end: int, line: str) -> None:
        flush_paragraph()
        units.append((start, end, line))
        candidate_flags.append(False)

    for start, end, line in _iter_lines(content):
        stripped = line.rstrip("\n")

        if _FENCE_RE.match(stripped):
            append_structural(start, end, line)
            in_fence = not in_fence
            continue
        if in_fence:
            append_structural(start, end, line)
            continue
        if not stripped.strip():
            append_structural(start, end, line)
            continue
        if _HEADING_RE.match(stripped) or _SETEXT_RE.match(stripped):
            append_structural(start, end, line)
            continue
        if _TABLE_SEPARATOR_RE.match(stripped) and "-" in stripped:
            append_structural(start, end, line)
            continue

        if para_start is None:
            para_start = start
        para_end = end

    flush_paragraph()

    candidate_indices = [i for i, is_candidate in enumerate(candidate_flags) if is_candidate]
    if not candidate_indices:
        return RelevanceResult(text=content, anchor=anchor, skipped_reason="no paragraph text to score")

    candidate_texts = [units[i][2].rstrip("\n") for i in candidate_indices]
    keep, skipped = _score_candidates(anchor, candidate_texts, embed_fn, threshold)
    if keep is None:
        return RelevanceResult(
            text=content, anchor=anchor, candidates=len(candidate_indices),
            skipped_reason=skipped,
        )

    keep_map = dict(zip(candidate_indices, keep))
    text, dropped, kept_spans = _rebuild_with_omissions(
        units, keep_map, unit_label=("paragraph", "paragraphs"), marker=marker
    )

    return RelevanceResult(
        text=text,
        anchor=anchor,
        candidates=len(candidate_indices),
        dropped=len(dropped),
        dropped_lines=dropped,
        kept_spans=kept_spans,
    )
