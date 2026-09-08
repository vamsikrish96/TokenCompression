"""The compressor: detect, mask, reduce.

This is the entry point. One call does three things:

1. **Detect** the content type with rules (:mod:`compresskit.detector`).
2. **Mask** the structural parts with the handler for that type
   (:mod:`compresskit.handlers`).
3. **Reduce** everything the mask did not claim
   (:mod:`compresskit.reducers`).

The value of the design is the second step. A naive compressor truncates a
40 KB JSON payload and the model loses the schema; masking first means every
key, bracket and identifier survives while the long string values shrink. The
same applies to code (signatures survive, bodies shrink) and to diffs (changed
lines survive, context shrinks).

Usage:
    >>> from compresskit import compress
    >>> result = compress(payload)
    >>> result.compressed          # what to send to the model
    >>> result.savings_percentage  # what it bought you
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .detector import ContentType, DetectionResult, RuleDetector
from .handlers.base import NoOpHandler, StructureHandler
from .handlers.code_handler import (
    CodeStructureHandler,
    align_mask_to_lines,
    balance_mask_brackets,
)
from .handlers.data_handlers import ConfigStructureHandler, TabularStructureHandler
from .handlers.json_handler import JSONStructureHandler
from .handlers.mixed_handler import MixedContentHandler
from .lossless import compact_lossless, split_fold_note
from .handlers.text_handlers import (
    DiffStructureHandler,
    LogStructureHandler,
    MarkdownStructureHandler,
    SearchStructureHandler,
)
from .masks import StructureMask, compute_entropy_mask_for_content, mask_to_spans
from .reducers import ReducerConfig, reduce_text
from .relevance import (
    EmbedFn,
    filter_log_by_relevance,
    filter_prose_by_relevance,
    filter_search_by_relevance,
)

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate a token count at roughly four characters per token.

    This is a rule of thumb, not a tokenizer. It is close enough for reporting
    a ratio and for deciding whether compression was worth it; it is not close
    enough to budget a context window against a hard limit. Pass your own
    counter as ``token_estimator`` if you have a real tokenizer available.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return len(text) // 4


@dataclass
class CompressorConfig:
    """Configuration for :class:`Compressor`.

    Attributes:
        use_entropy_preservation: Preserve high-entropy words (API keys, UUIDs,
            hashes) wherever they appear, on top of whatever the handler
            marked. These cannot be reconstructed from context, so losing them
            loses information outright.
        entropy_threshold: Normalized Shannon entropy above which a word is
            preserved (0.0-1.0). Higher is more selective.
        min_content_length: Content shorter than this is returned unchanged.
            Below roughly this size the marker text costs more than the
            compression saves.
        min_span_length: Compressible spans shorter than this are left alone,
            for the same reason. This is the single span floor: it overrides
            ``reducer.min_length``, so lowering it here is enough. (Both used
            to apply independently, and lowering one alone silently did
            nothing.)
        use_lossless: Run reversible compaction before anything lossy. It
            strips ANSI colour, folds repeated lines and blocks, and regroups
            grep output -- all verified reversible, and a no-op when the
            content does not repeat. Free savings, so it runs first: folding
            repeats in text that truncation has already mangled finds nothing.
        keep_mask: Return the structure mask on the result. Off by default: the
            mask is one Python bool per character, which costs several times
            the content's own memory. Turn it on for reporting and debugging,
            where seeing exactly which characters were protected is the point.
        fold_constant_fields: For JSON record arrays and delimited rows, lift
            fields whose value is identical in every record into a leading
            comment and remove them from each record. Reversible, and it
            reaches redundancy nothing else here can: sixty records carrying
            ``"role": "member"`` hold that fact sixty times, never adjacently
            and never in a span long enough to truncate, so every other stage
            measures them at 0%.

            Off by default because it changes the payload's *shape*. That is
            harmless when a model reads the data to answer a question, and
            harmful when the payload is the template for the model's own output
            -- asked to return the same JSON, a model shown reduced records may
            hand back reduced records, or invent the missing fields. Nothing
            here can tell those two uses apart from the content; only you can.
        use_role_aware_reduction: Let a handler's span roles steer where the
            reduction budget is spent. Without it every compressible span is
            cut by the same fraction, so a lookup table's hundredth entry is
            as expensive to keep as the one branch that sets a discount. The
            mask cannot express that difference -- it only has True and False
            -- which is why the roles travel beside it.
        role_ratio_scale: Multiplier on ``reducer.target_ratio`` per role. Below
            1.0 cuts harder than the global setting, above 1.0 cuts more
            gently; the product is clamped to (0.05, 1.0]. Roles with no entry
            here are reduced at the configured ratio.
        log_relevance_embed_fn: Batch embedder used to drop whole low-relevance
            log lines before anything else runs -- see
            :func:`compresskit.relevance.filter_log_by_relevance`. ``None``
            (the default) leaves logs to the rule-based handler alone: no
            model call, and every line under ``min_span_length`` survives
            untouched because there is nothing safe to cut from it. This is
            deliberately scoped to logs. Code stays parser-and-rule-only --
            a wrong relevance call there can produce something that looks
            valid and is not, which no log line can do.
        log_relevance_threshold: Minimum cosine similarity to the log's own
            error material for a line to survive. Only consulted when
            ``log_relevance_embed_fn`` is set. Code *statements* stay
            parser-and-rule-only regardless of any embedder configured here
            or below -- a wrong relevance call on a line of logic can
            produce something that looks valid and is not, which no log
            line, comment or docstring can do. See
            ``code_comment_redundancy_embed_fn`` for the one place code gets
            an embedding-based option, and note it is scoped to comment and
            docstring *characters* only, never a statement.
        search_relevance_embed_fn: Batch embedder used to drop whole
            low-relevance grep-style match lines -- see
            :func:`compresskit.relevance.filter_search_by_relevance`. Unlike
            logs, a grep result carries no internal signal for what
            mattered, so the anchor is the search query itself, passed per
            call as ``compress(content, search_query="...")``. ``None`` (the
            default) leaves search output untouched.
        search_relevance_threshold: Minimum cosine similarity to the query
            for a match line to survive. Only consulted when
            ``search_relevance_embed_fn`` is set.
        prose_relevance_embed_fn: Batch embedder used to drop whole
            low-relevance paragraphs from prose (PR descriptions, review
            comments) -- see
            :func:`compresskit.relevance.filter_prose_by_relevance`. Prose
            also carries no internal signal, so the anchor -- typically the
            diff the prose is about -- is passed per call as
            ``compress(content, prose_anchor="...")``. ``None`` (the
            default) leaves prose untouched. Markdown structure (headings,
            fenced code, table rows) is never a candidate regardless.
        prose_relevance_threshold: Minimum cosine similarity to the anchor
            for a paragraph to survive. Only consulted when
            ``prose_relevance_embed_fn`` is set.
        code_comment_redundancy_embed_fn: Batch embedder used by the code
            handler to judge each comment or docstring against its own
            adjacent code, and drop it when it only restates what the code
            already says -- see
            :func:`compresskit.handlers.code_handler._score_comment_redundancy`.
            The inverse of the three filters above: there, high similarity
            to an anchor means *keep*; here, high similarity to the code
            means *drop*. Self-contained, like the log filter -- nothing to
            supply per call, since each candidate is compared to its own
            adjacent function or class rather than to one shared anchor.
            ``None`` (the default) leaves comments and docstrings to
            ``preserve_comments``/``preserve_docstrings`` alone. Never a
            candidate: any code statement, and any comment or docstring not
            clearly tied to one code unit (a module banner, a license
            header), which is protected outright rather than judged.
        code_comment_redundancy_threshold: Minimum cosine similarity to the
            adjacent code for a comment or docstring to be judged redundant
            and left compressible. Only consulted when
            ``code_comment_redundancy_embed_fn`` is set.
        reducer: How to reduce the spans the mask did not claim.
    """

    use_entropy_preservation: bool = True
    entropy_threshold: float = 0.85
    min_content_length: int = 100
    min_span_length: int = 50
    use_lossless: bool = True
    keep_mask: bool = False
    fold_constant_fields: bool = False
    use_role_aware_reduction: bool = True
    log_relevance_embed_fn: EmbedFn | None = None
    log_relevance_threshold: float = 0.35
    search_relevance_embed_fn: EmbedFn | None = None
    search_relevance_threshold: float = 0.35
    prose_relevance_embed_fn: EmbedFn | None = None
    prose_relevance_threshold: float = 0.35
    code_comment_redundancy_embed_fn: EmbedFn | None = None
    code_comment_redundancy_threshold: float = 0.5
    role_ratio_scale: dict[str, float] = field(
        default_factory=lambda: {
            # Repeated entries: the redundancy compression exists to find, so
            # they are cut at half the global ratio. Nothing else is rescaled
            # by default. Cutting logic *more gently* was tried and measured
            # across this package: every setting above 1.0 lost savings AND
            # lost more of the lines it was meant to protect, because a
            # gentler per-span ratio only moves where the cut falls, it does
            # not decide what the cut lands on. Use ``preserve_logic`` on the
            # code handler for that -- it is a guarantee rather than a nudge.
            "data": 0.5,
        }
    )
    reducer: ReducerConfig = field(default_factory=ReducerConfig)


@dataclass
class CompressionResult:
    """Result of a compression run.

    Attributes:
        compressed: The compressed content -- what you send onward.
        original: The input, kept verbatim. Reduction is one-way, so this is
            the only route back.
        compression_ratio: ``len(compressed) / len(original)``. Lower is more
            compressed.
        tokens_before: Estimated tokens before compression.
        tokens_after: Estimated tokens after compression.
        content_type: The detected content type.
        detection_confidence: How much to trust that detection (0.0-1.0).
        handler_used: Name of the structure handler that ran.
        preservation_ratio: Fraction of characters the mask marked structural.
        lossless_chars_saved: Characters removed by the reversible pre-pass,
            before any lossy work. Zero when the content had nothing to fold.
        mask: One bool per character, True where the character was protected
            as structural. Present only when ``CompressorConfig.keep_mask`` is
            set. Note it aligns to the content the mask was built over, which
            is the *compacted* text when ``lossless_chars_saved`` is non-zero,
            not to ``original``.
        metadata: Detection and handler details.
    """

    compressed: str
    original: str
    compression_ratio: float
    tokens_before: int
    tokens_after: int
    content_type: ContentType
    detection_confidence: float
    handler_used: str
    preservation_ratio: float
    lossless_chars_saved: int = 0
    mask: list[bool] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        """Estimated tokens saved. Never negative."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def savings_percentage(self) -> float:
        """Estimated percentage of tokens saved."""
        if self.tokens_before == 0:
            return 0.0
        return (self.tokens_saved / self.tokens_before) * 100


#: Which reversible compaction to try, per content type. "text" is the safe
#: default: it only strips ANSI colour and folds repeated blocks, both of which
#: are harmless on anything.
_LOSSLESS_KIND: dict[ContentType, str] = {
    ContentType.LOG: "log",
    ContentType.DIFF: "diff",
    ContentType.TEXT: "text",
    ContentType.MARKDOWN: "text",
    ContentType.CONFIG: "config",
    ContentType.SEARCH: "search",
    ContentType.UNKNOWN: "text",
    # Opt-in only -- downgraded to "text" unless
    # CompressorConfig.fold_constant_fields is set. See _lossless_kind_for.
    ContentType.JSON: "json_records",
    ContentType.TABULAR: "tabular_columns",
}

#: Kinds that rewrite the payload's shape rather than only its whitespace.
_SHAPE_CHANGING_KINDS = frozenset({"json_records", "tabular_columns"})

#: Line-comment token per language, handed to the reducer so an omitted middle
#: is announced as a comment rather than spliced in as an inline "...". Only
#: consulted for ``ContentType.CODE``.
_COMMENT_PREFIX: dict[str, str] = {
    "python": "#",
    "perl": "#",
    "ruby": "#",
    "bash": "#",
    "javascript": "//",
    "typescript": "//",
    "go": "//",
    "rust": "//",
    "java": "//",
    "c": "//",
    "cpp": "//",
    "csharp": "//",
    "kotlin": "//",
    "swift": "//",
    "php": "//",
    "sql": "--",
}


def _dominant_role(roles: list[str], start: int, end: int) -> str:
    """The role holding the most characters in ``roles[start:end]``.

    A span is one contiguous run of compressible characters and can straddle
    more than one role -- the tail of a body and the table declared after it.
    The majority decides, since the reducer treats the span as a unit.

    Args:
        roles: Per-character role labels.
        start: Span start, inclusive.
        end: Span end, exclusive.

    Returns:
        The winning role, or ``""`` when the span carries no labels.
    """
    counts: dict[str, int] = {}
    for role in roles[start:end]:
        if role:
            counts[role] = counts.get(role, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda role: counts[role])


def default_handlers(
    config: CompressorConfig | None = None,
) -> dict[ContentType, StructureHandler]:
    """Build the default content-type to handler routing table.

    ``TEXT``, ``LOG`` and ``UNKNOWN`` route to :class:`MixedContentHandler`,
    which splits the content and gives each section its own handler. Those are
    the three verdicts a mixed prompt actually receives: instructions wrapping
    a payload read as ``TEXT``, a log quoting a JSON body reads as ``LOG``, and
    anything unrecognised reads as ``UNKNOWN``. Sending them to a single
    handler is what loses the payload inside.

    ``JSON``, ``CODE``, ``DIFF``, ``MARKDOWN`` and ``SEARCH`` go straight to
    their own handler: the detector only returns those when the whole input
    really is that one thing, so there is nothing to split. This also sidesteps
    a real gap for SEARCH specifically: the segmenter that backs
    ``MixedContentHandler`` only ever tags a sub-section CODE, JSON, LOG or
    TEXT, never SEARCH, so routing it through ``mixed`` left grep output with
    no protection at all -- "no typed sections found" claimed nothing, and the
    generic reducer was free to truncate mid-match-line.

    Args:
        config: Where ``code_comment_redundancy_embed_fn`` and its threshold
            come from, so the CODE handler is built with them already wired
            in. ``None`` (the default) builds the CODE handler exactly as
            before -- no embedder, comments and docstrings left to
            ``preserve_comments``/``preserve_docstrings``.

    Returns:
        A fresh routing table. Mutate the copy, not a shared default.
    """
    section_handlers: dict[ContentType, StructureHandler] = {
        ContentType.JSON: JSONStructureHandler(),
        ContentType.CODE: CodeStructureHandler(
            comment_redundancy_embed_fn=(
                config.code_comment_redundancy_embed_fn if config else None
            ),
            comment_redundancy_threshold=(
                config.code_comment_redundancy_threshold if config else 0.5
            ),
        ),
        ContentType.DIFF: DiffStructureHandler(),
        ContentType.LOG: LogStructureHandler(),
        ContentType.MARKDOWN: MarkdownStructureHandler(),
        ContentType.TABULAR: TabularStructureHandler(),
        ContentType.CONFIG: ConfigStructureHandler(),
        ContentType.SEARCH: SearchStructureHandler(),
    }
    mixed = MixedContentHandler(handlers=section_handlers)

    return {
        **section_handlers,
        ContentType.TEXT: mixed,
        ContentType.LOG: mixed,
        ContentType.UNKNOWN: mixed,
    }


class Compressor:
    """Structure-preserving, rule-based compressor.

    Example:
        >>> compressor = Compressor()
        >>> result = compressor.compress(large_json_payload)
        >>> result.content_type
        <ContentType.JSON: 'json'>
    """

    def __init__(
        self,
        config: CompressorConfig | None = None,
        handlers: dict[ContentType, StructureHandler] | None = None,
        reduce_fn: Callable[[str], str] | None = None,
        detector: Any | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ):
        """Initialize the compressor.

        Args:
            config: Compression configuration.
            handlers: Content-type routing table. Defaults to
                :func:`default_handlers`.
            reduce_fn: How to reduce a non-structural span. Defaults to
                :func:`compresskit.reducers.reduce_text` with ``config.reducer``.
                Supply your own to plug in a different compressor.
            detector: Content detector. Anything with ``detect(str)`` (and
                optionally ``detect_batch``) works. Defaults to
                :class:`~compresskit.detector.RuleDetector`.
            token_estimator: Token counter. Defaults to :func:`estimate_tokens`.
        """
        self.config = config or CompressorConfig()
        self._detector = detector or RuleDetector()
        self._handlers = handlers if handlers is not None else default_handlers(self.config)
        self._noop_handler = NoOpHandler()
        # A caller-supplied reduce_fn is used verbatim. Only the default one is
        # re-derived per call, to pick up the language's comment token.
        self._custom_reduce_fn = reduce_fn
        self._reduce_fn = reduce_fn or self._default_reduce_fn
        self._estimate_tokens = token_estimator or estimate_tokens

        # One floor, not two. The compressor skips spans below
        # ``min_span_length`` and the reducer independently skips text below
        # ``ReducerConfig.min_length``; with both defaulting to 50, lowering
        # either one alone changed nothing at all and looked like the setting
        # did not work. ``min_span_length`` is the knob, so it wins.
        self._reducer_config = replace(
            self.config.reducer, min_length=self.config.min_span_length
        )

    def _default_reduce_fn(self, text: str) -> str:
        """Reduce a span with the configured reducer settings."""
        return reduce_text(text, self._reducer_config)

    def _lossless_kind_for(self, content_type: ContentType) -> str:
        """Which reversible pre-pass to run for this content.

        Shape-changing kinds are downgraded to ``"text"`` unless the caller
        asked for them, so the default behaviour is byte-for-byte what it was
        before they existed.
        """
        kind = _LOSSLESS_KIND.get(content_type, "text")
        if kind in _SHAPE_CHANGING_KINDS and not self.config.fold_constant_fields:
            return "text"
        return kind

    def _reduce_fn_for(
        self,
        content_type: ContentType,
        language: str | None,
    ) -> Callable[[str], str]:
        """The reducer to use for this content.

        Source code gets its language's comment token, so a removed middle
        comes back as ``# ... 12 lines omitted ...`` on its own line instead of
        an inline ``" ... "`` that breaks the syntax and hides how much went.
        Every other content type keeps the shared config untouched.

        ``language`` comes from the handler's metadata rather than the
        detector: the handler is the one that decided which grammar to apply.
        """
        if self._custom_reduce_fn is not None:
            return self._custom_reduce_fn
        if content_type is not ContentType.CODE:
            return self._default_reduce_fn

        prefix = _COMMENT_PREFIX.get(language or "")
        if prefix is None:
            return self._default_reduce_fn

        config = replace(self._reducer_config, comment_prefix=prefix)
        return lambda text: reduce_text(text, config)

    def _reducer_config_for(
        self,
        content_type: ContentType,
        language: str | None,
    ) -> ReducerConfig | None:
        """The config behind :meth:`_reduce_fn_for`, for per-role rescaling.

        Returns ``None`` when a custom reducer is installed: it was supplied as
        an opaque callable, so there is no ratio to scale and role-aware
        reduction stands aside rather than guessing.
        """
        if self._custom_reduce_fn is not None:
            return None
        if content_type is not ContentType.CODE:
            return self._reducer_config
        prefix = _COMMENT_PREFIX.get(language or "")
        if prefix is None:
            return self._reducer_config
        return replace(self._reducer_config, comment_prefix=prefix)

    def compress(
        self,
        content: str,
        content_type: ContentType | None = None,
        **kwargs: Any,
    ) -> CompressionResult:
        """Compress content, preserving its structure.

        Args:
            content: Content to compress.
            content_type: Skip detection and use this type instead.
            **kwargs: Passed through to the handler (for example
                ``language="python"`` for the code handler).

        Returns:
            A :class:`CompressionResult`.
        """
        if not content or len(content) < self.config.min_content_length:
            tokens = self._estimate_tokens(content)
            return CompressionResult(
                compressed=content,
                original=content,
                compression_ratio=1.0,
                tokens_before=tokens,
                tokens_after=tokens,
                content_type=ContentType.UNKNOWN,
                detection_confidence=0.0,
                handler_used="none",
                preservation_ratio=1.0,
                metadata={"skipped": "content too short"},
            )

        if content_type is None:
            detection = self._detector.detect(content)
        else:
            detection = DetectionResult(
                content_type=content_type,
                confidence=1.0,
                raw_label="override",
            )

        original = content

        # Whole-unit relevance filtering runs first, before either pre-pass
        # below: it selects which lines/paragraphs survive, and the search
        # lossless pass in particular *reformats* survivors (grouping
        # consecutive matches under one filename header), which no longer
        # looks like `path:line:text` -- run relevance after that and its
        # match-line regex finds nothing. Selection has to precede
        # formatting, not follow it. Opt-in, and exactly one of the three
        # ever fires per call since detection.content_type is a single
        # value. Code is never in this list -- see
        # CompressorConfig.log_relevance_embed_fn.
        relevance_dropped = 0
        relevance_skipped: str | None = None
        # Paragraphs the prose filter judged relevant, so they can be
        # protected from the ordinary paragraph-body compression
        # MarkdownStructureHandler still applies afterward -- surviving
        # *because it matters* should not then be randomly truncated by a
        # pass that has no idea a relevance decision was ever made. Relocated
        # by substring search once `content` is final (see below), since the
        # lossless pre-pass that runs after this can shift offsets.
        prose_kept_texts: list[str] = []
        if self.config.log_relevance_embed_fn is not None and detection.content_type is ContentType.LOG:
            relevance_result = filter_log_by_relevance(
                content,
                self.config.log_relevance_embed_fn,
                threshold=self.config.log_relevance_threshold,
                marker=self.config.reducer.marker,
            )
            content = relevance_result.text
            relevance_dropped = relevance_result.dropped
            relevance_skipped = relevance_result.skipped_reason
        elif self.config.search_relevance_embed_fn is not None and detection.content_type is ContentType.SEARCH:
            relevance_result = filter_search_by_relevance(
                content,
                kwargs.get("search_query", ""),
                self.config.search_relevance_embed_fn,
                threshold=self.config.search_relevance_threshold,
                marker=self.config.reducer.marker,
            )
            content = relevance_result.text
            relevance_dropped = relevance_result.dropped
            relevance_skipped = relevance_result.skipped_reason
        elif self.config.prose_relevance_embed_fn is not None and detection.content_type in (
            ContentType.TEXT, ContentType.MARKDOWN,
        ):
            relevance_result = filter_prose_by_relevance(
                content,
                kwargs.get("prose_anchor", ""),
                self.config.prose_relevance_embed_fn,
                threshold=self.config.prose_relevance_threshold,
                marker=self.config.reducer.marker,
            )
            content = relevance_result.text
            relevance_dropped = relevance_result.dropped
            relevance_skipped = relevance_result.skipped_reason
            prose_kept_texts = relevance_result.kept_texts

        # Reversible pre-pass, before anything lossy. Compaction keeps the
        # content's format intact, so it cannot change what the content is --
        # detection runs first only to pick the kind. `original` keeps the true
        # input either way, so it stays the one honest route back.
        lossless_saved = 0
        lossless_kind = self._lossless_kind_for(detection.content_type)
        if self.config.use_lossless:
            compacted = compact_lossless(content, lossless_kind)
            if len(compacted) < len(content):
                lossless_saved = len(content) - len(compacted)
                content = compacted

        # A fold note is an instruction, not data. Held aside so the reducer
        # cannot truncate it, and spliced back on at the end -- the same
        # separation of instructions from payload the gateway pattern applies.
        fold_note, content = split_fold_note(content)

        handler = self._handlers.get(detection.content_type, self._noop_handler)

        # Hand the detected language to the handler rather than making it
        # re-derive one. An explicit caller-supplied language still wins.
        if detection.language and "language" not in kwargs:
            kwargs["language"] = detection.language

        # Character-level tokens: the masks are indexed by character, so a
        # span maps straight back to a slice of the original string.
        tokens = list(content)

        handler_result = handler.get_mask(content, tokens, **kwargs)
        structure_mask = handler_result.mask

        # Union in the entropy signal. Scored over whole words rather than the
        # character tokens above: single characters never reach the length
        # floor, which would make this a silent no-op.
        # Not for code. Entropy scores whole words, so in source it protects
        # long identifiers and f-strings scattered through function bodies --
        # islands that split a body into fragments no reducer can cut safely.
        # Measured over this package's own modules it cost 16 points of
        # compression ratio and left every file unparseable. The signal it
        # exists for -- a credential that cannot be reconstructed -- is
        # already covered for code, where imports, signatures and every
        # top-level assignment are structural by construction.
        if self.config.use_entropy_preservation and detection.content_type is not ContentType.CODE:
            entropy_mask = compute_entropy_mask_for_content(
                content,
                threshold=self.config.entropy_threshold,
            )
            structure_mask = structure_mask.union(entropy_mask)

        # Protect what the prose relevance filter chose to keep. Relocated by
        # substring search rather than trusting the offsets it originally
        # reported: the lossless pre-pass between there and here can shift
        # them, and a fresh search against the content the handler actually
        # saw is simple and always correct instead of threading positions
        # through every intervening transform.
        if prose_kept_texts:
            protect = [False] * len(content)
            cursor = 0
            for kept_text in prose_kept_texts:
                found = content.find(kept_text, cursor)
                if found == -1:
                    continue
                end = found + len(kept_text)
                for i in range(found, end):
                    protect[i] = True
                cursor = end
            structure_mask = structure_mask.union(StructureMask(tokens=tokens, mask=protect))

        # A handler's veto outranks every other signal. Entropy sees a
        # timestamp as a high-value identifier and protects it; on a run of
        # verbatim repeats that protection is what stops line dedup from
        # recognising the run at all, so the repeats vanish without the
        # "[xN]" that tells the model they existed.
        forced = handler_result.force_compressible
        if forced and len(forced) == len(structure_mask.mask):
            structure_mask = StructureMask(
                tokens=structure_mask.tokens,
                mask=[
                    keep and not veto
                    for keep, veto in zip(structure_mask.mask, forced)
                ],
                metadata=structure_mask.metadata,
            )

        # Code is read line by line, and truncation keeps whole lines. A mask
        # that protects only part of a line hands the reducer a fragment whose
        # indentation is expendable, and dedented code does not parse.
        if detection.content_type is ContentType.CODE:
            aligned = align_mask_to_lines(content, structure_mask.mask)
            language = handler_result.metadata.get("language")
            if language:
                # Bracket balance is a whole-document property; the reducer
                # only ever sees one span.
                aligned = balance_mask_brackets(content, aligned, language)
            structure_mask = StructureMask(
                tokens=structure_mask.tokens,
                mask=aligned,
                metadata=structure_mask.metadata,
            )

        language = handler_result.metadata.get("language")
        reduce_fn = self._reduce_fn_for(detection.content_type, language)
        base_config = self._reducer_config_for(detection.content_type, language)

        # Roles are per character of the same content the mask was built over,
        # so they survive the line-alignment and bracket-balancing above --
        # those rewrite which characters are protected, never how many there
        # are.
        roles = handler_result.span_roles
        if roles is not None and len(roles) != len(structure_mask.mask):
            logger.warning(
                "Handler %r returned %d role labels for %d characters; "
                "ignoring them",
                handler_result.handler_name,
                len(roles),
                len(structure_mask.mask),
            )
            roles = None

        compressed = fold_note + self._reduce_with_mask(
            content, structure_mask, reduce_fn, roles=roles, base_config=base_config
        )

        # Ratios and token counts are reported against the true input, not
        # against the compacted intermediate -- otherwise the reversible pass
        # would silently vanish from the savings you are shown.
        return CompressionResult(
            compressed=compressed,
            original=original,
            compression_ratio=len(compressed) / len(original),
            tokens_before=self._estimate_tokens(original),
            tokens_after=self._estimate_tokens(compressed),
            content_type=detection.content_type,
            detection_confidence=detection.confidence,
            handler_used=handler_result.handler_name,
            preservation_ratio=structure_mask.preservation_ratio,
            lossless_chars_saved=lossless_saved,
            mask=list(structure_mask.mask) if self.config.keep_mask else None,
            metadata={
                "detection": {
                    "raw_label": detection.raw_label,
                    "language": detection.language,
                },
                "handler": handler_result.metadata,
                "lossless": {
                    "kind": lossless_kind,
                    "chars_saved": lossless_saved,
                },
                "relevance": {
                    "dropped_lines": relevance_dropped,
                    "skipped_reason": relevance_skipped,
                },
            },
        )

    def compress_batch(
        self,
        contents: list[str],
        **kwargs: Any,
    ) -> list[CompressionResult]:
        """Compress several contents.

        Args:
            contents: Contents to compress.
            **kwargs: Passed through to the handlers.

        Returns:
            Results in the same order as the input.
        """
        if not contents:
            return []

        if hasattr(self._detector, "detect_batch"):
            detections = self._detector.detect_batch(contents)
        else:
            detections = [self._detector.detect(c) for c in contents]

        return [
            self.compress(content, content_type=detection.content_type, **kwargs)
            for content, detection in zip(contents, detections)
        ]

    def _reduce_fn_for_role(
        self,
        role: str,
        base_config: ReducerConfig,
        fallback: Callable[[str], str],
    ) -> Callable[[str], str]:
        """Scale the reduction budget to what kind of material a span holds.

        Args:
            role: Dominant role of the span, possibly empty.
            base_config: The reducer config this content type would otherwise
                use, comment prefix and all.
            fallback: Reducer to use when the role carries no opinion.

        Returns:
            A reducer for this span.
        """
        if not self.config.use_role_aware_reduction:
            return fallback
        scale = self.config.role_ratio_scale.get(role)
        if scale is None or scale == 1.0:
            return fallback

        ratio = min(max(base_config.target_ratio * scale, 0.05), 1.0)
        config = replace(base_config, target_ratio=ratio)
        return lambda text: reduce_text(text, config)

    def _reduce_with_mask(
        self,
        content: str,
        mask: StructureMask,
        reduce_fn: Callable[[str], str] | None = None,
        roles: list[str] | None = None,
        base_config: ReducerConfig | None = None,
    ) -> str:
        """Reduce the spans the mask left unclaimed, splice the rest verbatim.

        Args:
            content: Original content.
            mask: Character-aligned structure mask.
            reduce_fn: How to reduce one span. Defaults to the compressor's
                configured reducer.
            roles: Optional per-character role labels from the handler. When
                given together with ``base_config``, each span is reduced at a
                rate matched to what it holds rather than at one flat ratio.
            base_config: The reducer config ``reduce_fn`` was built from. Roles
                are ignored without it, since there would be nothing to scale.

        Returns:
            Content with non-structural spans reduced.
        """
        reduce_fn = reduce_fn or self._reduce_fn
        parts: list[str] = []

        for span in mask_to_spans(mask):
            span_content = content[span.start : span.end]

            if span.is_structural or len(span_content) < self.config.min_span_length:
                parts.append(span_content)
                continue

            span_fn = reduce_fn
            if roles is not None and base_config is not None:
                span_fn = self._reduce_fn_for_role(
                    _dominant_role(roles, span.start, span.end), base_config, reduce_fn
                )

            try:
                parts.append(span_fn(span_content))
            except Exception:
                # A reducer failing must cost compression ratio, not content.
                # Splicing the original span back keeps the output valid.
                logger.warning(
                    "Reducer failed on span [%d:%d]; keeping it verbatim",
                    span.start,
                    span.end,
                    exc_info=True,
                )
                parts.append(span_content)

        return "".join(parts)

    def get_handler(self, content_type: ContentType) -> StructureHandler:
        """Return the handler registered for ``content_type``.

        Args:
            content_type: Content type to look up.

        Returns:
            The registered handler, or the no-op handler.
        """
        return self._handlers.get(content_type, self._noop_handler)

    def register_handler(
        self,
        content_type: ContentType,
        handler: StructureHandler,
    ) -> None:
        """Register a handler for a content type, replacing any existing one.

        Args:
            content_type: Content type to handle.
            handler: Handler instance.
        """
        self._handlers[content_type] = handler


def compress(content: str, **kwargs: Any) -> CompressionResult:
    """Compress one piece of content with default settings.

    Builds a fresh :class:`Compressor` each call. For anything hot, construct
    one compressor and reuse it -- the handlers hold compiled patterns and,
    where tree-sitter is installed, cached parsers.

    Args:
        content: Content to compress.
        **kwargs: Passed to :meth:`Compressor.compress`.

    Returns:
        A :class:`CompressionResult`.

    Example:
        >>> compress('{"users": [{"id": 1}]}').content_type
        <ContentType.UNKNOWN: 'unknown'>
    """
    return Compressor().compress(content, **kwargs)
