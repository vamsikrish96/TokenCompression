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
from .handlers.code_handler import CodeStructureHandler
from .handlers.data_handlers import ConfigStructureHandler, TabularStructureHandler
from .handlers.json_handler import JSONStructureHandler
from .handlers.mixed_handler import MixedContentHandler
from .lossless import compact_lossless
from .handlers.text_handlers import (
    DiffStructureHandler,
    LogStructureHandler,
    MarkdownStructureHandler,
)
from .masks import StructureMask, compute_entropy_mask_for_content, mask_to_spans
from .reducers import ReducerConfig, reduce_text

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
        reducer: How to reduce the spans the mask did not claim.
    """

    use_entropy_preservation: bool = True
    entropy_threshold: float = 0.85
    min_content_length: int = 100
    min_span_length: int = 50
    use_lossless: bool = True
    keep_mask: bool = False
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
}


def default_handlers() -> dict[ContentType, StructureHandler]:
    """Build the default content-type to handler routing table.

    ``TEXT``, ``LOG`` and ``UNKNOWN`` route to :class:`MixedContentHandler`,
    which splits the content and gives each section its own handler. Those are
    the three verdicts a mixed prompt actually receives: instructions wrapping
    a payload read as ``TEXT``, a log quoting a JSON body reads as ``LOG``, and
    anything unrecognised reads as ``UNKNOWN``. Sending them to a single
    handler is what loses the payload inside.

    ``JSON``, ``CODE``, ``DIFF`` and ``MARKDOWN`` go straight to their own
    handler: the detector only returns those when the whole input really is
    that one thing, so there is nothing to split.

    Returns:
        A fresh routing table. Mutate the copy, not a shared default.
    """
    section_handlers: dict[ContentType, StructureHandler] = {
        ContentType.JSON: JSONStructureHandler(),
        ContentType.CODE: CodeStructureHandler(),
        ContentType.DIFF: DiffStructureHandler(),
        ContentType.LOG: LogStructureHandler(),
        ContentType.MARKDOWN: MarkdownStructureHandler(),
        ContentType.TABULAR: TabularStructureHandler(),
        ContentType.CONFIG: ConfigStructureHandler(),
    }
    mixed = MixedContentHandler(handlers=section_handlers)

    return {
        **section_handlers,
        ContentType.TEXT: mixed,
        ContentType.LOG: mixed,
        ContentType.SEARCH: mixed,
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
        self._handlers = handlers if handlers is not None else default_handlers()
        self._noop_handler = NoOpHandler()
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

        # Reversible pre-pass, before anything lossy. Compaction keeps the
        # content's format intact, so it cannot change what the content is --
        # detection runs first only to pick the kind. `original` keeps the true
        # input either way, so it stays the one honest route back.
        original = content
        lossless_saved = 0
        if self.config.use_lossless:
            kind = _LOSSLESS_KIND.get(detection.content_type, "text")
            compacted = compact_lossless(content, kind)
            if len(compacted) < len(content):
                lossless_saved = len(content) - len(compacted)
                content = compacted

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
        if self.config.use_entropy_preservation:
            entropy_mask = compute_entropy_mask_for_content(
                content,
                threshold=self.config.entropy_threshold,
            )
            structure_mask = structure_mask.union(entropy_mask)

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

        compressed = self._reduce_with_mask(content, structure_mask)

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
                    "kind": _LOSSLESS_KIND.get(detection.content_type, "text"),
                    "chars_saved": lossless_saved,
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

    def _reduce_with_mask(self, content: str, mask: StructureMask) -> str:
        """Reduce the spans the mask left unclaimed, splice the rest verbatim.

        Args:
            content: Original content.
            mask: Character-aligned structure mask.

        Returns:
            Content with non-structural spans reduced.
        """
        parts: list[str] = []

        for span in mask_to_spans(mask):
            span_content = content[span.start : span.end]

            if span.is_structural or len(span_content) < self.config.min_span_length:
                parts.append(span_content)
                continue

            try:
                parts.append(self._reduce_fn(span_content))
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
