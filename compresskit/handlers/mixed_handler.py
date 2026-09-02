"""Per-region routing: one handler per section, not one per file.

Every other handler in this package answers "what is structural in this
content?" for content of a single kind. This one answers it for content that
is several kinds at once, which is what a real prompt usually is.

It splits the input into typed sections (:mod:`compresskit.segmenter`), runs
the registered handler for each section's type over just that section, and
offsets the resulting mask into the parent. Sections with no handler -- prose,
most importantly -- are preserved verbatim.

Preserving prose is the point, not a side effect. In a mixed prompt the prose
is your instructions. Compressing them is how "Do not invent account numbers"
becomes "Do not inven ...", leaving the model bound by a rule it can no longer
read. The data next to those instructions is what there is to gain from, and it
is handled by the handler that understands it.

One deliberate exception: when the splitter finds nothing but prose, this
handler claims nothing at all, exactly as :class:`NoOpHandler` would. Plain
text then compresses as it always did. Prose is protected because there is
something beside it worth protecting it from -- not on its own account.
"""

from __future__ import annotations

from typing import Any

from ..detector import ContentType
from ..masks import StructureMask
from ..segmenter import ContentSection, split_into_sections
from .base import BaseStructureHandler, HandlerResult, StructureHandler


class MixedContentHandler(BaseStructureHandler):
    """Routes each section of mixed content to the handler for its type.

    Example:
        >>> from compresskit.handlers import JSONStructureHandler
        >>> handler = MixedContentHandler({ContentType.JSON: JSONStructureHandler()})
        >>> result = handler.get_mask('Note this:\\n{"a": 1, "b": 2}\\nEnd.')
        >>> result.metadata["sections"]
        3
    """

    def __init__(
        self,
        handlers: dict[ContentType, StructureHandler] | None = None,
        preserve_prose: bool = True,
    ):
        """Initialize the mixed-content handler.

        Args:
            handlers: Section-type to handler routing table. Types absent from
                it fall through to the prose rule.
            preserve_prose: Keep prose sections verbatim when the input has
                typed sections beside them. Turning this off compresses your
                instructions along with everything else.
        """
        super().__init__(name="mixed")
        self._handlers = handlers or {}
        self.preserve_prose = preserve_prose

    #: Fence languages that name a content type we handle better than "code".
    #: ```diff and ```json blocks are typed SOURCE_CODE by the splitter, since
    #: every fence is; sending a patch to the code handler loses whole diff
    #: lines, because that handler looks for signatures and finds none.
    _FENCE_LANGUAGES: dict[str, ContentType] = {
        "diff": ContentType.DIFF,
        "patch": ContentType.DIFF,
        "json": ContentType.JSON,
        "jsonl": ContentType.JSON,
        "log": ContentType.LOG,
        "console": ContentType.LOG,
        "yaml": ContentType.CONFIG,
        "yml": ContentType.CONFIG,
        "toml": ContentType.CONFIG,
        "ini": ContentType.CONFIG,
        "csv": ContentType.TABULAR,
        "tsv": ContentType.TABULAR,
        "md": ContentType.MARKDOWN,
        "markdown": ContentType.MARKDOWN,
    }

    def _section_type(self, section: ContentSection) -> ContentType:
        """The type to route a section by, honouring a fence's language tag."""
        if section.is_code_fence and section.language:
            override = self._FENCE_LANGUAGES.get(section.language.lower())
            if override is not None and override in self._handlers:
                return override
        return section.content_type

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Split, route each section, and stitch the masks back together."""
        sections = split_into_sections(content)
        typed = [s for s in sections if self._section_type(s) in self._handlers]

        # Nothing but prose: claim nothing, so plain text keeps compressing
        # the way it does without this handler in the way.
        if not typed:
            return HandlerResult(
                mask=StructureMask.empty(tokens),
                handler_name=self.name,
                confidence=1.0,
                metadata={
                    "sections": len(sections),
                    "routed": 0,
                    "reason": "no typed sections found",
                },
            )

        mask = [False] * len(content)
        # A section's veto has to reach the compressor too, or a repeated
        # log run inside a mixed prompt loses its dedup marker.
        forced = [False] * len(content)
        line_starts = _line_start_offsets(content)
        routed: list[dict] = []

        for section in sections:
            start, end = _section_bounds(section, line_starts, len(content))
            if start >= end:
                continue

            kind = self._section_type(section)
            handler = self._handlers.get(kind)
            if handler is None:
                if self.preserve_prose:
                    mask[start:end] = [True] * (end - start)
                    routed.append(_note(section, kind, "preserved", start, end))
                else:
                    routed.append(_note(section, kind, "compressible", start, end))
                continue

            # Run the sub-handler over the section alone, then shift its mask
            # into place. Every handler returns a mask aligned to the string it
            # was given, so this is arithmetic rather than translation.
            body = content[start:end]
            sub = handler.get_mask(body, list(body), language=section.language)
            mask[start : start + len(sub.mask.mask)] = sub.mask.mask
            if sub.force_compressible:
                forced[start : start + len(sub.force_compressible)] = (
                    sub.force_compressible
                )
            routed.append(_note(section, kind, sub.handler_name, start, end))

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.9,
            force_compressible=forced if any(forced) else None,
            metadata={
                "sections": len(sections),
                "routed": len(typed),
                "detail": routed,
            },
        )


def _line_start_offsets(content: str) -> list[int]:
    """Character offset of each line start, for mapping line spans to slices.

    ``ContentSection`` carries line numbers because the splitter is
    line-oriented; masks are character-indexed. This is the bridge.

    Args:
        content: The content the sections came from.

    Returns:
        One offset per line, in order.
    """
    offsets = [0]
    for index, char in enumerate(content):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _section_bounds(
    section: ContentSection,
    line_starts: list[int],
    length: int,
) -> tuple[int, int]:
    """Character range a section covers, including its trailing newline.

    Args:
        section: The section to place.
        line_starts: Offsets from :func:`_line_start_offsets`.
        length: Total content length, used to clamp the end.

    Returns:
        ``(start, end)`` character offsets, end exclusive.
    """
    if section.start_line >= len(line_starts):
        return length, length

    start = line_starts[section.start_line]
    end_line = section.end_line + 1
    end = line_starts[end_line] if end_line < len(line_starts) else length
    return start, min(end, length)


def _note(
    section: ContentSection,
    section_type: ContentType,
    disposition: str,
    start: int,
    end: int,
) -> dict:
    """One row of routing detail, for the result metadata."""
    return {
        "type": section_type.value,
        "handler": disposition,
        "language": section.language,
        "chars": end - start,
        "lines": [section.start_line, section.end_line],
    }
