"""compresskit -- structure-preserving, rule-based context compression.

A vendored, dependency-free subset of Headroom's ``headroom.compression``
package, for environments where you can copy a folder but cannot install a
package or download model weights.

**Pure standard library.** No Magika, no Kompress, no model weights, no network
calls. Everything is regexes, a JSON tokenizer and Shannon entropy. Drop the
folder in your source tree and import it.

The idea in one paragraph: before you shrink a blob of context, work out which
parts of it carry the structure -- JSON keys, function signatures, diff hunk
headers, log levels, secrets and identifiers -- and only shrink the rest. A
truncated JSON payload costs a model the schema; a masked one keeps every key
and loses only the long string values.

Quick start:
    >>> from compresskit import compress
    >>> result = compress(payload)
    >>> result.compressed
    >>> result.savings_percentage

With configuration:
    >>> from compresskit import Compressor, CompressorConfig, ReducerConfig
    >>> config = CompressorConfig(reducer=ReducerConfig(target_ratio=0.5))
    >>> result = Compressor(config=config).compress(payload)

Reduction is lossy and one-way. ``result.original`` is the only route back, so
hold on to it if you need to show a user what was sent.
"""

from .compressor import (
    CompressionResult,
    Compressor,
    CompressorConfig,
    compress,
    default_handlers,
    estimate_tokens,
)
from .detector import ContentType, DetectionResult, RuleDetector
from .handlers import (
    BaseStructureHandler,
    CodeStructureHandler,
    ConfigStructureHandler,
    DiffStructureHandler,
    HandlerResult,
    JSONStructureHandler,
    LogStructureHandler,
    MarkdownStructureHandler,
    MixedContentHandler,
    NoOpHandler,
    StructureHandler,
    TabularStructureHandler,
    extract_json_schema,
    is_tree_sitter_available,
)
from .lossless import compact_lossless
from .masks import (
    EntropyScore,
    MaskSpan,
    StructureMask,
    apply_mask_to_text,
    compute_entropy_mask,
    compute_entropy_mask_for_content,
    mask_to_spans,
)
from .reducers import ReducerConfig, reduce_text
from .segmenter import ContentSection, split_into_sections

__version__ = "0.1.0"

__all__ = [
    # Simple API
    "compress",
    # Full API
    "Compressor",
    "CompressorConfig",
    "CompressionResult",
    "ReducerConfig",
    "reduce_text",
    # Detection
    "RuleDetector",
    "ContentType",
    "DetectionResult",
    # Handlers
    "default_handlers",
    "StructureHandler",
    "BaseStructureHandler",
    "HandlerResult",
    "NoOpHandler",
    "JSONStructureHandler",
    "CodeStructureHandler",
    "ConfigStructureHandler",
    "TabularStructureHandler",
    "DiffStructureHandler",
    "LogStructureHandler",
    "MarkdownStructureHandler",
    "MixedContentHandler",
    "extract_json_schema",
    "is_tree_sitter_available",
    # Masks
    "StructureMask",
    "MaskSpan",
    "EntropyScore",
    "mask_to_spans",
    "apply_mask_to_text",
    "compute_entropy_mask",
    "compute_entropy_mask_for_content",
    # Segmentation
    "split_into_sections",
    "ContentSection",
    # Lossless pre-pass
    "compact_lossless",
    # Misc
    "estimate_tokens",
    "__version__",
]
