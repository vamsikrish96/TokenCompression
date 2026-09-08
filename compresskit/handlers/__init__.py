"""Structure handlers.

Each handler knows how to find the structural parts of one content type and
returns a :class:`~compresskit.masks.StructureMask` marking what must survive.

Handlers never compress. They only decide what is navigational -- keys,
signatures, headers, changed lines -- and leave the rest to the reducers. That
split is what lets you swap in a better reducer later without touching any of
the format knowledge here.
"""

from .base import BaseStructureHandler, HandlerResult, NoOpHandler, StructureHandler
from .code_handler import CodeStructureHandler, is_tree_sitter_available
from .data_handlers import ConfigStructureHandler, TabularStructureHandler
from .json_handler import JSONStructureHandler, extract_json_schema
from .mixed_handler import MixedContentHandler
from .text_handlers import (
    DiffStructureHandler,
    LogStructureHandler,
    MarkdownStructureHandler,
    SearchStructureHandler,
)

__all__ = [
    "BaseStructureHandler",
    "CodeStructureHandler",
    "ConfigStructureHandler",
    "DiffStructureHandler",
    "HandlerResult",
    "JSONStructureHandler",
    "LogStructureHandler",
    "MixedContentHandler",
    "MarkdownStructureHandler",
    "NoOpHandler",
    "SearchStructureHandler",
    "StructureHandler",
    "TabularStructureHandler",
    "extract_json_schema",
    "is_tree_sitter_available",
]
