"""Code structure handler using AST parsing.

Extracts structural elements from source code:
- Import statements
- Function/method signatures
- Class definitions
- Type annotations
- Decorators
- Top-level declarations (module constants, ``package``, attributes)

Function bodies are marked as compressible while preserving signatures.
This enables the LLM to see all available functions/methods while body
implementations are compressed.

Uses tree-sitter for parsing when available, falls back to regex patterns.

The regex fallback is not a toy. tree-sitter is an optional native
dependency, so on any machine without it the fallback *is* the handler, and
a signature it fails to recognise is not merely unpreserved -- it lands in a
compressible span and gets truncated away. It therefore matches signatures
with an anchor plus a bracket-aware scanner rather than with a single
regex: a pattern like ``\\([^)]*\\)`` cannot cross the ``)`` in
``def f(a, b=dict())`` and fails to match the declaration *at all*.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from ..masks import StructureMask
from ..relevance import EmbedFn, cosine_similarity
from .base import BaseStructureHandler, HandlerResult

logger = logging.getLogger(__name__)

# Lazy-loaded tree-sitter
_tree_sitter_available: bool | None = None
_tree_sitter_local = threading.local()

#: Fraction of the source that may sit inside tree-sitter ERROR nodes before
#: the parse is treated as a failure. A grammar applied to the wrong language
#: (C++ parsed as Python, say) still returns a tree, just one made almost
#: entirely of errors -- and an error-riddled tree yields no structural spans,
#: which is indistinguishable from "this file has no structure" right up until
#: the compressor truncates the whole file.
_MAX_PARSE_ERROR_RATIO = 0.20

#: Preservation below this counts as "the handler recognised nothing". See
#: ``CodeStructureHandler._extract_mask``.
_DEGRADED_PRESERVATION = 0.02


def _check_tree_sitter() -> bool:
    """Check if tree-sitter is available and can actually parse.

    Constructs a parser and runs a minimal parse so that ABI mismatches
    between ``tree_sitter`` and ``tree_sitter_language_pack`` surface here
    instead of silently falling back to the text compressor at request time.
    """
    global _tree_sitter_available
    if _tree_sitter_available is None:
        try:
            from tree_sitter import Parser
            from tree_sitter_language_pack import get_language

            parser = Parser()
            parser.language = get_language("python")
            tree = parser.parse(b"x = 1\n")
            if tree.root_node.child_count == 0:
                raise RuntimeError("tree-sitter parse returned empty tree")
            _tree_sitter_available = True
        except ImportError:
            _tree_sitter_available = False
        except Exception:
            logger.warning(
                "tree-sitter imported but failed to parse; "
                "code-aware compression disabled (ABI mismatch?)"
            )
            _tree_sitter_available = False
    return _tree_sitter_available


def _get_parser(language: str) -> Any:
    """Return a **thread-local** tree-sitter parser for ``language``.

    tree-sitter ``Parser`` objects are pyo3 ``unsendable`` — touching
    one from a thread other than its creator panics. Handlers are routinely
    called from a thread pool, so parsers must never be shared across
    threads: one parser per (thread, language).
    """
    if not _check_tree_sitter():
        raise ImportError(
            "tree-sitter is not installed; the regex fallback handles this. "
            "Install tree-sitter and tree-sitter-language-pack for AST-accurate "
            "signature detection."
        )

    cache: dict[str, Any] | None = getattr(_tree_sitter_local, "parsers", None)
    if cache is None:
        cache = {}
        _tree_sitter_local.parsers = cache

    if language not in cache:
        from tree_sitter_language_pack import get_parser

        cache[language] = get_parser(language)  # type: ignore[arg-type]

    return cache[language]


# tree-sitter API compatibility. tree-sitter-language-pack switched to a
# Rust binding (>=1.0) where node accessors are METHODS (kind(),
# start_byte(), child(i)) and parse() takes str; the classic pybind API
# uses attributes (.type, .start_byte, .children) and parse(bytes).
# Without this shim the tree-sitter path raises TypeError on modern
# installs and silently falls back to regex.


def _ts_parse(parser: Any, content: str) -> Any:
    try:
        return parser.parse(content.encode("utf-8"))
    except TypeError:
        return parser.parse(content)


def _ts_root(tree: Any) -> Any:
    root = tree.root_node
    return root() if callable(root) else root


def _ts_kind(node: Any) -> str:
    kind = getattr(node, "type", None)
    if isinstance(kind, str):
        return kind
    return str(node.kind())


def _ts_start_byte(node: Any) -> int:
    start = node.start_byte
    return int(start()) if callable(start) else int(start)


def _ts_end_byte(node: Any) -> int:
    end = node.end_byte
    return int(end()) if callable(end) else int(end)


def _ts_children(node: Any) -> list[Any]:
    children = getattr(node, "children", None)
    if children is not None and not callable(children):
        return list(children)
    return [node.child(i) for i in range(node.child_count())]


def _ts_error_bytes(node: Any) -> int:
    """Total bytes covered by ERROR/MISSING nodes in this subtree.

    Does not recurse into an ERROR node -- its whole span already counts,
    and its children are salvage, not signal.
    """
    kind = _ts_kind(node)
    if kind in ("ERROR", "MISSING"):
        return _ts_end_byte(node) - _ts_start_byte(node)
    return sum(_ts_error_bytes(child) for child in _ts_children(node))


@dataclass
class CodeSpan:
    """A span of code with its structural role."""

    start: int
    end: int
    role: str  # "import", "signature", "body", "decorator", etc.
    is_structural: bool


# ---------------------------------------------------------------------------
# Bracket-aware scanning
#
# Every language here delimits a signature with brackets that can nest and
# with strings that can contain anything. A regex character class cannot
# express "the matching close paren", so the scanners below do it directly.
# ---------------------------------------------------------------------------

_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset(")]}")

#: Line-comment tokens per language, so a scanner does not mistake a bracket
#: or quote inside a comment for real syntax.
_LINE_COMMENT: dict[str, tuple[str, ...]] = {
    "python": ("#",),
    "perl": ("#",),
    "ruby": ("#",),
    "bash": ("#",),
    "javascript": ("//",),
    "typescript": ("//",),
    "go": ("//",),
    "rust": ("//",),
    "java": ("//",),
    "c": ("//",),
    "cpp": ("//",),
    "csharp": ("//",),
    "kotlin": ("//",),
    "swift": ("//",),
    "php": ("//", "#"),
    "sql": ("--",),
}


def _skip_string(content: str, index: int) -> int:
    """Index just past the string literal starting at ``index``.

    Handles triple quotes and backslash escapes. An unterminated
    single-line string ends at its newline rather than swallowing the rest
    of the file -- source being compressed is not always well-formed.
    """
    quote = content[index]
    if content.startswith(quote * 3, index):
        end = content.find(quote * 3, index + 3)
        return len(content) if end == -1 else end + 3

    i = index + 1
    n = len(content)
    while i < n:
        char = content[i]
        if char == "\\":
            i += 2
            continue
        if char == quote:
            return i + 1
        if char == "\n":
            return i
        i += 1
    return n


def _skip_noise(content: str, index: int, comments: tuple[str, ...]) -> int:
    """Advance past a string literal or line comment, else return ``index``."""
    char = content[index]
    if char in "\"'`":
        return _skip_string(content, index)
    for token in comments:
        if content.startswith(token, index):
            end = content.find("\n", index)
            return len(content) if end == -1 else end
    return index


def _scan_balanced(content: str, index: int, comments: tuple[str, ...] = ("#",)) -> int:
    """Index just past the bracket matching the one at ``index``, or -1.

    This is the fix for the single worst fallback bug: ``\\([^)]*\\)`` cannot
    cross a nested ``)``, so ``def f(a, b=dict(), c=1):`` matched *nothing*
    and the whole declaration was compressed away.
    """
    if index >= len(content) or content[index] not in _OPEN_TO_CLOSE:
        return -1

    depth = 0
    i = index
    n = len(content)
    while i < n:
        moved = _skip_noise(content, i, comments)
        if moved != i:
            i = moved
            continue
        char = content[i]
        if char in _OPEN_TO_CLOSE:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                return -1
        i += 1
    return -1


def _scan_to(
    content: str,
    index: int,
    stops: str,
    comments: tuple[str, ...] = ("#",),
) -> int:
    """Index of the first ``stops`` character at bracket depth 0, or -1.

    Returns -1 when an unmatched closing bracket is reached first, which
    means the construct ended without the expected terminator.
    """
    depth = 0
    i = index
    n = len(content)
    while i < n:
        moved = _skip_noise(content, i, comments)
        if moved != i:
            i = moved
            continue
        char = content[i]
        if depth == 0 and char in stops:
            return i
        if char in _OPEN_TO_CLOSE:
            depth += 1
        elif char in _CLOSERS:
            if depth == 0:
                return -1
            depth -= 1
        i += 1
    return -1


# Terminator kinds for _SIGNATURE_RULES.
#: optional balanced parens, then to the ``:`` that opens the body (Python).
_COLON = "colon"
#: optional balanced parens, then to the ``{`` or ``;`` on the same line.
_BRACE = "brace"
#: to the closing ``;`` (imports, ``use``, field declarations).
_SEMI = "semi"
#: the anchor plus its optional balanced argument list (decorators).
_PAREN = "paren"
#: a required balanced bracket, kept whole (``import (...)`` blocks).
_BLOCK = "block"
#: to end of line.
_EOL = "eol"


def _rule(pattern: str, terminator: str) -> tuple[re.Pattern[str], str]:
    return re.compile(pattern, re.MULTILINE), terminator


# Anchors match only up to the point where a scanner takes over, so they stay
# small and cannot fail on nesting. ``^[ \t]*`` rather than ``^\s*``: under
# re.MULTILINE ``\s`` matches newlines, so ``^\s*`` starts a match on blank
# lines *above* the construct and reports the wrong offset.
_SIGNATURE_RULES: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "python": [
        _rule(r"^[ \t]*(?:async[ \t]+)?def[ \t]+\w+[ \t]*(?=\()", _COLON),
        _rule(r"^[ \t]*class[ \t]+\w+", _COLON),
        _rule(r"^[ \t]*@[\w.]+", _PAREN),
    ],
    "javascript": [
        _rule(
            r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?"
            r"function[ \t]*\*?[ \t]*\w*[ \t]*(?=\()",
            _BRACE,
        ),
        _rule(r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?class[ \t]+\w+", _BRACE),
        _rule(
            r"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+\w+[ \t]*=[ \t]*"
            r"(?:async[ \t]+)?(?=\(|\w+[ \t]*=>)",
            _EOL,
        ),
        # Class members: constructors, methods, getters/setters. JS/TS had no
        # member pattern at all, so in ordinary OO code nearly every function
        # was invisible to the fallback.
        _rule(
            r"^[ \t]+(?:(?:static|async|get|set|public|private|protected|readonly"
            r"|override|declare|abstract)[ \t]+)*\*?[ \t]*[#\w$]+[ \t]*(?=\()",
            _BRACE,
        ),
    ],
    "typescript": [
        _rule(
            r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?"
            r"function[ \t]*\*?[ \t]*\w*[ \t]*(?:<[^\n>]*>)?[ \t]*(?=\()",
            _BRACE,
        ),
        _rule(
            r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:abstract[ \t]+)?"
            r"class[ \t]+\w+(?:<[^\n>]*>)?",
            _BRACE,
        ),
        _rule(r"^[ \t]*(?:export[ \t]+)?interface[ \t]+\w+(?:<[^\n>]*>)?", _BRACE),
        _rule(r"^[ \t]*(?:export[ \t]+)?enum[ \t]+\w+", _BRACE),
        _rule(r"^[ \t]*(?:export[ \t]+)?type[ \t]+\w+(?:<[^\n>]*>)?[ \t]*=", _EOL),
        _rule(
            r"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+\w+[ \t]*(?::[^=\n]+)?=[ \t]*"
            r"(?:async[ \t]+)?(?=\(|\w+[ \t]*=>)",
            _EOL,
        ),
        _rule(
            r"^[ \t]+(?:(?:static|async|get|set|public|private|protected|readonly"
            r"|override|declare|abstract)[ \t]+)*\*?[ \t]*[#\w$]+\??[ \t]*"
            r"(?:<[^\n>]*>)?[ \t]*(?=\()",
            _BRACE,
        ),
    ],
    "go": [
        _rule(r"^[ \t]*func[ \t]*(?:\([^)\n]*\)[ \t]*)?\w*[ \t]*(?=\()", _BRACE),
        _rule(r"^[ \t]*package[ \t]+\w+", _EOL),
        _rule(r"^[ \t]*type[ \t]+\w+(?:\[[^\n\]]*\])?[ \t]+\w+", _BRACE),
        _rule(r"^[ \t]*(?:const|var|import)[ \t]+(?=\()", _BLOCK),
    ],
    "rust": [
        _rule(
            r"^[ \t]*(?:pub(?:\([^)\n]*\))?[ \t]+)?"
            r"(?:default[ \t]+|const[ \t]+|async[ \t]+|unsafe[ \t]+|extern[ \t]+\"[^\"\n]*\"[ \t]+)*"
            r"fn[ \t]+\w+(?:<[^\n>]*>)?[ \t]*(?=\()",
            _BRACE,
        ),
        _rule(r"^[ \t]*(?:unsafe[ \t]+)?impl(?:<[^\n>]*>)?[ \t]+", _BRACE),
        _rule(
            r"^[ \t]*(?:pub(?:\([^)\n]*\))?[ \t]+)?(?:struct|enum|trait|union|mod)[ \t]+\w+",
            _BRACE,
        ),
        _rule(r"^[ \t]*(?:pub(?:\([^)\n]*\))?[ \t]+)?(?:const|static|type)[ \t]+\w+", _SEMI),
        _rule(r"^[ \t]*#!?(?=\[)", _BLOCK),
    ],
    "java": [
        _rule(
            r"^[ \t]*(?:(?:public|private|protected|static|final|abstract"
            r"|synchronized|native|strictfp|default)[ \t]+)*"
            r"(?:<[^\n>]*>[ \t]*)?"
            r"[\w.$]+(?:<[^\n;=]*?>)?(?:\[\])*[ \t]+\w+[ \t]*(?=\()",
            _BRACE,
        ),
        _rule(
            r"^[ \t]*(?:(?:public|private|protected|static|final|abstract|sealed)[ \t]+)*"
            r"(?:class|interface|enum|record)[ \t]+\w+(?:<[^\n>]*>)?",
            _BRACE,
        ),
        _rule(r"^[ \t]*@[\w.]+", _PAREN),
    ],
    "perl": [
        _rule(r"^[ \t]*sub[ \t]+\w+", _BRACE),
        _rule(r"^[ \t]*(?:package|class|role)[ \t]+[\w:]+", _EOL),
    ],
}

# Language-specific AST node types that are structural
_STRUCTURAL_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "import_statement",
        "import_from_statement",
        "function_definition",  # Just the signature part
        "class_definition",
        "decorated_definition",
        "type_alias_statement",
    },
    "javascript": {
        "import_statement",
        "export_statement",
        "function_declaration",
        "class_declaration",
        "method_definition",
        "field_definition",
        "arrow_function",  # Signature only
    },
    "typescript": {
        "import_statement",
        "export_statement",
        "function_declaration",
        "function_signature",
        "class_declaration",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
        "public_field_definition",
        "property_signature",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
    },
    "go": {
        "import_declaration",
        "package_clause",
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "const_declaration",
        "var_declaration",
        "interface_type",
    },
    "rust": {
        "use_declaration",
        "function_item",
        "function_signature_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "attribute_item",
        "inner_attribute_item",
        "const_item",
        "static_item",
        "type_item",
    },
    "java": {
        "import_declaration",
        "package_declaration",
        "class_declaration",
        "method_declaration",
        "constructor_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "field_declaration",
        "annotation",
        "marker_annotation",
    },
    "perl": {
        "use_statement",
        "use_version_statement",
        "subroutine_declaration_statement",
        "method_declaration_statement",
        "package_statement",
        "class_statement",
        "role_statement",
    },
    "c": {
        "preproc_include",
        "preproc_def",
        "function_definition",
        "declaration",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
    },
    "cpp": {
        "preproc_include",
        "preproc_def",
        "function_definition",
        "declaration",
        "field_declaration",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
        "template_declaration",
        "type_definition",
    },
    "csharp": {
        "using_directive",
        "namespace_declaration",
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
        "field_declaration",
    },
    "ruby": {
        "call",  # require / require_relative
        "class",
        "module",
        "method",
        "singleton_method",
    },
    "php": {
        "namespace_definition",
        "namespace_use_declaration",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "function_definition",
        "method_declaration",
        "property_declaration",
    },
    "kotlin": {
        "import_header",
        "package_header",
        "class_declaration",
        "object_declaration",
        "function_declaration",
        "property_declaration",
    },
    "swift": {
        "import_declaration",
        "class_declaration",
        "protocol_declaration",
        "function_declaration",
        "property_declaration",
    },
}

# Body child node types for container definitions (classes, impls,
# traits). A container's span up to its body is structural (the
# signature); the body itself is NOT marked -- recursion into the body
# emits signature spans for nested functions/methods, leaving their
# bodies compressible.
#
# ``interface_body`` is deliberately absent. An interface is nothing but
# member signatures, so treating its body as compressible preserved
# ``interface Shape`` and deleted every member it declared.
_CONTAINER_BODY_TYPES: frozenset[str] = frozenset(
    {
        "block",  # python class body
        "statement_block",  # js/ts
        "compound_statement",  # c/cpp
        "class_body",  # js/ts/java class body
        "declaration_list",  # rust impl/trait body
        "enum_body",  # java enum body
    }
)

# ---------------------------------------------------------------------------
# Span roles
#
# The mask answers "may this character be cut?". It cannot answer "what is
# this character?", and the two are not the same question. A lookup table's
# entries and an algorithm's control flow are both compressible, but cutting
# the first is nearly free and cutting the second removes the reason the
# function exists. Roles travel beside the mask so the reducer can spend its
# budget on the cheap material first instead of cutting whatever happens to
# come next in the file.
# ---------------------------------------------------------------------------

ROLE_SIGNATURE = "signature"
ROLE_DOCSTRING = "docstring"
ROLE_COMMENT = "comment"
ROLE_LOGIC = "logic"  # control flow and assignment: expensive to lose
ROLE_DATA = "data"  # repeated literal entries: cheap to lose
ROLE_BODY = "body"  # a function body with nothing more specific said about it

#: Node kinds that sit under a block but are not statements of it. Everything
#: else directly under a block is behaviour, and is labelled
#: :data:`ROLE_LOGIC` -- structural only when ``preserve_logic`` is on, but
#: labelled either way so the reducer knows to cut it last.
_NON_LOGIC_KINDS: frozenset[str] = frozenset(
    {
        "comment",  # handled by preserve_comments
        # A bare string under a block is a docstring, not behaviour. Labelling
        # it logic would overwrite the docstring role it already carries.
        "string",
        "expression_statement",
        "decorated_definition",
        "function_definition",
        "class_definition",
        "function_declaration",
        "class_declaration",
        "method_definition",
        "{",
        "}",
        ";",
        ":",
    }
)

#: Kept as documentation of what "logic" means per language, and consulted by
#: callers that want the vocabulary. The walk itself no longer matches on it:
#: see the block-children comment in ``visit_node``.
_LOGIC_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        "match_statement",
        "return_statement",
        "raise_statement",
        "assert_statement",
        "expression_statement",
        "assignment",
        "augmented_assignment",
        "delete_statement",
        "global_statement",
        "nonlocal_statement",
        "break_statement",
        "continue_statement",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "try_statement",
        "switch_statement",
        "return_statement",
        "throw_statement",
        "expression_statement",
        "variable_declaration",
        "lexical_declaration",
        "break_statement",
        "continue_statement",
    },
    "go": {
        "if_statement",
        "for_statement",
        "type_switch_statement",
        "expression_switch_statement",
        "return_statement",
        "expression_statement",
        "short_var_declaration",
        "assignment_statement",
        "var_declaration",
        "defer_statement",
        "go_statement",
    },
    "rust": {
        "if_expression",
        "for_expression",
        "while_expression",
        "loop_expression",
        "match_expression",
        "return_expression",
        "expression_statement",
        "let_declaration",
    },
}
_LOGIC_NODE_TYPES["typescript"] = _LOGIC_NODE_TYPES["javascript"]
_LOGIC_NODE_TYPES["tsx"] = _LOGIC_NODE_TYPES["javascript"]
_LOGIC_NODE_TYPES["jsx"] = _LOGIC_NODE_TYPES["javascript"]

#: Logic statements that own a block. Only the header (``for x in xs:``) is
#: kept structural -- protecting the whole node would preserve every nested
#: body with it and compress nothing. The block's contents are reached by
#: recursion and judged on their own.
_COMPOUND_LOGIC_TYPES: frozenset[str] = frozenset(
    {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "try_statement",
        "with_statement",
        "match_statement",
        "switch_statement",
        "type_switch_statement",
        "expression_switch_statement",
        "if_expression",
        "for_expression",
        "while_expression",
        "loop_expression",
        "match_expression",
    }
)

#: Collection literals. Spread over several lines they are a table of entries
#: -- the one thing in source code that is genuinely redundant -- so their
#: interior is labelled :data:`ROLE_DATA` and cut before anything else.
_COLLECTION_NODE_TYPES: frozenset[str] = frozenset(
    {
        "dictionary",
        "list",
        "set",
        "tuple",
        "object",
        "array",
        "literal_value",
        "composite_literal",
    }
)

# Language-detection markers for _detect_language
_LANGUAGE_MARKERS: dict[str, list[str]] = {
    "python": ["def ", "import ", "from ", "class ", "async def"],
    "javascript": ["function ", "const ", "let ", "var ", "=>"],
    "typescript": ["interface ", "type ", ": string", ": number"],
    "go": ["func ", "package ", "import (", "type "],
    "rust": ["fn ", "let mut", "impl ", "pub fn", "use "],
    "java": ["public class", "private ", "protected ", "void "],
    "perl": ["sub ", "my $", "our $", "package ", "use strict"],
    "c": ["#include", "int main(", "struct ", "typedef ", "printf("],
    "cpp": ["#include <", "std::", "namespace ", "template <", "::"],
    "csharp": ["using System", "namespace ", "public class", "var ", "async Task"],
    "ruby": ["def ", "end\n", "require ", "module ", "puts "],
    "php": ["<?php", "$this->", "function ", "public function", "namespace "],
    "kotlin": ["fun ", "val ", "var ", "package ", "import "],
    "swift": ["func ", "let ", "var ", "import ", "guard "],
    "bash": ["#!/bin/", "echo ", "fi\n", "done\n", "esac"],
    "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT INTO", "CREATE TABLE"],
}

# Import patterns for the fallback. Dotted and relative forms are the
# common case inside a package, and every one of them used to miss.
#
# Anchored at column 0, not at any indentation. A function-local import is
# part of that function's body; marking it structural left a protected
# island in the middle of a compressible body, and the truncation around it
# then kept indented lines whose `if`/`try` header had been removed.
_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(
        r"^(?:import[ \t]+[\w.]+|from[ \t]+[\w.]*[ \t]*import)", re.MULTILINE
    ),
    "javascript": re.compile(r"^(?:import\b|export[ \t]+.*from\b|require[ \t]*\()", re.M),
    "typescript": re.compile(r"^(?:import\b|export[ \t]+.*from\b|require[ \t]*\()", re.M),
    "go": re.compile(r'^import[ \t]+(?:\(|")', re.MULTILINE),
    "rust": re.compile(r"^(?:pub[ \t]+)?use[ \t]+\S", re.MULTILINE),
    "java": re.compile(r"^import[ \t]+(?:static[ \t]+)?[\w.*]+;", re.MULTILINE),
    "perl": re.compile(r"^(?:use|require)[ \t]+[\w:]+", re.MULTILINE),
    "c": re.compile(r"^#[ \t]*include\b", re.MULTILINE),
    "cpp": re.compile(r"^#[ \t]*include\b", re.MULTILINE),
    "csharp": re.compile(r"^using[ \t]+[\w.]+;", re.MULTILINE),
    "ruby": re.compile(r"^require(?:_relative)?[ \t]+\S", re.MULTILINE),
    "php": re.compile(r"^(?:use|require|include)(?:_once)?\b", re.MULTILINE),
    "kotlin": re.compile(r"^(?:import|package)[ \t]+[\w.]+", re.MULTILINE),
    "swift": re.compile(r"^import[ \t]+\w+", re.MULTILINE),
}

#: Node types that describe the module's interface and only mean that at the
#: outermost level. Nested inside a function they are ordinary statements.
_MODULE_LEVEL_ONLY: frozenset[str] = frozenset(
    {
        "use_declaration",
        "preproc_include",
        "using_directive",
        "package_clause",
        "package_declaration",
        "package_header",
        "namespace_use_declaration",
    }
)

#: Languages whose top level is delimited by indentation rather than braces.
_INDENT_LANGUAGES = frozenset({"python"})

#: Keywords that open a block, so a top-level span stops at the ``:``.
_PY_COMPOUND_KEYWORDS = (
    "if",
    "elif",
    "else",
    "for",
    "while",
    "with",
    "try",
    "except",
    "finally",
    "match",
    "case",
    "def",
    "class",
    "async",
)


def _comments_for(language: str) -> tuple[str, ...]:
    return _LINE_COMMENT.get(language, ("#",))


def _regex_signature_spans(content: str, language: str) -> list[CodeSpan]:
    """Signature spans found by anchor match plus bracket-aware scan."""
    rules = _SIGNATURE_RULES.get(language, [])
    comments = _comments_for(language)
    spans: list[CodeSpan] = []

    for pattern, terminator in rules:
        for match in pattern.finditer(content):
            end = _signature_end(content, match.end(), terminator, comments)
            if end > match.start():
                spans.append(
                    CodeSpan(
                        start=match.start(),
                        end=end,
                        role="signature",
                        is_structural=True,
                    )
                )
    return spans


def _signature_end(
    content: str,
    index: int,
    terminator: str,
    comments: tuple[str, ...],
) -> int:
    """Where the signature that begins before ``index`` ends, or -1."""
    n = len(content)
    cursor = index

    if terminator == _BLOCK:
        while cursor < n and content[cursor] in " \t":
            cursor += 1
        return _scan_balanced(content, cursor, comments)

    if terminator in (_COLON, _BRACE, _PAREN):
        probe = cursor
        while probe < n and content[probe] in " \t":
            probe += 1
        if probe < n and content[probe] == "(":
            closed = _scan_balanced(content, probe, comments)
            if closed == -1:
                return -1
            cursor = closed
        elif terminator == _PAREN:
            # A bare decorator/annotation: @cache, @Override. Stopping at the
            # anchor rather than end-of-line is what lets "@cache  # memo"
            # survive -- requiring \s*$ dropped the decorator entirely.
            return cursor

    if terminator == _PAREN:
        return cursor

    if terminator == _COLON:
        stop = _scan_to(content, cursor, ":\n", comments)
        return stop + 1 if stop != -1 and content[stop] == ":" else -1

    if terminator == _BRACE:
        # Bounded to the current line on purpose. Without the newline stop, a
        # bare call statement inside a body ("save(order)") would scan on to
        # some distant "{" and preserve everything in between.
        stop = _scan_to(content, cursor, "{;\n", comments)
        return stop + 1 if stop != -1 and content[stop] in "{;" else -1

    if terminator == _SEMI:
        stop = _scan_to(content, cursor, ";\n", comments)
        return stop + 1 if stop != -1 and content[stop] == ";" else -1

    # _EOL
    stop = content.find("\n", cursor)
    return n if stop == -1 else stop


def _docstring_spans(
    content: str,
    language: str,
    signatures: list[CodeSpan],
) -> list[CodeSpan]:
    """Preserve the string literal that opens a body.

    Truncating a body that begins with a docstring cut straight through the
    literal: the opening quotes were dropped and the closing quotes kept, so
    the file ended up with a stray terminator and would not parse at all.

    Preserving the docstring is both the cheapest way to avoid that and the
    highest-value line in a body for whoever reads the result. The tree-sitter
    path does this from the AST; this is the same rule for the fallback.
    """
    if language not in _INDENT_LANGUAGES:
        return []

    spans: list[CodeSpan] = []
    n = len(content)
    for signature in signatures:
        index = signature.end
        while index < n and content[index] in " \t\r\n":
            index += 1
        if index < n and content[index] in "\"'":
            spans.append(
                CodeSpan(
                    start=index,
                    end=_skip_string(content, index),
                    role="docstring",
                    is_structural=True,
                )
            )
    return spans


def _import_spans(content: str, language: str) -> list[CodeSpan]:
    """Import spans, extended over block forms.

    ``import (\\n "fmt"\\n)`` and ``use a::{b, c};`` used to end at the first
    newline, preserving the keyword and deleting every name it brought in.
    """
    pattern = _IMPORT_PATTERNS.get(language)
    if pattern is None:
        return []

    comments = _comments_for(language)
    spans: list[CodeSpan] = []
    for match in pattern.finditer(content):
        end = content.find("\n", match.end())
        end = len(content) if end == -1 else end

        opener = _scan_to(content, match.end(), "(\n{", comments)
        if opener != -1 and content[opener] in "({":
            closed = _scan_balanced(content, opener, comments)
            if closed > end:
                end = closed
        spans.append(CodeSpan(start=match.start(), end=end, role="import", is_structural=True))
    return spans


def _top_level_spans(content: str, language: str) -> list[CodeSpan]:
    """Preserve the shape of declarations at the outermost nesting level.

    Only imports and signatures used to be structural, so everything else at
    module level -- constants, ``package main``, ``if __name__ ==
    "__main__":``, Rust attributes -- was compressible. A truncated span then
    ate the opening of a dict and left its closing brace behind, which is
    both broken syntax and a silent loss of the names the module exports.

    For a multi-line value the *shape* is kept and the contents stay
    compressible: ``STATUS_MAP = {`` and its closing ``}`` survive, the
    entries between them do not. A 500-line lookup table should still
    compress.
    """
    comments = _comments_for(language)
    spans: list[CodeSpan] = []
    n = len(content)
    line_starts = [0]
    for i, char in enumerate(content):
        if char == "\n":
            line_starts.append(i + 1)

    index = 0
    while index < len(line_starts):
        start = line_starts[index]
        line_end = content.find("\n", start)
        line_end = n if line_end == -1 else line_end
        line = content[start:line_end]
        stripped = line.strip()

        # Only column-0 statements are "top level"; blank lines and
        # continuations of a previous statement are skipped.
        if not stripped or line[:1] in " \t":
            index += 1
            continue
        if any(stripped.startswith(token) for token in comments):
            index += 1
            continue

        # A closing bracket alone on a line closes a multi-line value; keep it
        # so the construct it terminates is not left dangling. It is one line
        # and no more -- scanning from an unmatched closer finds no depth-0
        # newline at all, and falling back to end-of-file would preserve the
        # entire remainder of the module.
        if stripped[0] in _CLOSERS:
            spans.append(
                CodeSpan(start=start, end=line_end, role="top-level", is_structural=True)
            )
            index += 1
            continue

        # A logical statement runs to the first newline outside any bracket.
        # -1 means the scan hit an unmatched closer or ran out of content, in
        # which case the physical line is the honest answer.
        statement_end = _scan_to(content, start, "\n", comments)
        statement_end = line_end if statement_end == -1 else statement_end

        opener = _scan_to(content, start, "([{", comments)
        opens_block = opener != -1 and opener < statement_end

        if language in _INDENT_LANGUAGES and stripped.split(" ")[0].rstrip(
            ":"
        ) in _PY_COMPOUND_KEYWORDS:
            colon = _scan_to(content, start, ":", comments)
            end = colon + 1 if colon != -1 else statement_end
        elif opens_block:
            end = opener + 1
        else:
            end = statement_end

        spans.append(CodeSpan(start=start, end=end, role="top-level", is_structural=True))
        index += 1

    return spans


def align_mask_to_lines(content: str, mask: list[bool]) -> list[bool]:
    """Promote every partially-structural line to fully structural.

    Entropy preservation scores whole *words*, so on a line like
    ``        self.repo.save(total)`` it protects the identifier and leaves
    the eight spaces in front of it compressible. Truncating the fragment
    around that island deletes the indentation and splices the identifier
    back at column 0 -- valid characters, invalid code.

    A line is the unit that survives truncation intact, so for code the mask
    snaps to line boundaries. The cost is preserving the rest of a line that
    already had a protected word on it, which is close to free; the
    alternative is output the model cannot parse.
    """
    aligned = list(mask)
    start = 0
    length = len(content)
    while start < length:
        end = content.find("\n", start)
        end = length if end == -1 else end
        window = aligned[start:end]
        if any(window) and not all(window):
            aligned[start:end] = [True] * (end - start)
        start = end + 1
    return aligned


def balance_mask_brackets(content: str, mask: list[bool], language: str) -> list[bool]:
    """Promote any compressible run that is not bracket-balanced.

    Truncation removes lines from *within* one compressible run, and the
    reducer keeps that removal self-contained. What it cannot see is a run
    that straddles a bracket whose partner lives in a different run -- drop
    the middle of ``],\\n    "javascript": [`` and the output has a closer
    with no opener, whichever way the reducer cuts.

    Balance is a property of the whole document, so it is enforced here,
    where the whole mask is in hand: a run that opens more than it closes (or
    closes what it did not open) is preserved entire.
    """
    comments = _comments_for(language)
    balanced = list(mask)
    n = len(content)
    start = 0
    while start < n:
        if balanced[start]:
            start += 1
            continue
        end = start
        while end < n and not balanced[end]:
            end += 1

        depth = 0
        lowest = 0
        i = start
        while i < end:
            moved = _skip_noise(content, i, comments)
            if moved != i:
                i = min(moved, end)
                continue
            if content[i] in _OPEN_TO_CLOSE:
                depth += 1
            elif content[i] in _CLOSERS:
                depth -= 1
                lowest = min(lowest, depth)
            i += 1

        if depth != 0 or lowest != 0:
            balanced[start:end] = [True] * (end - start)
        start = end + 1
    return balanced


def _score_comment_redundancy(
    candidates: list[tuple[CodeSpan, str, str]],
    embed_fn: EmbedFn,
    threshold: float,
) -> list[CodeSpan]:
    """Judge each comment or docstring against its own adjacent code.

    This is the inverse of :mod:`compresskit.relevance`: there, high
    similarity to an anchor means *keep*. Here, high similarity to the code a
    comment sits beside means it only restates what the code already says --
    safe to drop. Low similarity means it carries something the code does
    not (a "why", a caveat, an incident reference) -- kept protected.

    Unlike the three filters in :mod:`compresskit.relevance`, there is no
    single shared anchor: each candidate is scored against *its own*
    adjacent code, so every candidate contributes two vectors rather than
    sharing one anchor vector.

    Fails open: an embedder that raises or returns the wrong shape keeps
    every candidate protected rather than guessing which ones are safe to
    drop.

    Args:
        candidates: ``(span, comment_text, adjacent_code_text)`` for every
            comment or docstring tied to exactly one code unit. A candidate
            not clearly tied to one unit (module banner, license header,
            a comment between two methods) is never in this list -- it is
            protected by the caller before scoring is reached.
        embed_fn: Batch embedder.
        threshold: Minimum cosine similarity to the adjacent code for a
            candidate to be judged redundant and left compressible.

    Returns:
        One :class:`CodeSpan` per candidate, ``is_structural=True`` (kept)
        unless judged redundant.
    """
    if not candidates:
        return []

    texts: list[str] = []
    for _span, comment_text, anchor_text in candidates:
        texts.append(comment_text)
        texts.append(anchor_text)

    try:
        vectors = embed_fn(texts)
    except Exception:
        logger.warning(
            "comment redundancy embedder raised; keeping all %d candidates",
            len(candidates),
            exc_info=True,
        )
        vectors = None

    if vectors is None or len(vectors) != len(texts):
        return [
            CodeSpan(start=span.start, end=span.end, role=span.role, is_structural=True)
            for span, _c, _a in candidates
        ]

    result: list[CodeSpan] = []
    for i, (span, _comment_text, _anchor_text) in enumerate(candidates):
        similarity = cosine_similarity(vectors[i * 2], vectors[i * 2 + 1])
        redundant = similarity >= threshold
        result.append(
            CodeSpan(start=span.start, end=span.end, role=span.role, is_structural=not redundant)
        )
    return result


class CodeStructureHandler(BaseStructureHandler):
    """Handler for source code.

    Preserves:
    - Import/use statements
    - Function/method signatures (not bodies)
    - Class/struct/interface definitions
    - Type declarations
    - Decorators/annotations
    - Top-level declarations, and the shape of multi-line values

    Marks as compressible:
    - Function/method bodies
    - Comments (optionally preserved, or judged individually against their
      own adjacent code when ``comment_redundancy_embed_fn`` is set)
    - Whitespace

    Example:
        >>> handler = CodeStructureHandler()
        >>> code = '''
        ... def hello(name: str) -> str:
        ...     message = f"Hello, {name}!"
        ...     return message
        ... '''
        >>> result = handler.get_mask(code, language="python")
        >>> # Signature "def hello(name: str) -> str:" preserved
        >>> # Body content compressed
    """

    def __init__(
        self,
        preserve_comments: bool = False,
        use_tree_sitter: bool = True,
        default_language: str = "python",
        preserve_top_level: bool = True,
        preserve_docstrings: bool = True,
        safe_fallback: bool = True,
        preserve_logic: bool = False,
        preserve_blank_separators: bool = True,
        comment_redundancy_embed_fn: EmbedFn | None = None,
        comment_redundancy_threshold: float = 0.5,
    ):
        """Initialize the code handler.

        Args:
            preserve_comments: Whether to preserve comments as structural.
            use_tree_sitter: Whether to use tree-sitter for parsing.
                Falls back to regex if False or unavailable.
            default_language: Default language when detection fails.
            preserve_top_level: Preserve declarations at the outermost nesting
                level (module constants, ``package``, ``if __name__``) and the
                shape of multi-line values. See :func:`_top_level_spans`.
            preserve_docstrings: Preserve the leading string literal of a
                module, class or function body. It is the highest-value line
                in a body for a reader and the cheapest to keep.
            safe_fallback: When the handler recognises no structure at all in
                content that looks like code, preserve it verbatim instead of
                letting it be truncated. Refusing to compress unparsed code
                costs ratio; truncating it costs the code.
            preserve_logic: Treat the statements inside a body -- control flow,
                assignments, returns -- as structural. Off by default, which
                keeps the historical behaviour: a signature is the interface
                and the body is filler. That assumption suits "summarise this
                module's API" and fails "explain what this function does",
                where the answer is exactly the material the default drops.
                Turning it on trades ratio for behaviour-preserving output.
            preserve_blank_separators: Keep the blank lines that separate two
                preserved declarations. They cost a byte each and their
                absence is what makes compressed output stop looking like the
                file it came from.
            comment_redundancy_embed_fn: Batch embedder used to judge a
                comment or docstring against its own adjacent code and drop
                it when the comment only restates what the code already
                says. ``None`` (the default) leaves comments and docstrings
                to ``preserve_comments``/``preserve_docstrings`` alone: no
                model call. This never touches a code statement -- only
                comment and docstring *characters* are ever candidates, the
                same bright line :mod:`compresskit.relevance` draws for
                logs, search and prose. A comment not clearly tied to one
                function or class (a module banner, a license header, a
                comment sitting between two methods) is protected outright
                rather than judged, since there is no single adjacent unit
                to compare it to. tree-sitter only -- the regex fallback has
                no reliable notion of "this comment's enclosing function",
                so it leaves comments exactly as ``preserve_comments``
                already does.
            comment_redundancy_threshold: Minimum cosine similarity to the
                adjacent code for a comment or docstring to be judged
                redundant and left compressible. Only consulted when
                ``comment_redundancy_embed_fn`` is set.
        """
        super().__init__(name="code")
        self.preserve_comments = preserve_comments
        self.use_tree_sitter = use_tree_sitter
        self.default_language = default_language
        self.preserve_top_level = preserve_top_level
        self.preserve_docstrings = preserve_docstrings
        self.safe_fallback = safe_fallback
        self.preserve_logic = preserve_logic
        self.comment_redundancy_embed_fn = comment_redundancy_embed_fn
        self.comment_redundancy_threshold = comment_redundancy_threshold
        self.preserve_blank_separators = preserve_blank_separators

    def can_handle(self, content: str) -> bool:
        """Check if content looks like source code."""
        # Quick heuristic checks
        code_indicators = [
            "def ",
            "class ",
            "function ",
            "import ",
            "const ",
            "let ",
            "var ",
            "func ",
            "fn ",
            "pub ",
            "package ",
            "struct ",
            "interface ",
        ]
        return any(indicator in content for indicator in code_indicators)

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        language: str | None = None,
        **kwargs: Any,
    ) -> HandlerResult:
        """Extract structure mask from code.

        Args:
            content: Source code content.
            tokens: Character-level tokens.
            language: Programming language (auto-detected if None).
            **kwargs: Additional options.

        Returns:
            HandlerResult with mask marking structural elements.
        """
        # Detect language if not provided
        if language is None:
            language = self._detect_language(content)

        result: HandlerResult | None = None
        if self.use_tree_sitter and _check_tree_sitter():
            try:
                result = self._extract_with_tree_sitter(content, tokens, language)
            except Exception as e:
                # Was logged at debug. A silent downgrade to the regex path is
                # exactly what let this failure mode hide.
                logger.warning("Tree-sitter parsing failed, using regex fallback: %s", e)

        if result is None:
            result = self._extract_with_regex(content, tokens, language)

        return self._guard_degraded(content, tokens, result)

    def _guard_degraded(
        self,
        content: str,
        tokens: list[str],
        result: HandlerResult,
    ) -> HandlerResult:
        """Preserve content verbatim when nothing structural was recognised.

        A mask with no structural spans is indistinguishable, downstream, from
        a mask over genuinely structureless content -- so the compressor
        truncates the whole file to the target ratio and the code is gone. An
        unknown language, an unparseable file or a construct no rule matches
        all land here. Preserving is the only safe reading.
        """
        if not self.safe_fallback:
            return result
        if result.mask.preservation_ratio > _DEGRADED_PRESERVATION:
            return result
        if not self.can_handle(content):
            return result

        logger.warning(
            "Code handler recognised no structure (language=%s, parser=%s); "
            "preserving content verbatim rather than truncating it",
            result.metadata.get("language"),
            result.metadata.get("parser"),
        )
        return HandlerResult(
            mask=StructureMask.full(tokens),
            handler_name=self.name,
            confidence=0.0,
            metadata={**result.metadata, "degraded": "no structure recognised"},
        )

    def _extract_with_tree_sitter(
        self,
        content: str,
        tokens: list[str],
        language: str,
    ) -> HandlerResult:
        """Extract structure using tree-sitter AST.

        Args:
            content: Source code.
            tokens: Character tokens.
            language: Language name.

        Returns:
            HandlerResult with mask.
        """
        parser = _get_parser(language)
        tree = _ts_parse(parser, content)
        root = _ts_root(tree)

        # A grammar applied to the wrong language still returns a tree. Reject
        # one made mostly of errors so the regex path gets its turn, instead
        # of emitting an empty mask that reads as "no structure here".
        total_bytes = max(1, len(content.encode("utf-8")))
        error_ratio = _ts_error_bytes(root) / total_bytes
        if error_ratio > _MAX_PARSE_ERROR_RATIO:
            raise ValueError(
                f"{language} grammar produced {error_ratio:.0%} error nodes; "
                "content is probably not this language"
            )

        # Collect structural spans
        spans: list[CodeSpan] = []

        # Comments/docstrings tied to exactly one code unit, held aside for a
        # single batched redundancy call after the walk finishes rather than
        # decided node-by-node -- see _score_comment_redundancy.
        redundancy_candidates: list[tuple[CodeSpan, str, str]] = []

        # tree-sitter reports byte offsets; testing whether a node covers more
        # than one line is a byte-slice question, so keep one encoded view
        # rather than re-encoding the file at every collection literal.
        content_bytes = content.encode("utf-8")

        def add_docstring(body_node: Any, owner_node: Any | None) -> None:
            """Preserve a body's leading string literal.

            Python docstrings are ``expression_statement`` -> ``string``,
            not ``comment``, so ``preserve_comments`` never reached them.

            ``owner_node`` is the function/class this docstring belongs to,
            used as the redundancy anchor. ``None`` for the module docstring
            -- it documents the whole file, not one code unit, so it is
            always protected outright rather than judged.
            """
            if not self.preserve_docstrings:
                return
            strings = ("string", "concatenated_string")
            for child in _ts_children(body_node):
                kind = _ts_kind(child)
                if kind in ("comment", "\n", ":"):
                    continue
                # Grammar versions disagree on whether a bare string statement
                # is wrapped in expression_statement; accept either shape.
                inner = _ts_children(child)
                is_doc = kind in strings or (
                    kind == "expression_statement" and inner and _ts_kind(inner[0]) in strings
                )
                if is_doc:
                    doc_span = CodeSpan(
                        start=_ts_start_byte(child),
                        end=_ts_end_byte(child),
                        role="docstring",
                        is_structural=True,
                    )
                    if self.comment_redundancy_embed_fn is not None and owner_node is not None:
                        doc_start, doc_end = _ts_start_byte(child), _ts_end_byte(child)
                        anchor_bytes = (
                            content_bytes[_ts_start_byte(owner_node) : doc_start]
                            + content_bytes[doc_end : _ts_end_byte(body_node)]
                        )
                        redundancy_candidates.append((
                            doc_span,
                            content_bytes[doc_start:doc_end].decode("utf-8", "ignore"),
                            anchor_bytes.decode("utf-8", "ignore"),
                        ))
                    else:
                        spans.append(doc_span)
                return

        def visit_node(
            node: Any, depth: int = 0, enclosing: tuple[Any, Any] | None = None
        ) -> None:
            """Visit AST node and collect structural spans.

            ``enclosing`` is the ``(owner_node, body_node)`` of the nearest
            function or method this node sits inside, or ``None`` at module
            level or inside a class body but outside any one method. It is
            the redundancy anchor for a comment found at this point in the
            walk -- see the ``comment`` branch below -- and is only ever
            read there; every other branch passes it through unchanged.
            """
            node_type = _ts_kind(node)
            structural_types = _STRUCTURAL_NODE_TYPES.get(language, set())
            children = _ts_children(node)
            child_enclosing = enclosing

            # An import nested inside a function is part of that function's
            # body, not the module's interface. Marking it structural leaves a
            # protected island mid-body, and the truncation around that island
            # keeps indented lines whose `if`/`try` header has been removed.
            if depth > 1 and ("import" in node_type or node_type in _MODULE_LEVEL_ONLY):
                for child in children:
                    visit_node(child, depth + 1, enclosing)
                return

            # Label -- and, when asked, protect -- the statements a body is
            # made of. A statement is whatever sits directly under a block,
            # which is grammar-agnostic in a way that matching node kinds is
            # not: these grammars do not agree on whether a bare call is an
            # `expression_statement` or just a `call`, and matching the
            # wrapper silently missed every statement that had none.
            # Recursion continues regardless -- preserving a `for` header must
            # not exempt its own body from being judged.
            if node_type in _CONTAINER_BODY_TYPES:
                for child in children:
                    child_kind = _ts_kind(child)
                    if child_kind in structural_types or child_kind in _NON_LOGIC_KINDS:
                        continue  # a nested definition, judged on its own terms
                    block = None
                    if child_kind in _COMPOUND_LOGIC_TYPES:
                        for grandchild in _ts_children(child):
                            if _ts_kind(grandchild) in _CONTAINER_BODY_TYPES:
                                block = grandchild
                                break
                    spans.append(
                        CodeSpan(
                            start=_ts_start_byte(child),
                            # For a compound statement only the header is
                            # claimed; claiming the whole node would drag every
                            # nested body in with it and compress nothing.
                            end=(
                                _ts_start_byte(block)
                                if block is not None
                                else _ts_end_byte(child)
                            ),
                            role=ROLE_LOGIC,
                            is_structural=self.preserve_logic,
                        )
                    )

            if node_type in _COLLECTION_NODE_TYPES:
                start_byte = _ts_start_byte(node)
                end_byte = _ts_end_byte(node)
                # A collection written on one line is an expression, not a
                # table; there is nothing redundant in it to find.
                if content_bytes.count(b"\n", start_byte, end_byte):
                    spans.append(
                        CodeSpan(
                            start=start_byte + 1,  # inside the bracket
                            end=max(start_byte + 1, end_byte - 1),
                            role=ROLE_DATA,
                            is_structural=False,
                        )
                    )

            # Check if this is a structural node type
            if node_type in structural_types:
                # For functions, only the signature is structural
                if "function" in node_type or "method" in node_type:
                    # Find the body node and exclude it. A comment that sits
                    # between the header and the first real statement is, in
                    # this grammar, a sibling of the block rather than one of
                    # its children -- the block's own start byte lands after
                    # it. Left alone, the signature span (header to body
                    # start) would silently swallow that comment's bytes as
                    # protected interface text no matter what the redundancy
                    # judgment below decides, so the signature boundary stops
                    # at the first such leading comment instead of at the
                    # block when one is present.
                    body_node = None
                    signature_end = None
                    for child in children:
                        child_kind = _ts_kind(child)
                        if child_kind in ("block", "statement_block", "compound_statement"):
                            body_node = child
                            if signature_end is None:
                                signature_end = _ts_start_byte(body_node)
                            break
                        if child_kind == "comment" and signature_end is None:
                            # Stop at the *start of the comment's own line*,
                            # not at its first character: a signature ending
                            # mid-line at the "#" still claims that line's
                            # leading indentation, leaving the line
                            # part-protected. align_mask_to_lines promotes
                            # any part-protected line to fully protected,
                            # which would silently override the redundancy
                            # judgment for the comment sharing that line.
                            comment_start = _ts_start_byte(child)
                            newline = content_bytes.rfind(b"\n", 0, comment_start)
                            signature_end = newline + 1

                    if body_node:
                        # Signature is from start to body start (or to the
                        # first leading comment, whichever comes first).
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=signature_end,
                                role="signature",
                                is_structural=True,
                            )
                        )
                        add_docstring(body_node, node)
                        # Everything inside this body -- comments included --
                        # is now tied to this one function for redundancy
                        # purposes, overriding whatever enclosed the function
                        # itself (a closure's comment is judged against the
                        # inner function, not the outer one).
                        child_enclosing = (node, body_node)
                        # Body is compressible. The span is also what gives
                        # every character in it the "body" role, so it is read
                        # twice: once by the mask (which ignores non-structural
                        # spans) and once by the role projection (which does
                        # not).
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(body_node),
                                end=_ts_end_byte(body_node),
                                role="body",
                                is_structural=False,
                            )
                        )
                    else:
                        # No body found, preserve whole thing
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_end_byte(node),
                                role=node_type,
                                is_structural=True,
                            )
                        )
                elif node_type == "decorated_definition":
                    # Wrapper around decorator(s) + definition. Emit no
                    # span: recursion marks the decorators and gives the
                    # inner function its signature/body split. A whole-
                    # node span here would preserve the function body.
                    pass
                else:
                    # Container definitions (class, impl, trait): the
                    # signature runs to the body start; the body is NOT
                    # marked, so nested function bodies stay compressible
                    # (recursion emits their signature spans). Leaf
                    # declarations (imports, type aliases, structs) have
                    # no such body child and are preserved whole.
                    body_node = None
                    for child in children:
                        if _ts_kind(child) in _CONTAINER_BODY_TYPES:
                            body_node = child
                            break

                    if body_node is not None:
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_start_byte(body_node),
                                role="signature",
                                is_structural=True,
                            )
                        )
                        add_docstring(body_node, node)
                        # A comment directly in a class body but outside any
                        # one method is not clearly tied to a single unit --
                        # reset rather than inherit whatever enclosed the
                        # class, so it is protected outright, not judged.
                        child_enclosing = None
                    else:
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_end_byte(node),
                                role=node_type,
                                is_structural=True,
                            )
                        )
            elif node_type == "decorator":
                # Decorators are structural (preserved) on their own so
                # the decorated_definition wrapper doesn't need a span.
                spans.append(
                    CodeSpan(
                        start=_ts_start_byte(node),
                        end=_ts_end_byte(node),
                        role="decorator",
                        is_structural=True,
                    )
                )
            elif node_type == "comment":
                if self.comment_redundancy_embed_fn is not None:
                    if enclosing is not None:
                        owner_node, body_node = enclosing
                        c_start, c_end = _ts_start_byte(node), _ts_end_byte(node)
                        anchor_bytes = (
                            content_bytes[_ts_start_byte(owner_node) : c_start]
                            + content_bytes[c_end : _ts_end_byte(body_node)]
                        )
                        redundancy_candidates.append((
                            CodeSpan(start=c_start, end=c_end, role="comment", is_structural=False),
                            content_bytes[c_start:c_end].decode("utf-8", "ignore"),
                            anchor_bytes.decode("utf-8", "ignore"),
                        ))
                    else:
                        # Not clearly tied to one code unit -- protect
                        # outright rather than guess what it is about.
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_end_byte(node),
                                role="comment",
                                is_structural=True,
                            )
                        )
                elif self.preserve_comments:
                    spans.append(
                        CodeSpan(
                            start=_ts_start_byte(node),
                            end=_ts_end_byte(node),
                            role="comment",
                            is_structural=True,
                        )
                    )

            # Recurse into children
            for child in children:
                visit_node(child, depth + 1, child_enclosing)

        add_docstring(root, None)  # module docstring -- never judged, see above
        visit_node(root)

        # One batched call scores every comment/docstring collected above
        # against its own adjacent code. Deferred to here, after the walk
        # completes, so the redundancy embedder is called once per document
        # rather than once per candidate.
        if self.comment_redundancy_embed_fn is not None and redundancy_candidates:
            spans.extend(
                _score_comment_redundancy(
                    redundancy_candidates,
                    self.comment_redundancy_embed_fn,
                    self.comment_redundancy_threshold,
                )
            )

        # tree-sitter spans are BYTE offsets into the UTF-8 encoding;
        # the mask is indexed by CHARACTER. Any non-ASCII character
        # (docstrings, comments, string literals) shifts every later
        # span, so convert before masking. Skipped for pure-ASCII
        # content where the offsets coincide.
        spans = self._byte_spans_to_char_spans(spans, content)

        # Character-offset spans, so these are appended after conversion.
        if self.preserve_top_level:
            spans.extend(_top_level_spans(content, language))

        # Build mask from spans
        mask = self._spans_to_mask(spans, len(content))
        if self.preserve_blank_separators:
            mask = self._preserve_separator_blanks(content, mask)
        roles = self._spans_to_roles(spans, len(content))

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.95,
            span_roles=roles,
            metadata={
                "language": language,
                "parser": "tree-sitter",
                "structural_spans": len([s for s in spans if s.is_structural]),
                "preserve_logic": self.preserve_logic,
                "role_counts": _role_counts(roles),
                "comment_redundancy": {
                    "candidates": len(redundancy_candidates),
                    "dropped": sum(
                        1
                        for span in spans
                        if span.role in ("comment", "docstring") and not span.is_structural
                    ),
                }
                if self.comment_redundancy_embed_fn is not None
                else None,
            },
        )

    def _extract_with_regex(
        self,
        content: str,
        tokens: list[str],
        language: str,
    ) -> HandlerResult:
        """Extract structure using anchors plus bracket-aware scanning.

        Args:
            content: Source code.
            tokens: Character tokens.
            language: Language name.

        Returns:
            HandlerResult with mask.
        """
        spans = _import_spans(content, language)
        signatures = _regex_signature_spans(content, language)
        spans.extend(signatures)
        if self.preserve_docstrings:
            spans.extend(_docstring_spans(content, language, signatures))
        if self.preserve_top_level:
            spans.extend(_top_level_spans(content, language))

        # Build mask from spans
        mask = self._spans_to_mask(spans, len(content))
        if self.preserve_blank_separators:
            mask = self._preserve_separator_blanks(content, mask)
        # No AST, so no statement-level roles: the regex path can label the
        # declarations it recognises and nothing finer. Reporting the roles it
        # does have beats reporting none, but preserve_logic has no effect
        # here and the metadata says so rather than implying otherwise.
        roles = self._spans_to_roles(spans, len(content))

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.7,  # Lower confidence for regex
            span_roles=roles,
            metadata={
                "language": language,
                "parser": "regex",
                "structural_spans": len(spans),
                "preserve_logic": False,
                "role_counts": _role_counts(roles),
            },
        )

    @staticmethod
    def _byte_spans_to_char_spans(spans: list[CodeSpan], content: str) -> list[CodeSpan]:
        """Convert byte-offset spans to character-offset spans.

        tree-sitter reports node positions as byte offsets in the UTF-8
        encoding. For pure-ASCII content byte == char and the spans are
        returned unchanged. Otherwise a byte->char table is built once
        and every span endpoint is remapped.
        """
        n_bytes = len(content.encode("utf-8"))
        if n_bytes == len(content):
            return spans

        # byte_to_char[b] = index of the character containing byte b;
        # byte_to_char[n_bytes] = len(content) so exclusive ends map.
        byte_to_char = [0] * (n_bytes + 1)
        byte_pos = 0
        for char_idx, ch in enumerate(content):
            ch_width = len(ch.encode("utf-8"))
            for b in range(byte_pos, byte_pos + ch_width):
                byte_to_char[b] = char_idx
            byte_pos += ch_width
        byte_to_char[n_bytes] = len(content)

        return [
            CodeSpan(
                start=byte_to_char[min(span.start, n_bytes)],
                end=byte_to_char[min(span.end, n_bytes)],
                role=span.role,
                is_structural=span.is_structural,
            )
            for span in spans
        ]

    def _spans_to_mask(self, spans: list[CodeSpan], length: int) -> list[bool]:
        """Convert spans to character-level mask.

        Args:
            spans: List of code spans.
            length: Total content length.

        Returns:
            Boolean mask aligned to characters.
        """
        mask = [False] * length

        for span in spans:
            if span.is_structural:
                start = min(max(span.start, 0), length)
                end = min(span.end, length)
                if start < end:
                    mask[start:end] = [True] * (end - start)

        return mask

    def _spans_to_roles(self, spans: list[CodeSpan], length: int) -> list[str]:
        """Project span roles onto a per-character array.

        Non-structural spans are laid down first and structural ones on top,
        so a docstring keeps its own role rather than inheriting "body" from
        the span that encloses it. Order of arrival therefore does not matter,
        which it otherwise would: the body span is appended after the
        docstring span that sits inside it.

        Args:
            spans: Spans collected by whichever parser ran.
            length: Content length in characters.

        Returns:
            Role per character; empty string where nothing was claimed.
        """
        roles = [""] * length

        for structural in (False, True):
            for span in spans:
                if span.is_structural is not structural or not span.role:
                    continue
                start = min(max(span.start, 0), length)
                end = min(span.end, length)
                if start < end:
                    roles[start:end] = [span.role] * (end - start)

        return roles

    def _preserve_separator_blanks(self, content: str, mask: list[bool]) -> list[bool]:
        """Keep blank lines that sit between two preserved lines.

        A run of blank lines is unclaimed by every span, so it reads to the
        reducer as compressible material between two protected declarations
        and disappears. The saving is a byte per line and the cost is that
        ``return order`` ends up welded to the next ``def``, which no longer
        resembles the file it came from.

        Only runs that separate two *preserved* lines are kept, and at most
        two lines of any run: blank space inside a body being cut is still
        fair game.

        Args:
            content: Original source.
            mask: Character mask to extend.

        Returns:
            A new mask with separator blanks marked structural.
        """
        out = list(mask)
        offsets: list[tuple[int, int]] = []
        pos = 0
        for line in content.split("\n"):
            offsets.append((pos, pos + len(line)))
            pos += len(line) + 1

        lines = content.split("\n")
        n = len(lines)

        def is_preserved(index: int) -> bool:
            start, end = offsets[index]
            return any(out[start:end])

        index = 0
        while index < n:
            if lines[index].strip():
                index += 1
                continue

            run_start = index
            while index < n and not lines[index].strip():
                index += 1
            run_end = index  # exclusive

            above = run_start - 1
            below = run_end
            if above < 0 or below >= n:
                continue
            if not (is_preserved(above) and is_preserved(below)):
                continue

            # Cap the run: two blank lines is the widest separator any of
            # these languages uses conventionally.
            for line_index in range(run_start, min(run_end, run_start + 2)):
                start, end = offsets[line_index]
                # The line itself is empty; it is the newline terminating it
                # that has to survive.
                for char_index in range(start, min(end + 1, len(out))):
                    out[char_index] = True

        return out

    def _detect_language(self, content: str) -> str:
        """Detect programming language from content.

        Args:
            content: Source code content.

        Returns:
            Language name (lowercase).
        """
        scores: dict[str, int] = {}
        for lang, patterns in _LANGUAGE_MARKERS.items():
            scores[lang] = sum(1 for p in patterns if p in content)

        if not scores or max(scores.values()) == 0:
            return self.default_language

        return max(scores, key=lambda k: scores[k])


def _role_counts(roles: list[str]) -> dict[str, int]:
    """Characters claimed per role, for the handler metadata.

    Only for reporting: it is what lets a caller see that a file was 40%
    ``data`` and 12% ``logic`` without re-deriving the spans.
    """
    counts: dict[str, int] = {}
    for role in roles:
        if role:
            counts[role] = counts.get(role, 0) + 1
    return counts


def is_tree_sitter_available() -> bool:
    """Check if tree-sitter is available."""
    return _check_tree_sitter()
