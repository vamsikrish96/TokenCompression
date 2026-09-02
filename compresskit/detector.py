"""Rule-based content type detection.

Headroom detects content types with Magika, a small ONNX model. That is not an
option in an environment where you cannot download model weights, so this
package replaces it with deterministic rules: no model, no network, no
third-party package.

The rules are honest heuristics, not a classifier. They route content to a
structure handler; a wrong route costs compression ratio, never correctness,
because every handler falls back to "preserve nothing extra" rather than
"corrupt the content".

Detection is a comparison, not a ladder. Every candidate type is scored on
one scale -- the fraction of non-blank lines carrying evidence for it -- and
the highest score wins.

That matters more than it sounds. Trying types in a fixed order and taking the
first hit misreads anything that quotes another format: a log containing a
Python traceback matched a code marker on its first line and was classified as
source; release notes quoting a patch were classified as a diff. In both cases
a minority signal won purely because it was tested first. Scoring lets the
majority signal win instead.

Two exceptions, both deliberate:

* **JSON is not scored.** It either parses or it does not, and a parse is proof
  rather than evidence, so it short-circuits with confidence 1.0.
* **Text is the floor.** When no type reaches a modest share of lines, the
  content is prose that happens to contain a bracket or a keyword.

Confidence tracks the winning share, so a file that is 90% log lines reports
higher than one that is 20%.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """High-level content categories for compression routing."""

    JSON = "json"
    CODE = "code"
    LOG = "log"
    DIFF = "diff"
    MARKDOWN = "markdown"
    SEARCH = "search"
    TABULAR = "tabular"
    CONFIG = "config"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result of content detection.

    Attributes:
        content_type: The routed category.
        confidence: 0.0-1.0. Only the JSON branch, which parses, reaches 1.0;
            treat everything below that as a best guess.
        raw_label: The specific label detected (e.g. "python", "markdown").
        language: For code, the voted language. None otherwise.
        metadata: Detector-specific extras.
    """

    content_type: ContentType
    confidence: float
    raw_label: str
    language: str | None = None
    metadata: dict = field(default_factory=dict)


# A unified-diff / git-patch header, anchored to line starts so a "---" rule in
# Markdown or a stray "@@" in prose does not trip it.
_DIFF_HEADER_RE = re.compile(
    r"^(?:diff --git |index [0-9a-f]{7,}|@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)",
    re.MULTILINE,
)

# The ---/+++ file-header pair, which only means "diff" when adjacent.
_DIFF_FILE_PAIR_RE = re.compile(r"^--- .*\n\+\+\+ ", re.MULTILINE)

# A log level in level position: at the start of a line, or after up to four
# leading fields that actually look like log fields.
#
# A leading field must be bracketed ("[main]"), digit-led ("2026-01-01",
# "12345"), or end in a colon or bracket ("app[7]:"). Allowing any \S+ here
# would match the sentence "The report showed an ERROR in the totals" -- four
# words and then a level -- and route English prose to the log handler.
_LOG_LINE_RE = re.compile(
    r"^(?:(?:\[[^\]\n]*\]|[0-9]\S*|\S*[:\]])[ \t]+){0,4}\[?"
    r"(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL)"
    r"\]?[\s:\]]",
    re.MULTILINE,
)

# Per-language markers. These are word-boundary regexes rather than substring
# tests on purpose: substring matching counts "define" as a Go `func` and any
# English sentence containing " var " as JavaScript.
_LANGUAGE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+\w+[ \t]*\(", re.MULTILINE),
        re.compile(r"^[ \t]*class[ \t]+\w+.*:", re.MULTILINE),
        re.compile(r"^[ \t]*(?:from[ \t]+[\w.]+[ \t]+)?import[ \t]+\w", re.MULTILINE),
        re.compile(r"^[ \t]*@\w+", re.MULTILINE),
        re.compile(r"\bself\b"),
    ],
    "javascript": [
        re.compile(r"^[ \t]*(?:async[ \t]+)?function[ \t]+\w+[ \t]*\(", re.MULTILINE),
        re.compile(r"^[ \t]*(?:const|let|var)[ \t]+\w+[ \t]*=", re.MULTILINE),
        re.compile(r"=>[ \t]*[{(]"),
        re.compile(r"\brequire[ \t]*\("),
        re.compile(r"^[ \t]*module\.exports\b", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"^[ \t]*interface[ \t]+\w+", re.MULTILINE),
        re.compile(r"^[ \t]*type[ \t]+\w+[ \t]*=", re.MULTILINE),
        re.compile(r":[ \t]*(?:string|number|boolean|void|any)\b"),
        re.compile(
            r"^[ \t]*export[ \t]+(?:default[ \t]+)?(?:class|function|const)\b",
            re.MULTILINE,
        ),
    ],
    "go": [
        re.compile(r"^[ \t]*func[ \t]+(?:\([^)]*\)[ \t]*)?\w+[ \t]*\(", re.MULTILINE),
        re.compile(r"^[ \t]*package[ \t]+\w+[ \t]*$", re.MULTILINE),
        re.compile(r"^[ \t]*import[ \t]+\(", re.MULTILINE),
        re.compile(r"^[ \t]*type[ \t]+\w+[ \t]+(?:struct|interface)\b", re.MULTILINE),
        re.compile(r":=[ \t]*\S"),
    ],
    "rust": [
        re.compile(r"^[ \t]*(?:pub[ \t]+)?(?:async[ \t]+)?fn[ \t]+\w+", re.MULTILINE),
        re.compile(r"^[ \t]*use[ \t]+[\w:]+", re.MULTILINE),
        re.compile(r"^[ \t]*impl\b", re.MULTILINE),
        re.compile(r"\blet[ \t]+mut\b"),
        re.compile(r"^[ \t]*(?:pub[ \t]+)?(?:struct|enum|trait)[ \t]+\w+", re.MULTILINE),
    ],
    "java": [
        re.compile(r"^[ \t]*import[ \t]+[\w.]+;", re.MULTILINE),
        re.compile(r"^[ \t]*package[ \t]+[\w.]+;", re.MULTILINE),
        re.compile(
            r"\b(?:public|private|protected)[ \t]+(?:static[ \t]+)?[\w<>\[\]]+[ \t]+\w+[ \t]*\("
        ),
        re.compile(
            r"^[ \t]*(?:public|private|protected)[ \t]+(?:final[ \t]+)?class[ \t]+\w+",
            re.MULTILINE,
        ),
    ],
    "perl": [
        re.compile(r"^[ \t]*sub[ \t]+\w+", re.MULTILINE),
        re.compile(r"^[ \t]*use[ \t]+strict\b", re.MULTILINE),
        re.compile(r"\bmy[ \t]+[$@%]\w+"),
        re.compile(r"^[ \t]*package[ \t]+[\w:]+;", re.MULTILINE),
    ],
}

# A grep/ripgrep result row: "path:line:content". The reversible pass folds
# these under one path heading instead of repeating the path on every row,
# which is worth about 39% on real search output.
# The path must contain a "." or "/" so a timestamp does not qualify:
# "2026-09-02T11:20:00Z" splits as "...T11" + ":20:" and matched a bare
# path:line: pattern, which sent colourised logs to the search fold.
_SEARCH_LINE_RE = re.compile(r"^[^\s:]*[./][^\s:]*:\d+:")

# A delimited data row: the same separator at least twice on one line.
# Requiring two occurrences is what keeps an English sentence containing a
# single comma out of the tabular bucket.
_TABULAR_SEPARATORS = (",", ";", "\t", "|")

# A config assignment: a bare key, then ":" or "=", at the start of a line or
# after indentation. Matches YAML, TOML, INI and .env alike. The value may be
# empty, which is how a YAML parent key looks.
_CONFIG_LINE_RE = re.compile(r"^[ \t]*[A-Za-z_][\w.\-]*[ \t]*[:=](?![/=])")

# Markdown structure: headings, bullets, fences, links, tables, quotes.
_MARKDOWN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^#{1,6}[ \t]+\S", re.MULTILINE),
    re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+\S", re.MULTILINE),
    re.compile(r"^```", re.MULTILINE),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),
    re.compile(r"^[ \t]*\|.*\|[ \t]*$", re.MULTILINE),
    re.compile(r"^>[ \t]+\S", re.MULTILINE),
]


def score_languages(content: str) -> tuple[str, int]:
    """Vote on the most likely programming language.

    Args:
        content: Content to score.

    Returns:
        ``(language, score)``, where score is the number of distinct marker
        patterns that matched. A score of 0 means "probably not code" and the
        language is meaningless in that case.
    """
    best_language = ""
    best_score = 0
    for language, patterns in _LANGUAGE_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern.search(content))
        if score > best_score:
            best_language, best_score = language, score
    return best_language, best_score


def score_markdown(content: str) -> int:
    """Count how many distinct Markdown structures appear in ``content``."""
    return sum(1 for pattern in _MARKDOWN_PATTERNS if pattern.search(content))


def score_log(content: str) -> int:
    """Count lines that carry a log level in level position.

    Counting lines rather than testing for one match matters: a stack trace
    quoted inside a log is full of source code, so "is there any code here"
    and "is there any log here" both answer yes. The number of log-shaped
    lines is what separates a log that quotes code from a source file.

    Args:
        content: Content to score.

    Returns:
        Number of matching lines.
    """
    return len(_LOG_LINE_RE.findall(content))


# Line-level evidence, one predicate per type. Scoring every type on the same
# unit -- "what fraction of lines look like this?" -- is what lets them be
# compared. Trying types in a fixed order and taking the first hit is what
# classified a log quoting a traceback as source code, and a release note
# quoting a patch as a diff: in both cases the minority signal won because it
# was checked first.
# Only unambiguous diff metadata counts. Added/removed lines are deliberately
# NOT scored: "- fixed a bug" is a Markdown bullet, and counting it as a
# removal made release notes quoting a patch outscore Markdown itself. A real
# patch carries plenty of headers, so nothing is lost by ignoring the bodies.
_DIFF_LINE_RE = re.compile(
    r"^(?:diff --git |index [0-9a-f]{7,}|--- |\+\+\+ |@@ -\d)"
)
_MARKDOWN_LINE_RE = re.compile(
    r"^(?:[ \t]*#{1,6}[ \t]+\S|[ \t]*(?:[-*+]|\d+[.)])[ \t]+\S|[ \t]*```|"
    r"[ \t]*\|.*\||>[ \t]+\S)"
)


def _code_line_patterns() -> list[re.Pattern[str]]:
    """Every language marker that is anchored to a line start.

    Unanchored markers (``\\bself\\b``, ``:=``) are deliberately excluded from
    line scoring: they match inside stack traces and prose, which is exactly
    how a traceback used to be read as source.
    """
    return [
        pattern
        for patterns in _LANGUAGE_PATTERNS.values()
        for pattern in patterns
        if pattern.pattern.startswith("^")
    ]


_CODE_LINE_PATTERNS = _code_line_patterns()


def score_tabular(lines: list[str]) -> float:
    """Fraction of lines that share the modal field count of a delimiter.

    Counting delimiters per line is not enough: "The report, which was long,
    covered many things" has two commas and would score as a data row. What
    separates a table from prose is that every row has the *same* number of
    fields. Scoring the modal count, not the raw count, is what makes that
    distinction.

    Args:
        lines: Non-blank lines of the content.

    Returns:
        0.0-1.0, the share of lines agreeing on a field count of two or more.
    """
    best = 0.0
    for separator in _TABULAR_SEPARATORS:
        counts = [line.count(separator) for line in lines]
        populated = [c for c in counts if c >= 1]
        if len(populated) < 2:
            continue
        modal = max(set(populated), key=populated.count)
        agreeing = sum(1 for c in counts if c == modal)
        score = agreeing / len(lines)

        # Separator counts alone cannot tell a table from prose: an English
        # sentence with a comma in it looks like a two-column row, and a
        # sentence with two commas looks like a three-column one. What tables
        # have and prose does not is a key column -- one field position that
        # is short and unspaced on every row ("id", "ACC-0001", "2026-09-02").
        if not _has_key_like_column(lines, separator, modal + 1):
            continue

        best = max(best, score)
    return best


def _has_key_like_column(lines: list[str], separator: str, columns: int) -> bool:
    """True when some column is short and unspaced on every row.

    Args:
        lines: Non-blank lines of the content.
        separator: The delimiter under test.
        columns: Modal field count, so only real column positions are checked.

    Returns:
        Whether any single column looks like a key or code column.
    """
    for position in range(columns):
        for line in lines:
            fields = line.split(separator)
            if position >= len(fields):
                break
            field = fields[position].strip().strip('"')
            if not field or len(field) > 24 or " " in field:
                break
        else:
            return True
    return False


def score_types(content: str) -> dict[ContentType, float]:
    """Score every candidate type on one comparable scale.

    Each score is the fraction of non-blank lines that carry evidence for that
    type, so the scores can be ranked against each other. A line may be
    evidence for more than one type; the scores are not a probability
    distribution and do not sum to 1.

    JSON is absent here because it is not scored: it either parses or it does
    not, and a parse is proof rather than evidence.

    Args:
        content: Content to score.

    Returns:
        Fraction of lines matching, per content type.
    """
    lines = [line for line in content.split("\n") if line.strip()]
    if not lines:
        return {}

    counts = {ContentType.DIFF: 0, ContentType.LOG: 0, ContentType.CODE: 0,
              ContentType.MARKDOWN: 0, ContentType.TABULAR: 0,
              ContentType.CONFIG: 0, ContentType.SEARCH: 0}
    for line in lines:
        if _DIFF_LINE_RE.match(line):
            counts[ContentType.DIFF] += 1
        if _LOG_LINE_RE.match(line):
            counts[ContentType.LOG] += 1
        if _MARKDOWN_LINE_RE.match(line):
            counts[ContentType.MARKDOWN] += 1
        if any(pattern.match(line) for pattern in _CODE_LINE_PATTERNS):
            counts[ContentType.CODE] += 1
        if _CONFIG_LINE_RE.match(line):
            counts[ContentType.CONFIG] += 1
        if _SEARCH_LINE_RE.match(line):
            counts[ContentType.SEARCH] += 1

    total = len(lines)
    scores = {kind: count / total for kind, count in counts.items()}
    scores[ContentType.TABULAR] = score_tabular(lines)
    return scores


class RuleDetector:
    """Deterministic, dependency-free content type detector.

    Example:
        >>> RuleDetector().detect('{"users": [{"id": 1}]}').content_type
        <ContentType.JSON: 'json'>
        >>> RuleDetector().detect("def f(x):\\n    return x\\n").language
        'python'
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize the detector.

        Args:
            min_confidence: Results scoring below this are downgraded to
                ``ContentType.UNKNOWN``, which routes to the no-op handler.
        """
        self.min_confidence = min_confidence

    def detect(self, content: str) -> DetectionResult:
        """Classify ``content``.

        Args:
            content: The content to analyze.

        Returns:
            DetectionResult with type, confidence and (for code) language.
        """
        if not content or not content.strip():
            return DetectionResult(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                raw_label="empty",
            )

        result = self._classify(content)
        if result.confidence < self.min_confidence:
            return DetectionResult(
                content_type=ContentType.UNKNOWN,
                confidence=result.confidence,
                raw_label=result.raw_label,
                metadata={"reason": "below min_confidence"},
            )
        return result

    def detect_batch(self, contents: list[str]) -> list[DetectionResult]:
        """Classify several contents.

        Args:
            contents: Content strings to analyze.

        Returns:
            DetectionResults in the same order as the input.
        """
        return [self.detect(c) for c in contents]

    def _classify(self, content: str) -> DetectionResult:
        """Run the ordered rules. See the module docstring for the order."""
        stripped = content.strip()

        # 1. JSON. The prefix check is a cheap gate; the parse is what lets
        # this branch claim full confidence.
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
                return DetectionResult(
                    content_type=ContentType.JSON,
                    confidence=1.0,
                    raw_label="json",
                )
            # RecursionError, not just ValueError: json.loads recurses per nesting
            # level, and a payload like "[" * 5000 blows the stack. It is a
            # RuntimeError, so a ValueError guard misses it and the crash reaches
            # the caller -- for a gateway, that is a hostile payload taking the
            # process down rather than merely compressing badly.
            except (json.JSONDecodeError, ValueError, RecursionError):
                pass

        # 2. Everything else is scored on one scale and the best fit wins.
        # No type is checked "first": a log quoting a traceback and a release
        # note quoting a patch both used to be misread because the minority
        # signal was tested before the majority one.
        scores = score_types(content)
        if not scores:
            return DetectionResult(ContentType.TEXT, 0.5, "text")

        # A config file contains no def, class or import. Source that carries
        # type annotations does -- and "min_span_length: int = 50" is
        # indistinguishable from a config assignment line by line, so a
        # dataclass-heavy module out-scored itself as config. Language markers
        # settle it: real YAML has none, a Python module has several.
        code_language, code_markers = score_languages(content)
        if code_markers >= 2:
            scores[ContentType.CONFIG] = 0.0
            # Line share under-counts code: most lines of a real module are
            # bodies, comments and docstrings, so only a fraction carry a
            # line-anchored marker. The number of *distinct* markers is the
            # honest measure of "this is source", and two is already the floor
            # that keeps a lone `self` in a stack trace from qualifying.
            # Not when the content carries diff headers, though: a patch of
            # a Python file is full of Python markers, and "diff --git" is
            # unambiguous where a marker count is only suggestive.
            if not scores[ContentType.DIFF]:
                scores[ContentType.CODE] = max(
                    scores[ContentType.CODE], min(code_markers / 5, 0.9)
                )

        winner = max(scores, key=lambda kind: scores[kind])
        share = scores[winner]

        # Below this share of lines, no type is making a real case and the
        # content is prose that happens to contain a bracket or a keyword.
        if share < 0.15:
            return DetectionResult(
                content_type=ContentType.TEXT,
                confidence=0.5,
                raw_label="text",
                metadata={"scores": {k.value: round(v, 3) for k, v in scores.items()}},
            )

        language = None
        raw_label = winner.value
        if winner is ContentType.CODE:
            language = code_language
            raw_label = language or "code"

        return DetectionResult(
            content_type=winner,
            # Confidence tracks how much of the content agrees, so a file that
            # is 90% log lines reports higher than one that is 20%.
            confidence=min(0.5 + share / 2, 0.95),
            raw_label=raw_label,
            language=language,
            metadata={"scores": {k.value: round(v, 3) for k, v in scores.items()}},
        )

        # 6. Floor.
        return DetectionResult(
            content_type=ContentType.TEXT,
            confidence=0.5,
            raw_label="text",
        )
