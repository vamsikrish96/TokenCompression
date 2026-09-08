# Integrating compresskit into a model gateway

This is the deployment guide for wiring `compresskit` into a **pass-through
model gateway** — a proxy that sits between callers (agents, tools, services)
and an LLM endpoint, with no control over what a caller puts in a prompt and
no reliable signal for what the caller intends to do with it (navigate code
vs. edit it, for example). Every recommendation below is written for that
specific architecture, not for a tool that reads its own prompts.

If you want a general tour of the library instead — install, every handler,
every knob — see `README.md` in the repo root. This document only covers
what changes when you're the thing every prompt passes through.

## 1. This directory is the whole dependency

Everything you need is inside this folder (`compresskit/compresskit/`) —
`LICENSE`, `NOTICE`, and `py.typed` sit alongside the code for exactly this
reason: copy the folder into your source tree and it's a complete, standalone
package.

**Pure standard library.** Every module in here imports only from Python's
standard library — `re`, `json`, `dataclasses`, `enum`, `typing`,
`collections.abc`, `abc`, `threading`, `pathlib`, `math`, `logging`. No
`pip install` step, no network call, no model weights.

**tree-sitter is the one optional extra**, used only by the code handler for
AST-accurate parsing. It's imported lazily inside a function and wrapped in
`try/except ImportError` — if it isn't installed, the code handler falls back
to a regex-based parser automatically. Nothing breaks either way; you get a
slightly less precise signature boundary without it, not an error.

**No embedding SDK, no Azure client, no cloud dependency of any kind.**
Wherever this guide talks about embeddings, `compresskit` only ever sees a
plain Python callable — see §4. It has no idea what's behind it.

## 2. The one rule: compress the payload, never the instructions

A prompt a gateway forwards is almost always two things glued together: your
framing (system prompt, the caller's own instructions) and something bulky
pasted into it (a tool result, a log, a retrieved document, an API response).
The bulk is where the tokens and the redundancy are. The framing is small and
load-bearing — compress it and "Do not invent account numbers" can become "Do
not inven ...", leaving the model bound by an instruction it can no longer
read.

So: compress only the bulky, non-instructional part of what passes through
you. Never run `compress()` over a system prompt, a caller's own written
instructions, or anything else whose exact wording is the thing enforcing a
rule.

`gateway_integration.py`, in this same folder, is a runnable reference for
this pattern:

```bash
python compresskit/gateway_integration.py
```

It builds one `Compressor` at module scope, exposes a `compress_payload()`
function that fails open (a compression bug degrades your token bill, never
your answers), and shows the split between framing and payload in
`build_messages()`. Read it before writing your own integration — most of the
shape below is already there.

## 3. Quick start

```python
from compresskit import Compressor, CompressorConfig, ReducerConfig

compressor = Compressor(
    config=CompressorConfig(
        # The library default (100) is far too low for production traffic --
        # below a few thousand characters the marker text can cost more than
        # the compression saves. Tune this to your own traffic; 2,000 is a
        # reasonable starting floor.
        min_content_length=2_000,
        reducer=ReducerConfig(target_ratio=0.4),
    )
)

result = compressor.compress(payload)
result.compressed          # what you actually send onward
result.original             # unchanged input, for logging / "show me what was sent"
result.savings_percentage   # what it bought you
result.content_type         # ContentType.CODE, .JSON, .LOG, .DIFF, ...
result.handler_used         # which handler actually ran
```

Build the `Compressor` once and reuse it across requests — handlers hold
compiled regex patterns and, where tree-sitter is installed, cached parsers
per thread. Constructing a fresh one per request throws that away for no
reason.

## 4. Recommended configuration for a pass-through gateway

This is the section that matters most if you're deploying the exact
architecture this guide is written for. The defaults `compresskit` ships with
are tuned for a tool that knows its own intent (an agent compressing its own
context, say). A gateway doesn't have that luxury: it sees a chunk of code and
has no way to know whether the caller is about to read it or hand it to a
model that's about to edit it and return a diff. The settings below are the
answer that came out of working through that constraint.

### `preserve_logic=True` — the practical default for code

`CodeStructureHandler` defaults to `preserve_logic=False`: function bodies
(control flow, assignments, returns) are treated as compressible filler, kept
or cut by ratio like anything else. That's the right call when you know a
caller only wants a navigational summary — "what functions exist" — but wrong
the moment the caller is about to edit or explain what a function *does*,
where the body is exactly the material being asked about.

A gateway can't tell those two cases apart per request. When it can't, the
asymmetry in the mistake decides the default: guessing "keep the body" when
it wasn't needed costs some ratio; guessing "cut the body" when it was needed
can produce an edit built on code the model never actually saw. Set
`preserve_logic=True` unless you have a specific, reliable signal that a
given request is navigation-only.

This is not a `CompressorConfig` field — it lives on the handler, so you
register it explicitly:

```python
from compresskit import Compressor, CompressorConfig, ContentType, CodeStructureHandler

compressor = Compressor(config=CompressorConfig(...))
compressor.register_handler(
    ContentType.CODE,
    CodeStructureHandler(preserve_logic=True),
)
```

**If you also wire in the comment/docstring redundancy filter (§5), pass its
embedder into this same `CodeStructureHandler` call.** `register_handler`
replaces the handler `CompressorConfig` would otherwise have built with that
embedder already attached — omit it here and the feature silently goes dark
the moment you set `preserve_logic`, with no error. See §5 for the exact
call.

### The log relevance filter — safe to enable, self-contained

`log_relevance_embed_fn` drops whole low-relevance log lines, scored against
an anchor built from the log's *own* error/traceback material — nothing
external to supply, nothing a caller has to pass through your gateway
correctly for it to work. This is why it's the one relevance filter
recommended on by default here:

```python
config = CompressorConfig(
    log_relevance_embed_fn=your_embed_fn,   # see §5
    log_relevance_threshold=0.35,
)
```

Fails open: no error/traceback material to anchor to, or the embedder call
fails for any reason, and the log passes through untouched by this filter —
never a broken or truncated log line.

### Search and prose relevance filters — leave these off

`search_relevance_embed_fn` and `prose_relevance_embed_fn` need an anchor
supplied per call (`search_query`, `prose_anchor`) — a grep result or a PR
description has no internal signal like a log's error does, so the anchor has
to come from outside. A gateway that just passes prompts through has no
reliable way to guarantee that anchor is present, correctly shaped, and
actually the right thing to compare against, on every single call. Leaving
these `None` (the default) is the correct choice for this architecture, not
a missing feature — the code is fully built and tested if a specific,
reliable anchor source ever exists for your deployment, but don't turn these
on without one.

### The comment/docstring redundancy filter — safe, and scoped tightly

`code_comment_redundancy_embed_fn` judges each comment or docstring against
its own adjacent code (self-contained, like the log filter — nothing to
supply per call) and leaves it compressible when it's judged redundant with
what the code already says. Wired the same way as `preserve_logic`, on the
handler:

```python
from compresskit import Compressor, CompressorConfig, ContentType, CodeStructureHandler

embed_fn = your_embed_fn  # see §5
config = CompressorConfig(
    code_comment_redundancy_embed_fn=embed_fn,
    code_comment_redundancy_threshold=0.5,
)
compressor = Compressor(config=config)
compressor.register_handler(
    ContentType.CODE,
    CodeStructureHandler(
        preserve_logic=True,
        comment_redundancy_embed_fn=embed_fn,
        comment_redundancy_threshold=0.5,
    ),
)
```

The bright line this filter is built to never cross: only comment and
docstring *characters* are ever candidates, never a code statement. A
comment not clearly tied to one function or class (a module banner, a
license header, a comment sitting between two methods) is protected outright
by rule rather than judged, since there's no single adjacent unit to compare
it to. Expected value is modest — 5-8%, not a major lever — worth having on,
not worth building a demo around by itself.

### `fold_constant_fields` — off unless you know the payload's use

Lifts a field that's identical across every JSON record or CSV row into a
leading note instead of repeating it in every row. Reversible, and it's the
single biggest lever in the library (30%+ on a realistic bulk API listing) —
but it changes the payload's *shape*. Harmless when a model is reading the
data to answer a question; risky when the payload is the template for the
model's own output, since a model shown reduced records may hand back
reduced records or invent the missing fields. `compresskit` can't tell those
two uses apart from the content alone — only you know which one a given
request is. Leave it `False` unless you specifically know the payload is
read-only context for this call.

### Summary table

| Setting | Recommended for a pass-through gateway | Why |
|---|---|---|
| `CodeStructureHandler(preserve_logic=...)` | `True` | Gateway can't tell navigation from editing intent; wrong guess on "cut" is the costlier mistake |
| `log_relevance_embed_fn` | set | Self-contained anchor, fails open, safe by design |
| `search_relevance_embed_fn` | `None` (off) | Needs a caller-supplied anchor a gateway can't guarantee |
| `prose_relevance_embed_fn` | `None` (off) | Same reason |
| `code_comment_redundancy_embed_fn` | set | Self-contained anchor, never touches a code statement |
| `fold_constant_fields` | `False` unless payload is known read-only | Changes payload shape |
| `min_content_length` | 2,000+ (not the library default of 100) | Marker overhead dominates on small payloads |

## 5. Wiring in your embedder

Every embedding-driven feature above takes the same type, defined in
`compresskit.relevance`:

```python
from compresskit.relevance import EmbedFn
# EmbedFn = Callable[[list[str]], list[Sequence[float]]]
```

`compresskit` never imports an embedding SDK and has no idea what's behind
this callable — Azure, OpenAI, a self-hosted model, a test double all work
identically as far as the package is concerned. `EmbedFn` (along with
`filter_log_by_relevance`, `filter_search_by_relevance`,
`filter_prose_by_relevance`, `cosine_similarity`, `RelevanceResult`) lives in
`compresskit.relevance` specifically and is **not** re-exported from the
top-level `compresskit` package — import it from there directly.

A minimal Azure OpenAI adapter:

```python
def azure_embed_fn(texts: list[str]) -> list[list[float]]:
    """Batch embed. Same order out as in. Called at most once per filter per request."""
    response = azure_client.embeddings.create(
        input=texts,
        model="your-embedding-deployment-name",
    )
    return [item.embedding for item in response.data]
```

Use your own credential handling here (key vault, managed identity,
whatever your platform standardizes on) — this is intentionally outside
`compresskit`'s boundary, not something the package should own. If you want a
fuller reference for the actual HTTP call, retry-on-transient-error handling,
etc., `examples/azure_client.py` in the repo root (outside this package
folder) has a complete, tested implementation written against `.env`-file
config for local development — port the pattern, not the file, into your own
credential-management code.

Then plug the one function in everywhere this guide's config examples show
`your_embed_fn` / `embed_fn`. That's the entire integration surface — one
function, reused across every filter that wants it.

## 6. Safety guarantees you can rely on

These are invariants, not just current behavior, and hold regardless of what
embedder you plug in or what config you choose:

- **Every embedding-driven filter fails open.** No anchor material, a missing
  embedder, or an embedder that raises: the content comes back unchanged.
  A broken relevance signal costs you the savings that filter would have
  found, never correctness.
- **Whole units only, never a partial cut.** A log line, a paragraph, a
  comment is kept verbatim or fully compressible — never sliced mid-line by
  an embedding decision.
- **No embedding-based filter ever scores a code statement.** The comment
  redundancy filter's only candidates are comment and docstring characters;
  every other line of real code is governed by parser-and-rule logic only,
  same as before any embedder is involved.
- **A reducer refuses a cut it can't make safely** rather than producing a
  fragment that reads as a typo instead of a recognizable omission — this
  applies whether or not embeddings are involved at all.

## 7. What's deliberately not in this folder

- **No Azure/OpenAI/embedding client.** See §5 — bring your own, matching
  the `EmbedFn` contract.
- **No web UI, no demo server.** `examples/serve.py` and `examples/ui.html`
  (outside this package folder) are for local testing and demos only, not
  something to deploy.
- **No CLI.** `gateway_integration.py` in this folder is a runnable
  reference for the integration pattern, not a command-line tool meant for
  production traffic.

If you're copying this package into another repository, `compresskit/` (this
folder) is the entire unit to take — nothing outside it is required for the
package to function.
