# compresskit

Structure-preserving, rule-based context compression. A vendored subset of
[Headroom](https://github.com/headroom-ai/headroom)'s `headroom.compression`
package, stripped to run on the Python standard library alone.

Built for environments where you can copy a folder into a repo but cannot
`pip install` a package or download model weights.

```
No Magika.  No Kompress.  No HuggingFace.  No ONNX.  No network calls.
Just regexes, a JSON tokenizer, and Shannon entropy.
```

## The idea

Truncating a 40 KB JSON payload to fit a context window costs the model the
schema — it can no longer see which fields exist. Truncating a source file
costs it the function signatures.

So don't truncate first. **Work out which characters carry the structure, and
only shrink the rest.**

```
content ──▶ compact ──▶ detect ──▶ mask ──▶ reduce ──▶ compressed
          (reversible) (rules)  (handler) (reducers)
```

The first pass is **reversible**: it strips ANSI colour, folds repeated lines
and blocks, and regroups grep output, verifying the round trip before it
returns anything. It cannot lose meaning, and it is a no-op on content that
does not repeat. Everything after it is lossy, so the free savings are taken
first -- folding repeats in text that truncation has already mangled finds
nothing.

| Content  | Preserved                                          | Reduced                    |
| -------- | -------------------------------------------------- | -------------------------- |
| JSON     | keys, brackets, booleans, nulls, short values, IDs | long string values         |
| Code     | imports, signatures, class/type declarations, decorators | function bodies      |
| Diff     | file and hunk headers, every `+`/`-` line          | context lines              |
| Log      | WARN/ERROR lines and their stack frames, timestamps, levels | INFO/DEBUG message bodies |
| Markdown | headings, code fences, list markers, table rows    | paragraph prose            |
| Tabular  | header row, delimiters, ids, first rows, short cells | long free-text cells      |
| Config   | every key, sections, comments, short values, secrets | long values               |
| Text     | high-entropy words (secrets, UUIDs, hashes)        | everything else            |

Secrets, API keys, UUIDs and hashes are preserved everywhere, in every content
type. They cannot be reconstructed from context, so losing them loses
information outright rather than losing verbosity.

## Install

Copy the `compresskit/` folder into your project. That is the whole procedure.

```
your_project/
├── compresskit/          <-- copy this
│   ├── __init__.py
│   ├── compressor.py
│   ├── detector.py
│   ├── masks.py
│   ├── reducers.py
│   └── handlers/
└── your_code.py
```

Requires Python 3.10 or newer (for `X | Y` type syntax).

## Use

```python
from compresskit import compress

result = compress(payload)

result.compressed           # send this to the model
result.original             # reduction is one-way; this is the route back
result.content_type         # ContentType.JSON
result.savings_percentage   # 52.9
result.preservation_ratio   # 0.35 -- what the mask claimed as structural
```

Reuse a compressor for anything hot — the handlers hold compiled patterns:

```python
from compresskit import Compressor, CompressorConfig, ReducerConfig

compressor = Compressor(
    config=CompressorConfig(
        reducer=ReducerConfig(target_ratio=0.5),   # cut to ~50%, not ~30%
        min_content_length=500,                     # leave small payloads alone
    )
)

for blob in blobs:
    print(compressor.compress(blob).compressed)
```

Force a content type, or a language, when you already know:

```python
from compresskit import ContentType

compressor.compress(blob, content_type=ContentType.JSON)
compressor.compress(source, language="rust")
```

## Reversible compaction

Everything in the masking pipeline is lossy. `compresskit.lossless` is not: each
transform has an exact inverse, and `compact_lossless` runs the round trip
before returning. If the inverse does not reproduce the input, or the result is
not smaller, you get your input back unchanged. It never raises.

Measured on realistic content:

| Content | Saved |
| --- | ---: |
| Colourised CI log with a repeated warning burst | 69% |
| grep / ripgrep output | 39% |
| git diff | 25% |
| Log with all-distinct messages | 0% |
| JSON payload | 0% |

The zeroes are correct answers, not failures -- nothing repeats, so nothing
folds. Because the risk is zero, it runs by default; set
`CompressorConfig(use_lossless=False)` to skip it.

`result.lossless_chars_saved` reports what it removed before any lossy work
began. One honest exception to "lossless": ANSI colour codes are dropped
permanently, since they carry no meaning to a model. The guarantee is about
meaning, not bytes.

## Mixed content

A prompt is rarely one thing. It is instructions wrapping a JSON payload, or a
log quoting a traceback, or a README with a fenced example.

Detecting one type for the whole input and running one handler over it loses
whichever part was in the minority. A log with a JSON body in it is "a log", so
the JSON never reaches the JSON handler, and a `"message"` field disappears
into a ` ... ` marker while `"service"` -- identical on every line -- survives.

So the input is split first, and each section gets its own handler:

```
Answer using only the data.        -> prose   -> kept verbatim
2026-09-02 ERROR request failed:   -> log     -> LogStructureHandler
{"status": 500, "message": "..."}  -> JSON    -> JSONStructureHandler
```python                           -> code    -> CodeStructureHandler
def helper(rows): ...
```
QUESTION: what failed?             -> prose   -> kept verbatim
```

Several handlers, one prompt, one pass. Handlers fire only where their pattern
is present -- none are mandatory, and the order follows your input.

**Prose is preserved, not compressed.** In a mixed prompt the prose is your
instructions, and compressing them is how "Do not invent account numbers"
becomes "Do not inven ...", leaving the model bound by a rule it can no longer
read. Pass `preserve_prose=False` to `MixedContentHandler` to opt out.

One deliberate exception: when the splitter finds nothing but prose, the mixed
handler claims nothing, and plain text compresses exactly as it did before.
Prose is protected because there is something beside it worth protecting it
from -- not on its own account.

## Tuning

Everything is a constructor argument; nothing is global.

| Knob                                        | Default | Effect                                                     |
| ------------------------------------------- | ------- | ---------------------------------------------------------- |
| `ReducerConfig.target_ratio`                | `0.3`   | Fraction of characters to keep in reducible spans          |
| `ReducerConfig.marker`                      | `" ... "` | Marks where a span's middle was cut. `""` removes it       |
| `ReducerConfig.dedupe_lines`                | `True`  | Collapse repeated consecutive lines — the big win on logs  |
| `CompressorConfig.min_content_length`       | `100`   | Below this, content is returned untouched                  |
| `CompressorConfig.min_span_length`          | `50`    | Below this, a reducible span is left untouched. Drop to 20-30 for logs |
| `CompressorConfig.use_lossless`             | `True`  | Reversible pre-pass. No reason to turn it off              |
| `CompressorConfig.use_entropy_preservation` | `True`  | Preserve high-entropy words wherever they appear           |
| `CompressorConfig.entropy_threshold`        | `0.85`  | Higher is more selective about what counts as a secret     |
| `JSONStructureHandler.short_value_threshold`| `20`    | String values this short are kept verbatim                 |
| `JSONStructureHandler.max_array_items_full` | `3`     | Array items past this are compressed harder                |
| `LogStructureHandler.severity_floor`        | `WARN`  | Lines at or above this level survive whole                 |
| `DiffStructureHandler.preserve_context`     | `False` | `True` keeps the patch applicable, at the cost of the ratio |
| `TabularStructureHandler.full_rows`         | `3`     | Data rows kept verbatim as worked examples                 |
| `ConfigStructureHandler.preserve_comments`  | `True`  | Config comments usually explain a choice; keep them        |

### Which knob to reach for

The four that matter, and what each is actually for:

- **`target_ratio`** — how much of each *reducible* span to keep. It never
  touches structure, so past about 0.2 the returns fall away against a floor of
  protected text. Start at 0.3, lower it, and stop when the answers change.
- **`marker`** — the model's only signal that text was removed. `""` buys about
  10 points of ratio and gives that signal up. `" ... "` is the sweet spot; the
  old `" ...[compressed]... "` default cost ~9 points for no extra meaning.
- **`min_span_length`** — the smallest gap worth compressing. Bulk JSON leaves
  big gaps, so 50 is right. Logs leave small ones (a 45-character message after
  a preserved timestamp) and yield **nothing at all** until you drop to 20-30.
- **`min_content_length`** — the library default of 100 is too low for a
  gateway. Use 2000+.

The UI's **offline** checkbox has no library equivalent: it only tells the UI to
skip its two API calls. Compression never calls out to anything.

Everything else is safe on its defaults. `use_lossless` and
`use_entropy_preservation` should stay on -- one is free, the other is what
keeps secrets in the payload.

## Swapping pieces out

The three stages are independent. Replace any of them without touching the others.

```python
# Your own reducer instead of the rule-based one
compressor = Compressor(reduce_fn=my_compressor.compress)

# A real tokenizer instead of the chars/4 estimate
compressor = Compressor(token_estimator=lambda s: len(enc.encode(s)))

# A handler for a format of your own
compressor.register_handler(ContentType.TEXT, MyFixedWidthRecordHandler())

# A different detector entirely
compressor = Compressor(detector=MyDetector())
```

A `reduce_fn` that raises is caught: the span is spliced back verbatim and a
warning is logged. A broken reducer costs you compression ratio, never content.

## Optional: tree-sitter

If `tree-sitter` and `tree-sitter-language-pack` happen to be available, the
code handler uses real ASTs for signature detection instead of regexes
(`confidence` 0.95 vs 0.70). If they are not — the expected case here — the
regex path runs and nothing else changes. There is no error and no
configuration to set.

```python
from compresskit import is_tree_sitter_available
is_tree_sitter_available()   # False in a locked-down environment
```

`tree-sitter` is a native parser library, not a model — no weights are
downloaded. Worth asking for if your environment allows compiled wheels.

## What changed from upstream Headroom

| Upstream                                     | Here                                              |
| -------------------------------------------- | ------------------------------------------------- |
| `MagikaDetector` — ONNX model, 100+ types    | `RuleDetector` — ordered regex/parse rules        |
| Kompress — learned compressor for spans      | `reducers.py` — line dedup, whitespace, truncation |
| CCR store for retrieving originals           | Dropped. `result.original` holds the input        |
| `UniversalCompressor` / `UniversalCompressorConfig` | `Compressor` / `CompressorConfig`          |
| `headroom.compression.*` absolute imports    | Relative imports, so the folder can be renamed    |
| Handlers for JSON and code only              | Adds diff, log and Markdown handlers              |
| Detected language re-derived by the handler  | Detector's language vote is passed through        |

`masks.py`, `handlers/base.py`, `handlers/json_handler.py` and
`handlers/code_handler.py` are carried over close to verbatim, so they stay
diffable against upstream if you ever want to pull fixes across.

## Honest limits

- **Reduction is lossy and one-way.** There is no decompressor. Keep
  `result.original` if a user might need to see what was actually sent.
- **Token counts are estimates.** `estimate_tokens` assumes four characters per
  token. Good enough to report a ratio; not good enough to budget against a
  hard context limit. Pass a real tokenizer as `token_estimator` if you have one.
- **Detection is heuristic.** Only the JSON branch, which actually parses,
  reports confidence 1.0. A misroute costs compression ratio, not correctness —
  every handler's failure mode is "preserve more than necessary".
- **Compressed JSON still parses, but is not equivalent.** String values may be
  truncated. Never feed compressed output back into a system that consumes the
  real data.
- **Content below `min_content_length` is returned untouched.** Compressing a
  short string costs more in marker characters than it saves.
- **Diffs compress poorly by design.** A patch is mostly changed lines, and
  those are all preserved. Expect single-digit savings on a dense patch.

## Testing against a real endpoint

`examples/` holds an A/B harness that answers the two questions a ratio cannot:
did compression save **real** tokens, and did the answer survive?

```bash
cp examples/.env.example examples/.env    # fill in endpoint, key, deployment
python examples/ab_test.py --sample json --count-only
```

It sends the same question twice -- once with the original content, once with
the compressed content -- and reports `usage.prompt_tokens` from the service
for each. That is ground truth, and it is printed next to compresskit's own
chars/4 estimate so you can see how far the estimate drifts on your content.

```bash
python examples/ab_test.py --sample log                    # built-in samples: json, log, code
python examples/ab_test.py --file server.log --question "What failed and when?"
python examples/ab_test.py --file big.json --target-ratio 0.15 --show-compressed
```

Use `--count-only` first on anything large: it caps generation at one token, so
you measure the saving without paying to generate two full answers. Without it,
both answers are printed side by side -- read them. A 60% saving that loses the
answer is not a saving.

### The web UI

The quickest loop: paste a prompt, press Compare, read both answers.

```bash
python examples/serve.py     # opens http://localhost:8000
```

One page with a prompt box and two columns -- the prompt as sent and the
response, uncompressed on the left, compressed on the right, with the real token
counts from the service. A **target ratio** slider and **marker** field let you
tune compression without touching a file, and an **offline** checkbox runs
compression only, skipping both API calls while you experiment.

Two tabs above the results: **Side by side** shows each prompt in full;
**Diff** shows a GitHub-style split diff with line numbers, red for what
compression removed and green for what the model sees instead. Compression only
ever removes, so expect unchanged and shortened rows and almost no additions --
which makes it easy to see exactly what was lost.

A **max tokens** field caps the reply length (default 512). Raise it when you
ask for a long answer: if a reply hits the cap it gets cut mid-sentence, and you
would be comparing the cap rather than the compression.

Each Compare makes two API calls, one per arm. Offline mode is the free path.

The server is Python's stdlib `http.server` -- no Flask, no npm, nothing to
install. It binds `127.0.0.1` only, and your API key stays in the server
process: it is not in the page and not in any response the browser receives.
Do not put this on a shared host; it has no authentication, by design.

### Reports

Terminal output scrolls away. `--report` writes a file you can keep and share:

```bash
python examples/ab_test.py --sample json --report report.html
python examples/ab_test.py --file big.json --offline --report diff.html
python examples/ab_test.py --sample log --report run.json      # or .md
```

The HTML report has five sections:

1. **What it cost and saved** -- real `prompt_tokens` from the service next to
   compresskit's chars/4 estimate, and the drift between them.
2. **The answers** -- both replies side by side.
3. **What the mask protected** -- the original content with every preserved
   character highlighted green and every reducible one red. This is the clearest
   view of what the handler actually did.
4. **Prompt diff** -- line-level, uncompressed on the left, compressed on the
   right.
5. **Raw prompts** -- both in full, collapsed.

It is self-contained: no CDN, no fonts, no scripts fetched, so it opens off a
`file://` URL on a machine with no network. The API key is never written to it.

`--offline` skips the API entirely and still produces sections 1, 3, 4 and 5 --
useful for tuning `--target-ratio` before spending any quota.

The client is stdlib `urllib`; no `openai`, `requests` or `python-dotenv`
needed. Behind a corporate proxy it already honours `HTTPS_PROXY`; for a
private CA, point `SSL_CERT_FILE` at your bundle. `examples/.env` is gitignored.

`examples/` is a test harness, not part of the library. Vendor `compresskit/`
alone; leave this behind unless you want it.

## Tests

202 tests, no third-party dependencies beyond `pytest` itself. The mask, JSON
handler and code handler suites are Headroom's own, carried over unchanged
except for import paths.

```bash
cd compresskit
python -m pytest tests -q
```

Eight tree-sitter tests skip when the parser is unavailable.
