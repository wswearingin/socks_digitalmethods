## What classpatch does

Pick a base classifier (e.g. SSLAM on AudioSet's 527 labels), declare
**label-driven routes** into specialist models (e.g. BirdNET on bird
windows, or any other downstream classifier), optionally **remap labels
into a custom ontology** without further inference, run recursively over a
folder of clips, and produce both a navigable result tree per file and a
flat per-`(file, model, label)` survey suitable for an acoustic-ecology
CSV.

```
audio file → base classifier (SSLAM, AudioSet 527 labels)
              │
              ├── predictions matching "any_bird"  → BirdNET → (further specialists)
              ├── predictions matching "any_train" → TrainNet (hypothetical)
              └── predictions remapped into a custom "ecology" ontology (no inference)
```

## Architecture

- **`classpatch.base`** — the `AudioClassifier` ABC plus the segment and
  prediction data types. Every model wrapper satisfies the same
  `load` / `unload` / `classify(segment, score_floor, top_k)` contract,
  so models compose freely. `Segment.time_range` is always absolute in
  the original source file so timestamps stay honest across routing hops.

- **`classpatch.routing`** — `RouteBook`, the user-facing container.
  Holds named **label groups** (e.g. `"any_bird"` → frozenset of AudioSet
  bird labels), **routing rules** (model A's matching prediction
  dispatches to model B), and **label routes** (model A's matching
  prediction emits a synthetic prediction under a user-defined ontology
  with no further inference). A fluent builder declares both shapes; the
  whole book persists as JSON.

- **`classpatch.pipeline`** — `Pipeline`, the BFS work-queue executor.
  Owns the model registry, consumes a `RouteBook`, and enforces per-rule
  score thresholds, per-rule depth limits, per-parent dedup, cycle
  protection, and a per-run audio cache. Returns a `ResultNode` tree per
  audio file.

- **`classpatch.results`** — the `ResultNode` tree plus `render`
  (a human-readable indented printout) and JSON `save` / `load`. Each
  node records its model, segment time range, predictions, triggers
  (parent predictions that fired the rule), and children.

- **`classpatch.survey`** — the aggregation layer. `Survey.summarize`
  rolls one or more result trees into per-`(file, model, label)` tallies
  with detection counts, optional merged-event counts, union duration,
  score statistics, and CSV export. Ontology names from label routes
  appear in the `model` column alongside real models, so a single survey
  can mix raw classifier output with remapped categories.

- **`classpatch.models`** — concrete wrappers. Today: `SSLAMClassifier`
  (the AudioSet-2M finetuned checkpoint) and `StubClassifier` (a
  fixed-prediction stub used to wire downstream specialists during
  testing).

## Install

The project uses [uv](https://github.com/astral-sh/uv) for environment and
dependency management.

```bash
uv sync                                   # core only
uv sync --extra sslam --extra notebook    # SSLAM + JupyterLab UX
uv sync --extra all                       # everything available
```

| Extra      | Purpose                        | Dependencies            |
|------------|--------------------------------|-------------------------|
| `notebook` | run the notebook scaffold      | `jupyterlab`            |
| `sslam`    | register `SSLAMClassifier`     | `transformers`, `timm`  |
| `all`      | everything                     | union of the above      |

## Quickstart

```python
from pathlib import Path
from classpatch import Pipeline, RouteBook, SSLAMClassifier, Survey, render

book = RouteBook()
book.define_group("any_bird", {
    "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet",
})

# Label route: remap general AudioSet labels into a survey-friendly ontology.
ecology = book.ontology("ecology")
ecology.relabel("sslam").when("any_bird").above(0.20).as_label("bird_activity")
ecology.relabel("sslam").when_label("Speech").above(0.40).as_label("human_voice")

book.save("cache/routes.json")  # persist for the next session

pipeline = Pipeline(book).register(SSLAMClassifier())

audio_files = sorted(Path("audio/in").glob("*.wav"))
trees = pipeline.run_many(audio_files, entry_model="sslam")

for tree in trees.values():
    print(render(tree, top_k=5))

survey = Survey.summarize(trees, min_score=0.20)
survey.to_csv("cache/survey.csv")
```

## Notebook

[`notebook.ipynb`](notebook.ipynb) ships the same workflow as a 12-cell
notebook with intermediate explanations: imports → model registration →
route definition → label-route definition → single-file run + tree render
→ batch run + survey CSV export.

## Running SSLAM offline

`SSLAMClassifier` loads weights via
`transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)`,
which means the first call requires network access to populate the
HuggingFace cache (weights, config, and the custom `modeling_*.py` files
the model card ships).

After the cache is warmed once, set these environment variables to force
fully offline runs:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

To keep the cache inside this repo (portable between machines), also set:

```bash
export HF_HOME=/path/to/SOCKS/cache/hf
```

Alternative: snapshot the model to a local directory once and pass that
path as `model_id`:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="ta012/SSLAM_AS2M_Finetuned",
    local_dir="models/sslam",
)
SSLAMClassifier(model_id="models/sslam")  # no network needed after this
```

## BirdNET wrapper

Implementation of a future birdnet wrapper is blocked by a version conflict. Birdnet relies on the TensorFlow library, which is incompatible with python 3.14. Google is working on an update.
