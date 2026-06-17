"""Result tree types for the classpatch pipeline.

A :class:`ResultNode` represents a single model invocation. The executor
populates a tree of these per pipeline run, rooted at the entry model.
Each node owns:

- which model produced it,
- the absolute time range of its input segment within the source file,
- the predictions the model emitted (potentially across multiple internal
  chunks, each prediction carrying its own sub-range),
- the parent prediction(s) that triggered the invocation (empty for the
  root),
- the downstream nodes produced when matching routing rules fired.

The tree is the canonical record of "what did each model say about each
segment", and is the primary input to the survey / aggregation layer via
:meth:`ResultNode.iter_rows`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

from classpatch.base import Prediction


# ---------------------------------------------------------------------------
# Trigger + flat row types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    """A parent prediction that caused a downstream model invocation.

    Multiple ``Trigger`` entries on a single node mean several rules from the
    same parent matched the same chunk and were deduplicated into one
    invocation (e.g. SSLAM emitting both ``Bird`` and ``Chirp, tweet`` on the
    same 10 s window, both routing to BirdNET).
    """

    from_model: str
    label: str
    score: float
    time_range: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "from_model": self.from_model,
            "label": self.label,
            "score": self.score,
            "time_range": list(self.time_range),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Trigger":
        return cls(
            from_model=d["from_model"],
            label=d["label"],
            score=float(d["score"]),
            time_range=tuple(d["time_range"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ResultRow:
    """One prediction, flattened with enough context for tabular analysis.

    Produced by :meth:`ResultNode.iter_rows` and consumed by the survey
    layer to group by ``(source_path, label)`` etc.
    """

    source_path: Path
    model: str
    label: str
    score: float
    start_sec: float
    end_sec: float
    depth: int  # 0 = root (entry) model invocation; +1 per routing hop


# ---------------------------------------------------------------------------
# ResultNode
# ---------------------------------------------------------------------------


@dataclass
class ResultNode:
    """One model invocation in a pipeline run.

    Mutable on purpose: the executor builds it incrementally as classification
    completes and routing rules fire. Once a run finishes it can be treated
    as effectively immutable.
    """

    model: str
    source_path: Path
    segment_time_range: tuple[float, float]
    predictions: list[Prediction] = field(default_factory=list)
    triggers: list[Trigger] = field(default_factory=list)
    children: list["ResultNode"] = field(default_factory=list)

    # ----- mutation helpers --------------------------------------------

    def add_child(self, child: "ResultNode") -> "ResultNode":
        """Append ``child`` to :attr:`children` and return it."""
        self.children.append(child)
        return child

    def add_trigger(self, trigger: Trigger) -> None:
        """Record an additional parent prediction that caused this node.

        Used by the executor's dedup logic: when two rules from the same
        parent route to the same target on the same chunk, the second one
        merges its trigger into the existing node rather than spawning a
        duplicate invocation.
        """
        self.triggers.append(trigger)

    # ----- traversal ---------------------------------------------------

    def walk(self, depth: int = 0) -> Iterator[tuple["ResultNode", int]]:
        """Depth-first walk yielding ``(node, depth)`` pairs.

        ``depth`` is 0 at the root and increases by 1 for each routing hop.
        """
        yield self, depth
        for c in self.children:
            yield from c.walk(depth + 1)

    def iter_rows(
        self,
        *,
        min_score: float = 0.0,
        models: Iterator[str] | tuple[str, ...] | None = None,
    ) -> Iterator[ResultRow]:
        """Yield one :class:`ResultRow` per prediction in the tree.

        ``min_score`` filters out low-confidence detections; ``models``
        restricts to a subset of models, useful for narrowing aggregation
        to a single specialist's vocabulary.
        """
        model_filter = set(models) if models is not None else None
        for node, depth in self.walk():
            if model_filter is not None and node.model not in model_filter:
                continue
            for p in node.predictions:
                if p.score < min_score:
                    continue
                yield ResultRow(
                    source_path=node.source_path,
                    model=node.model,
                    label=p.label,
                    score=p.score,
                    start_sec=p.time_range[0],
                    end_sec=p.time_range[1],
                    depth=depth,
                )

    def predictions_by_chunk(self) -> dict[tuple[float, float], list[Prediction]]:
        """Group this node's predictions by their internal time_range.

        Each group is sorted by score descending; the dict itself is ordered
        by chunk start time. Used by :func:`render`.
        """
        groups: dict[tuple[float, float], list[Prediction]] = {}
        for p in self.predictions:
            groups.setdefault(p.time_range, []).append(p)
        for chunk_preds in groups.values():
            chunk_preds.sort(key=lambda p: p.score, reverse=True)
        return dict(sorted(groups.items(), key=lambda kv: kv[0]))

    # ----- persistence -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "source_path": str(self.source_path),
            "segment_time_range": list(self.segment_time_range),
            "predictions": [
                {
                    "label": p.label,
                    "score": p.score,
                    "time_range": list(p.time_range),
                }
                for p in self.predictions
            ],
            "triggers": [t.to_dict() for t in self.triggers],
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "ResultNode":
        return cls(
            model=d["model"],
            source_path=Path(d["source_path"]),
            segment_time_range=tuple(d["segment_time_range"]),  # type: ignore[arg-type]
            predictions=[
                Prediction(
                    label=p["label"],
                    score=float(p["score"]),
                    time_range=tuple(p["time_range"]),
                )
                for p in d.get("predictions", [])
            ],
            triggers=[Trigger.from_dict(t) for t in d.get("triggers", [])],
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )

    def save(self, path: Path | str, *, indent: int = 2) -> Path:
        """Persist this tree to ``path`` as JSON. Returns the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=indent)
        path.write_text(payload + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ResultNode":
        """Load a tree previously written by :meth:`save`."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ----- summaries ---------------------------------------------------

    def __repr__(self) -> str:
        start, end = self.segment_time_range
        return (
            f"ResultNode(model={self.model!r}, "
            f"range=[{start:.1f}-{end:.1f}s], "
            f"predictions={len(self.predictions)}, "
            f"children={len(self.children)})"
        )


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------


def render(
    node: ResultNode,
    *,
    top_k: int | None = 5,
    score_floor: float = 0.0,
) -> str:
    """Render a result tree as a human-readable indented string.

    ``top_k`` and ``score_floor`` control how many predictions are shown
    *per chunk* in the printout — they don't change the underlying tree.
    Pass ``top_k=None`` to print every prediction above ``score_floor``.

    Children are attached underneath the chunk in their parent that triggered
    them, so the routing flow reads top-to-bottom and left-to-right::

        sslam [0.0-1800.0s]
          [0.0-10.0s] Speech 0.852, Vehicle 0.490, ...
          [10.0-20.0s] Bird 0.710, ...
            birdnet [10.0-20.0s]  <- sslam Bird 0.71
              [12.0-15.0s] Northern Cardinal 0.830
    """
    lines: list[str] = []
    _render_into(
        node,
        lines,
        indent=0,
        top_k=top_k,
        score_floor=score_floor,
    )
    return "\n".join(lines)


def _render_into(
    node: ResultNode,
    lines: list[str],
    *,
    indent: int,
    top_k: int | None,
    score_floor: float,
) -> None:
    pad = "  " * indent
    start, end = node.segment_time_range

    trigger_info = ""
    if node.triggers:
        bits = ", ".join(f"{t.from_model} {t.label} {t.score:.2f}" for t in node.triggers)
        trigger_info = f"  <- {bits}"
    lines.append(f"{pad}{node.model} [{start:.1f}-{end:.1f}s]{trigger_info}")

    chunks = node.predictions_by_chunk()

    children_by_chunk: dict[tuple[float, float] | None, list[ResultNode]] = {}
    for child in node.children:
        key = child.triggers[0].time_range if child.triggers else None
        children_by_chunk.setdefault(key, []).append(child)

    rendered_chunks: set[tuple[float, float]] = set()
    for chunk_range, preds in chunks.items():
        filtered = [p for p in preds if p.score >= score_floor]
        if top_k is not None:
            filtered = filtered[:top_k]
        if filtered:
            cs, ce = chunk_range
            label_str = ", ".join(f"{p.label} {p.score:.3f}" for p in filtered)
            lines.append(f"{pad}  [{cs:.1f}-{ce:.1f}s] {label_str}")
        for child in children_by_chunk.pop(chunk_range, []):
            _render_into(
                child,
                lines,
                indent=indent + 2,
                top_k=top_k,
                score_floor=score_floor,
            )
        rendered_chunks.add(chunk_range)

    # Children whose triggering chunk wasn't in this node's predictions (rare,
    # but possible if a custom executor pre-filters predictions). Print them
    # below the chunks rather than dropping them silently.
    for chunk_range, orphan_list in children_by_chunk.items():
        if chunk_range is not None:
            cs, ce = chunk_range
            lines.append(f"{pad}  [{cs:.1f}-{ce:.1f}s] (no surviving predictions)")
        for child in orphan_list:
            _render_into(
                child,
                lines,
                indent=indent + 2,
                top_k=top_k,
                score_floor=score_floor,
            )


__all__ = [
    "ResultNode",
    "ResultRow",
    "Trigger",
    "render",
]
