"""Survey / aggregation layer for classpatch result trees.

Takes one or more :class:`ResultNode` trees produced by :class:`Pipeline.run`
and produces per-file, per-label tallies suitable for the acoustic-ecology
survey use case ("how many of each sound event happened in each recording").

Counting model
--------------

There are two count semantics, both supported:

- **Detection count** (default): every prediction above ``min_score`` is one
  detection. A single 30 s bird call that fires three consecutive 10 s
  SSLAM windows shows up as three detections.
- **Merged event count** (opt-in via ``merge_gap_sec``): adjacent detections
  of the same label whose time gap is at most ``merge_gap_sec`` collapse
  into one event. The 30 s bird call above becomes one event when
  ``merge_gap_sec >= 0``.

Each :class:`LabelTally` always reports *both* numbers so callers can decide
which one they prefer at analysis time, regardless of the merging setting.

"Location" is intentionally not modelled here: tallies are keyed by source
audio path, and the user is expected to derive location from the filename
downstream.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from classpatch.results import ResultNode, ResultRow


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A single filtered prediction row (post ``min_score`` / model filter)."""

    source_path: Path
    model: str
    label: str
    start_sec: float
    end_sec: float
    score: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class Event:
    """One or more contiguous detections of the same label collapsed together.

    Produced by :func:`merge_detections`. ``detection_count`` records how many
    individual detections went into this event; ``mean_score`` / ``max_score``
    summarise their confidences.
    """

    source_path: Path
    model: str
    label: str
    start_sec: float
    end_sec: float
    detection_count: int
    mean_score: float
    max_score: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class LabelTally:
    """Aggregated stats for one ``(source_path, model, label)`` combination.

    Reports both detection count and merged event count so the consumer can
    pick whichever interpretation they want without re-aggregating. When the
    survey was computed with ``merge_gap_sec=None`` the two are equal.
    """

    source_path: Path
    model: str
    label: str
    detection_count: int
    event_count: int
    total_duration_sec: float
    mean_score: float
    max_score: float
    first_sec: float
    last_sec: float

    def to_record(self) -> dict:
        """Flat dict, suitable for CSV / pandas / JSON export."""
        return {
            "source_path": str(self.source_path),
            "filename": self.source_path.name,
            "model": self.model,
            "label": self.label,
            "detection_count": self.detection_count,
            "event_count": self.event_count,
            "total_duration_sec": round(self.total_duration_sec, 3),
            "mean_score": round(self.mean_score, 4),
            "max_score": round(self.max_score, 4),
            "first_sec": round(self.first_sec, 3),
            "last_sec": round(self.last_sec, 3),
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def collect_detections(
    trees: ResultNode | Mapping[Path, ResultNode] | Iterable[ResultNode],
    *,
    min_score: float = 0.0,
    models: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
) -> list[Detection]:
    """Flatten one or more result trees into a list of :class:`Detection`s.

    Accepts a single :class:`ResultNode`, a ``{Path: ResultNode}`` mapping
    (matching :meth:`Pipeline.run_many` output), or any iterable of trees.
    """
    nodes = _normalise_trees(trees)
    model_filter = set(models) if models is not None else None
    label_filter = set(labels) if labels is not None else None

    detections: list[Detection] = []
    for node in nodes:
        for row in node.iter_rows(min_score=min_score, models=model_filter):
            if label_filter is not None and row.label not in label_filter:
                continue
            detections.append(_detection_from_row(row))
    return detections


def merge_detections(
    detections: Iterable[Detection],
    *,
    merge_gap_sec: float = 0.0,
) -> list[Event]:
    """Collapse adjacent same-label detections into :class:`Event`s.

    Two detections of the same ``(source_path, model, label)`` merge when
    the gap between the previous detection's ``end_sec`` and the next one's
    ``start_sec`` is at most ``merge_gap_sec`` (overlapping detections, where
    the gap is negative, always merge).

    With ``merge_gap_sec=0`` only touching or overlapping detections merge,
    which is the right behavior for computing union duration when adjacent
    model windows partially overlap.
    """
    if merge_gap_sec < 0:
        raise ValueError(f"merge_gap_sec must be >= 0, got {merge_gap_sec!r}.")

    grouped: dict[tuple[Path, str, str], list[Detection]] = {}
    for d in detections:
        grouped.setdefault((d.source_path, d.model, d.label), []).append(d)

    events: list[Event] = []
    for (source, model, label), group in grouped.items():
        group.sort(key=lambda d: (d.start_sec, d.end_sec))

        run_start = group[0].start_sec
        run_end = group[0].end_sec
        run_scores: list[float] = [group[0].score]

        for d in group[1:]:
            gap = d.start_sec - run_end
            if gap <= merge_gap_sec:
                run_end = max(run_end, d.end_sec)
                run_scores.append(d.score)
            else:
                events.append(
                    _build_event(source, model, label, run_start, run_end, run_scores)
                )
                run_start, run_end = d.start_sec, d.end_sec
                run_scores = [d.score]

        events.append(
            _build_event(source, model, label, run_start, run_end, run_scores)
        )

    return events


def _build_event(
    source: Path,
    model: str,
    label: str,
    start: float,
    end: float,
    scores: list[float],
) -> Event:
    return Event(
        source_path=source,
        model=model,
        label=label,
        start_sec=start,
        end_sec=end,
        detection_count=len(scores),
        mean_score=sum(scores) / len(scores),
        max_score=max(scores),
    )


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


@dataclass
class Survey:
    """A population of :class:`LabelTally` rows plus the events that produced them.

    Typical use::

        survey = Survey.summarize(pipeline.run_many(paths, entry_model="sslam"))
        survey.to_csv("counts.csv")
    """

    tallies: list[LabelTally]
    events: list[Event]
    detections: list[Detection]
    merge_gap_sec: float | None

    # ----- construction ------------------------------------------------

    @classmethod
    def summarize(
        cls,
        trees: ResultNode | Mapping[Path, ResultNode] | Iterable[ResultNode],
        *,
        min_score: float = 0.0,
        models: Iterable[str] | None = None,
        labels: Iterable[str] | None = None,
        merge_gap_sec: float | None = None,
    ) -> "Survey":
        """One-stop entry point.

        Parameters
        ----------
        trees :
            Result tree(s) to aggregate. Same shapes accepted as by
            :func:`collect_detections`.
        min_score :
            Drop predictions below this confidence.
        models :
            If given, only count predictions emitted by these models. Useful
            for restricting to a single specialist's vocabulary (e.g. species
            labels from a downstream classifier) and ignoring the coarser
            labels of the upstream model.
        labels :
            Optional whitelist of labels to keep.
        merge_gap_sec :
            ``None`` (default) -> no merging; event_count == detection_count.
            ``0`` or positive -> merge adjacent same-label detections whose
            gap is at most this many seconds.
        """
        detections = collect_detections(
            trees, min_score=min_score, models=models, labels=labels
        )

        # Union-duration events are always computed with a 0-gap merge so
        # overlapping windows don't double-count toward total_duration_sec.
        union_events = merge_detections(detections, merge_gap_sec=0.0)

        if merge_gap_sec is None:
            count_events = union_events  # not actually used for counting
        else:
            count_events = merge_detections(detections, merge_gap_sec=merge_gap_sec)

        tallies = _build_tallies(
            detections,
            union_events=union_events,
            count_events=count_events,
            count_by_events=merge_gap_sec is not None,
        )
        return cls(
            tallies=tallies,
            events=count_events,
            detections=detections,
            merge_gap_sec=merge_gap_sec,
        )

    # ----- querying ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.tallies)

    def __iter__(self):
        return iter(self.tallies)

    def for_file(self, source_path: Path | str) -> list[LabelTally]:
        path = Path(source_path)
        return [t for t in self.tallies if t.source_path == path]

    def for_model(self, model: str) -> list[LabelTally]:
        return [t for t in self.tallies if t.model == model]

    def top(self, n: int = 10) -> list[LabelTally]:
        """``n`` tallies with the highest ``detection_count``, ties on ``max_score``."""
        return sorted(
            self.tallies,
            key=lambda t: (t.detection_count, t.max_score),
            reverse=True,
        )[:n]

    # ----- export -----------------------------------------------------

    def to_records(self) -> list[dict]:
        """List of flat dicts, ready for pandas / JSON / etc."""
        return [t.to_record() for t in self.tallies]

    def to_csv(self, path: Path | str) -> Path:
        """Write tallies to a CSV file. Returns the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        if not records:
            fieldnames = [
                "source_path", "filename", "model", "label",
                "detection_count", "event_count", "total_duration_sec",
                "mean_score", "max_score", "first_sec", "last_sec",
            ]
        else:
            fieldnames = list(records[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return path

    def __repr__(self) -> str:
        files = len({t.source_path for t in self.tallies})
        return (
            f"Survey(tallies={len(self.tallies)}, "
            f"files={files}, "
            f"events={len(self.events)}, "
            f"detections={len(self.detections)}, "
            f"merge_gap_sec={self.merge_gap_sec})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_trees(
    trees: ResultNode | Mapping[Path, ResultNode] | Iterable[ResultNode],
) -> list[ResultNode]:
    if isinstance(trees, ResultNode):
        return [trees]
    if isinstance(trees, Mapping):
        return list(trees.values())
    return list(trees)


def _detection_from_row(row: ResultRow) -> Detection:
    return Detection(
        source_path=row.source_path,
        model=row.model,
        label=row.label,
        start_sec=row.start_sec,
        end_sec=row.end_sec,
        score=row.score,
    )


def _build_tallies(
    detections: list[Detection],
    *,
    union_events: list[Event],
    count_events: list[Event],
    count_by_events: bool,
) -> list[LabelTally]:
    """Aggregate detections + events into per-key :class:`LabelTally` rows."""

    Key = tuple[Path, str, str]

    det_by_key: dict[Key, list[Detection]] = {}
    for d in detections:
        det_by_key.setdefault((d.source_path, d.model, d.label), []).append(d)

    union_by_key: dict[Key, list[Event]] = {}
    for e in union_events:
        union_by_key.setdefault((e.source_path, e.model, e.label), []).append(e)

    count_by_key: dict[Key, list[Event]] = {}
    for e in count_events:
        count_by_key.setdefault((e.source_path, e.model, e.label), []).append(e)

    tallies: list[LabelTally] = []
    for key, dets in det_by_key.items():
        source_path, model, label = key
        scores = [d.score for d in dets]
        starts = [d.start_sec for d in dets]
        ends = [d.end_sec for d in dets]

        union_events_for_key = union_by_key.get(key, [])
        total_duration = sum(e.duration_sec for e in union_events_for_key)

        if count_by_events:
            event_count = len(count_by_key.get(key, []))
        else:
            event_count = len(dets)

        tallies.append(
            LabelTally(
                source_path=source_path,
                model=model,
                label=label,
                detection_count=len(dets),
                event_count=event_count,
                total_duration_sec=total_duration,
                mean_score=sum(scores) / len(scores),
                max_score=max(scores),
                first_sec=min(starts),
                last_sec=max(ends),
            )
        )

    tallies.sort(key=lambda t: (str(t.source_path), t.model, -t.detection_count, t.label))
    return tallies


__all__ = [
    "Detection",
    "Event",
    "LabelTally",
    "Survey",
    "collect_detections",
    "merge_detections",
]
