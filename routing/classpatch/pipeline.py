"""Pipeline executor for classpatch.

Walks the routing graph defined by a :class:`~classpatch.routing.RouteBook`:
classifies each segment with the entry model, then dispatches matching
predictions to downstream models per the rules. Builds a
:class:`~classpatch.results.ResultNode` tree as the audit/report output.

Execution model is a BFS work queue:

- One task per ``(model, segment, result_node)``.
- Within a task, every prediction is matched against every rule sourced
  from this task's model. Matching predictions enqueue downstream tasks.
- Dedup is per-parent: if two rules from the same parent match the same
  chunk and route to the same target, we record both triggers on a single
  child node and only enqueue one downstream task.
- Cycle and depth guards live on ``Segment.lineage``: a target model already
  present in the lineage is skipped, and lineage length cannot exceed the
  effective ``max_depth`` (per-rule override, else pipeline default).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from classpatch.base import (
    AudioClassifier,
    LineageStep,
    Prediction,
    Segment,
    load_mono_waveform,
)
from classpatch.results import ResultNode, Trigger
from classpatch.routing import LabelRoute, PayloadKind, RouteBook


DEFAULT_MAX_DEPTH = 3
DEFAULT_REPORT_TOP_K: int | None = 12
DEFAULT_REPORT_SCORE_FLOOR = 0.05


class PipelineError(RuntimeError):
    """Raised on configuration or runtime errors inside a Pipeline."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Task:
    model: str
    segment: Segment
    node: ResultNode


class _AudioCache:
    """Per-run cache of mono waveforms keyed by ``(source_path, target_sr)``.

    The same source file is typically needed at several sample rates (one
    per model) and across many time-window slices; loading once per
    (file, SR) pair lets all subsequent slices be cheap tensor views.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[Path, int], torch.Tensor] = {}

    def get(self, source: Path, target_sample_rate: int) -> torch.Tensor:
        key = (source, target_sample_rate)
        wave = self._cache.get(key)
        if wave is None:
            wave = load_mono_waveform(source, target_sample_rate)
            self._cache[key] = wave
        return wave


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Executes a multi-model routing pipeline on audio files.

    A pipeline owns:

    - a *model registry* (instances of :class:`AudioClassifier` keyed by
      ``classifier.name``),
    - a :class:`RouteBook` describing how predictions fan out downstream,
    - global execution config (``max_depth``, reporting thresholds).

    Routing thresholds (``min_score`` on each rule) are independent of
    reporting thresholds (``report_top_k`` / ``report_score_floor``). The
    executor always asks each model for its full prediction set, applies
    routing rules against that full set, and only filters predictions
    *before recording them on the result node*. This guarantees that any
    rule with ``min_score >= 0`` can fire regardless of how aggressive the
    report filter is.
    """

    def __init__(
        self,
        routes: RouteBook | None = None,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        report_top_k: int | None = DEFAULT_REPORT_TOP_K,
        report_score_floor: float = DEFAULT_REPORT_SCORE_FLOOR,
    ) -> None:
        self.routes: RouteBook = routes if routes is not None else RouteBook()
        self.max_depth = max_depth
        self.report_top_k = report_top_k
        self.report_score_floor = report_score_floor
        self._models: dict[str, AudioClassifier] = {}

    # ---- registry -----------------------------------------------------

    def register(self, classifier: AudioClassifier) -> "Pipeline":
        """Add a classifier to the registry (idempotent by ``classifier.name``)."""
        self._models[classifier.name] = classifier
        return self

    @property
    def models(self) -> Mapping[str, AudioClassifier]:
        return self._models

    # ---- run ----------------------------------------------------------

    def run(self, audio_path: Path | str, *, entry_model: str) -> ResultNode:
        """Process one audio file and return the root :class:`ResultNode`."""
        audio_path = Path(audio_path)
        self._validate(entry_model)

        cache = _AudioCache()
        entry = self._models[entry_model]
        wave = cache.get(audio_path, entry.target_sample_rate)
        duration = wave.numel() / entry.target_sample_rate
        root_segment = Segment(
            waveform=wave,
            sample_rate=entry.target_sample_rate,
            source_path=audio_path,
            time_range=(0.0, duration),
            lineage=[],
        )
        root_node = ResultNode(
            model=entry_model,
            source_path=audio_path,
            segment_time_range=root_segment.time_range,
        )

        queue: deque[_Task] = deque([_Task(entry_model, root_segment, root_node)])
        while queue:
            self._process_task(queue.popleft(), cache, queue)
        return root_node

    def run_many(
        self,
        audio_paths: Iterable[Path | str],
        *,
        entry_model: str,
    ) -> dict[Path, ResultNode]:
        """Run :meth:`run` on each path; returns a dict keyed by Path."""
        return {Path(p): self.run(p, entry_model=entry_model) for p in audio_paths}

    # ---- internals ----------------------------------------------------

    def _validate(self, entry_model: str) -> None:
        if entry_model not in self._models:
            raise PipelineError(
                f"Entry model {entry_model!r} is not registered."
            )
        for rule in self.routes.rules:
            if rule.from_model not in self._models:
                raise PipelineError(
                    f"Rule from_model={rule.from_model!r} is not a "
                    f"registered classifier."
                )
            for target in rule.to_models:
                if target not in self._models:
                    raise PipelineError(
                        f"Rule to_model={target!r} is not a "
                        f"registered classifier."
                    )
            if rule.match.group is not None and rule.match.group not in self.routes.groups:
                raise PipelineError(
                    f"Rule references undefined group "
                    f"{rule.match.group!r}."
                )
        for lroute in self.routes.label_rules:
            if lroute.from_model not in self._models:
                raise PipelineError(
                    f"Label route from_model={lroute.from_model!r} is not a "
                    f"registered classifier."
                )
            if lroute.target_ontology in self._models:
                raise PipelineError(
                    f"Label route target_ontology={lroute.target_ontology!r} "
                    f"collides with a registered classifier name; ontologies "
                    f"must use distinct names."
                )
            if lroute.match.group is not None and lroute.match.group not in self.routes.groups:
                raise PipelineError(
                    f"Label route references undefined group "
                    f"{lroute.match.group!r}."
                )

    def _process_task(
        self,
        task: _Task,
        cache: _AudioCache,
        queue: "deque[_Task]",
    ) -> None:
        clf = self._models[task.model]
        clf.load()
        # Ask the model for *all* predictions. Reporting filters happen
        # after routing so the two thresholds stay decoupled.
        full_preds = clf.classify(task.segment, score_floor=0.0, top_k=None)

        # Record (filtered) predictions on the node.
        task.node.predictions.extend(self._filter_for_report(full_preds))

        # Match each prediction against the model rules sourced from this
        # model and fan out, deduping within this parent.
        rules = self.routes.rules_for(task.model)
        if rules:
            children_index: dict[tuple[str, tuple[float, float]], ResultNode] = {}
            for pred in full_preds:
                for rule in rules:
                    labels = rule.match.resolve(self.routes.groups)
                    if pred.label not in labels:
                        continue
                    if pred.score < rule.min_score:
                        continue
                    for target_model in rule.to_models:
                        self._fire_rule(
                            task=task,
                            prediction=pred,
                            rule_payload=rule.payload,
                            rule_max_depth=rule.max_depth,
                            target_model=target_model,
                            cache=cache,
                            queue=queue,
                            children_index=children_index,
                        )

        # Match against label routes and synthesise ontology child nodes
        # inline -- no inference, no audio slicing, no queue.
        label_routes = self.routes.label_rules_for(task.model)
        if label_routes:
            self._process_label_routes(task, full_preds, label_routes)

    def _process_label_routes(
        self,
        task: _Task,
        full_preds: list[Prediction],
        label_routes: tuple[LabelRoute, ...],
    ) -> None:
        """Emit synthetic predictions for each matching label route.

        Synthetic predictions sharing ``(target_ontology, target_label,
        time_range)`` collapse to a single emitted prediction with score =
        max of contributing source scores. All contributing source
        predictions become triggers on the synthetic ontology child node.
        """
        # Map (ontology, target_label, time_range) -> list[Prediction] hits
        synth: dict[
            tuple[str, str, tuple[float, float]],
            list[Prediction],
        ] = {}
        # Map ontology -> set of triggering source predictions
        triggers_by_ontology: dict[str, dict[Trigger, None]] = {}

        for pred in full_preds:
            for route in label_routes:
                labels = route.match.resolve(self.routes.groups)
                if pred.label not in labels:
                    continue
                if pred.score < route.min_score:
                    continue
                key = (route.target_ontology, route.target_label, pred.time_range)
                synth.setdefault(key, []).append(pred)
                trig = Trigger(
                    from_model=task.model,
                    label=pred.label,
                    score=pred.score,
                    time_range=pred.time_range,
                )
                triggers_by_ontology.setdefault(route.target_ontology, {})[trig] = None

        if not synth:
            return

        # Group emitted predictions per ontology so each ontology gets
        # exactly one child node under this parent (per parent segment range).
        emissions_by_ontology: dict[str, list[Prediction]] = {}
        for (ontology, target_label, time_range), hits in synth.items():
            score = max(p.score for p in hits)
            emissions_by_ontology.setdefault(ontology, []).append(
                Prediction(label=target_label, score=score, time_range=time_range)
            )

        for ontology, preds in emissions_by_ontology.items():
            preds.sort(key=lambda p: (p.time_range, -p.score))
            triggers = list(triggers_by_ontology[ontology].keys())
            child = ResultNode(
                model=ontology,
                source_path=task.segment.source_path,
                segment_time_range=task.segment.time_range,
                predictions=preds,
                triggers=triggers,
            )
            task.node.add_child(child)

    def _fire_rule(
        self,
        *,
        task: _Task,
        prediction: Prediction,
        rule_payload: PayloadKind,
        rule_max_depth: int | None,
        target_model: str,
        cache: _AudioCache,
        queue: "deque[_Task]",
        children_index: dict[tuple[str, tuple[float, float]], ResultNode],
    ) -> None:
        target = self._models[target_model]
        downstream = self._build_downstream_segment(
            prediction=prediction,
            upstream_segment=task.segment,
            payload=rule_payload,
            target_sample_rate=target.target_sample_rate,
            cache=cache,
        )
        if downstream is None:
            return  # degenerate slice; skip

        trigger = Trigger(
            from_model=task.model,
            label=prediction.label,
            score=prediction.score,
            time_range=prediction.time_range,
        )
        dedup_key = (target_model, downstream.time_range)

        existing = children_index.get(dedup_key)
        if existing is not None:
            # Two rules from the same parent matched the same prediction and
            # routed to the same target. Keep one trigger per distinct
            # (label, score, time_range); skip exact duplicates.
            if trigger not in existing.triggers:
                existing.add_trigger(trigger)
            return

        # Depth guard: the lineage records ancestors; +1 accounts for the
        # step appended for the current parent (task.model) below.
        effective_max_depth = (
            rule_max_depth if rule_max_depth is not None else self.max_depth
        )
        new_depth = len(task.segment.lineage) + 1
        if new_depth > effective_max_depth:
            return

        # Cycle guard: reject the hop if the target model already appears
        # in the prospective lineage (ancestors + current parent).
        new_lineage_models = {step.model for step in task.segment.lineage}
        new_lineage_models.add(task.model)
        if target_model in new_lineage_models:
            return

        downstream = downstream.with_lineage(
            LineageStep(
                model=task.model,
                triggering_label=prediction.label,
                triggering_score=prediction.score,
            )
        )

        child = ResultNode(
            model=target_model,
            source_path=downstream.source_path,
            segment_time_range=downstream.time_range,
            triggers=[trigger],
        )
        task.node.add_child(child)
        children_index[dedup_key] = child
        queue.append(_Task(target_model, downstream, child))

    def _build_downstream_segment(
        self,
        *,
        prediction: Prediction,
        upstream_segment: Segment,
        payload: PayloadKind,
        target_sample_rate: int,
        cache: _AudioCache,
    ) -> Segment | None:
        source = upstream_segment.source_path
        wave = cache.get(source, target_sample_rate)
        full_duration = wave.numel() / target_sample_rate

        if payload == "triggering_window":
            start_s, end_s = prediction.time_range
            start_s = max(0.0, start_s)
            end_s = min(full_duration, end_s)
            if end_s <= start_s:
                return None
            s = int(round(start_s * target_sample_rate))
            e = int(round(end_s * target_sample_rate))
            if e <= s:
                return None
            return Segment(
                waveform=wave[s:e],
                sample_rate=target_sample_rate,
                source_path=source,
                time_range=(start_s, end_s),
                lineage=list(upstream_segment.lineage),
            )

        if payload == "whole_file":
            return Segment(
                waveform=wave,
                sample_rate=target_sample_rate,
                source_path=source,
                time_range=(0.0, full_duration),
                lineage=list(upstream_segment.lineage),
            )

        raise PipelineError(f"Unsupported payload kind {payload!r}.")

    def _filter_for_report(self, preds: list[Prediction]) -> list[Prediction]:
        """Apply ``report_top_k`` / ``report_score_floor`` per internal chunk."""
        if self.report_top_k is None and self.report_score_floor <= 0:
            return list(preds)

        by_chunk: dict[tuple[float, float], list[Prediction]] = {}
        for p in preds:
            by_chunk.setdefault(p.time_range, []).append(p)

        out: list[Prediction] = []
        for chunk_preds in by_chunk.values():
            chunk_preds.sort(key=lambda p: p.score, reverse=True)
            filtered = [p for p in chunk_preds if p.score >= self.report_score_floor]
            if self.report_top_k is not None:
                filtered = filtered[: self.report_top_k]
            out.extend(filtered)
        return out


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_REPORT_SCORE_FLOOR",
    "DEFAULT_REPORT_TOP_K",
    "Pipeline",
    "PipelineError",
]
