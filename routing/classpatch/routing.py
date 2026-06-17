"""Routing rules and label groups for the classpatch pipeline.

The :class:`RouteBook` is the user-facing container. It holds:

- named **label groups** (e.g. ``"any_bird" -> {"Bird", "Chirp, tweet", ...}``),
- **routing rules** that dispatch matching predictions to another model for
  further inference, and
- **label routes** that remap matching predictions into a user-defined
  *ontology* without running any new inference.

The executor consumes a :class:`RouteBook` to decide which downstream models
get which audio (routing rules) and which synthetic ontology predictions to
emit alongside model output (label routes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping


PayloadKind = Literal["triggering_window", "whole_file"]
_VALID_PAYLOADS: frozenset[str] = frozenset({"triggering_window", "whole_file"})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelGroup:
    """A named set of labels (e.g. ``"any_bird"`` -> {"Bird", "Chirp, tweet", ...}).

    Groups exist so rules can target multi-label patches of an ontology without
    re-listing every label. They are stored by name in a :class:`RouteBook`;
    rules reference them by name (not by value), so editing a group definition
    propagates to every rule that uses it.
    """

    name: str
    labels: frozenset[str]

    def to_dict(self) -> dict:
        return {"name": self.name, "labels": sorted(self.labels)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "LabelGroup":
        return cls(name=d["name"], labels=frozenset(d["labels"]))


@dataclass(frozen=True)
class LabelMatch:
    """Predicate used by a :class:`RoutingRule` to select firing predictions.

    Exactly one of ``labels`` (a literal set) or ``group`` (a name registered
    in the owning :class:`RouteBook`) must be set. The group form keeps
    serialised rules small and editable in one place.
    """

    labels: frozenset[str] | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        if (self.labels is None) == (self.group is None):
            raise ValueError(
                "LabelMatch must specify exactly one of `labels` or `group`."
            )

    def resolve(self, groups: Mapping[str, LabelGroup]) -> frozenset[str]:
        """Return the effective label set, resolving group references."""
        if self.group is not None:
            try:
                return groups[self.group].labels
            except KeyError as exc:
                raise KeyError(
                    f"LabelMatch references unknown group {self.group!r}."
                ) from exc
        assert self.labels is not None  # narrowing for type checkers
        return self.labels

    def to_dict(self) -> dict:
        if self.group is not None:
            return {"group": self.group}
        assert self.labels is not None
        return {"labels": sorted(self.labels)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "LabelMatch":
        if "group" in d:
            return cls(group=d["group"])
        return cls(labels=frozenset(d["labels"]))


@dataclass(frozen=True)
class RoutingRule:
    """A declarative edge in the model routing graph.

    "When ``from_model`` emits a prediction whose label is in ``match``
    (resolved against the owning route book's groups) and whose score is at
    least ``min_score``, dispatch the triggering segment to every model in
    ``to_models`` using ``payload`` to decide what audio to forward."

    ``max_depth`` overrides the pipeline-wide depth cap for this rule only;
    ``None`` means "use the pipeline default".
    """

    from_model: str
    match: LabelMatch
    min_score: float
    to_models: tuple[str, ...]
    payload: PayloadKind = "triggering_window"
    max_depth: int | None = None

    def __post_init__(self) -> None:
        if not self.to_models:
            raise ValueError("RoutingRule must have at least one target model.")
        if self.payload not in _VALID_PAYLOADS:
            raise ValueError(
                f"Unknown payload {self.payload!r}; expected one of "
                f"{sorted(_VALID_PAYLOADS)}."
            )
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                f"min_score must be in [0, 1]; got {self.min_score!r}."
            )

    def to_dict(self) -> dict:
        return {
            "from_model": self.from_model,
            "match": self.match.to_dict(),
            "min_score": self.min_score,
            "to_models": list(self.to_models),
            "payload": self.payload,
            "max_depth": self.max_depth,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "RoutingRule":
        return cls(
            from_model=d["from_model"],
            match=LabelMatch.from_dict(d["match"]),
            min_score=float(d["min_score"]),
            to_models=tuple(d["to_models"]),
            payload=d.get("payload", "triggering_window"),
            max_depth=d.get("max_depth"),
        )


@dataclass(frozen=True)
class LabelRoute:
    """A pure label-to-label remap (no inference).

    "When ``from_model`` emits a prediction matching ``match`` with score
    at least ``min_score``, emit a synthetic prediction labelled
    ``target_label`` under the ontology ``target_ontology`` for the same
    time range. The synthetic prediction's score is the source
    prediction's score."

    Multiple :class:`LabelRoute` records that produce the same
    ``(target_ontology, target_label, time_range)`` from different source
    predictions are collapsed by the executor: a single synthetic
    prediction is emitted with score equal to the **max** of the
    contributing source scores.
    """

    from_model: str
    match: LabelMatch
    min_score: float
    target_ontology: str
    target_label: str

    def __post_init__(self) -> None:
        if not self.target_ontology:
            raise ValueError("LabelRoute.target_ontology must be non-empty.")
        if not self.target_label:
            raise ValueError("LabelRoute.target_label must be non-empty.")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                f"min_score must be in [0, 1]; got {self.min_score!r}."
            )

    def to_dict(self) -> dict:
        return {
            "from_model": self.from_model,
            "match": self.match.to_dict(),
            "min_score": self.min_score,
            "target_ontology": self.target_ontology,
            "target_label": self.target_label,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "LabelRoute":
        return cls(
            from_model=d["from_model"],
            match=LabelMatch.from_dict(d["match"]),
            min_score=float(d["min_score"]),
            target_ontology=d["target_ontology"],
            target_label=d["target_label"],
        )


# ---------------------------------------------------------------------------
# Fluent builders
# ---------------------------------------------------------------------------


class RouteBuilder:
    """Partial state machine for declaring a single :class:`RoutingRule`.

    Returned by :meth:`RouteBook.route`. The rule is finalised and added to
    the book only when :meth:`to` is called.

    Example
    -------
    ::

        book.route("sslam").when("any_bird").above(0.2).to("birdnet")
    """

    def __init__(self, book: "RouteBook", from_model: str) -> None:
        self._book = book
        self._from_model = from_model
        self._match: LabelMatch | None = None
        self._min_score: float = 0.0

    def when(self, name_or_label: str) -> "RouteBuilder":
        """Match a registered group by name, or a single literal label.

        If ``name_or_label`` matches a group name registered on the parent
        :class:`RouteBook`, this is treated as a group reference. Otherwise
        it is treated as a single literal label. Use :meth:`when_group` or
        :meth:`when_label` when you need to disambiguate.
        """
        if name_or_label in self._book._groups:
            self._match = LabelMatch(group=name_or_label)
        else:
            self._match = LabelMatch(labels=frozenset({name_or_label}))
        return self

    def when_group(self, name: str) -> "RouteBuilder":
        """Match the named group (must already be registered)."""
        if name not in self._book._groups:
            raise KeyError(f"No group named {name!r} has been defined.")
        self._match = LabelMatch(group=name)
        return self

    def when_label(self, label: str) -> "RouteBuilder":
        """Match a single literal label."""
        self._match = LabelMatch(labels=frozenset({label}))
        return self

    def when_any(self, labels: Iterable[str]) -> "RouteBuilder":
        """Match any of a literal set of labels."""
        labels = frozenset(labels)
        if not labels:
            raise ValueError(".when_any() requires at least one label.")
        self._match = LabelMatch(labels=labels)
        return self

    def above(self, min_score: float) -> "RouteBuilder":
        """Require prediction score >= ``min_score``."""
        self._min_score = float(min_score)
        return self

    def to(
        self,
        *to_models: str,
        payload: PayloadKind = "triggering_window",
        max_depth: int | None = None,
    ) -> RoutingRule:
        """Finalise the rule, append it to the route book, and return it."""
        if self._match is None:
            raise ValueError(
                "Cannot finalise a route without .when() / .when_group() / "
                ".when_label() / .when_any()."
            )
        if not to_models:
            raise ValueError(".to() requires at least one target model name.")
        rule = RoutingRule(
            from_model=self._from_model,
            match=self._match,
            min_score=self._min_score,
            to_models=tuple(to_models),
            payload=payload,
            max_depth=max_depth,
        )
        self._book._rules.append(rule)
        return rule


class RelabelBuilder:
    """Partial state machine for declaring a single :class:`LabelRoute`.

    Returned by :meth:`OntologyBuilder.relabel`. Mirrors :class:`RouteBuilder`
    but terminates with :meth:`as_label` instead of ``.to(...)`` — the
    distinction makes it obvious at the call site whether a rule fans out
    to another *model* (inference) or to another *label* (remap).

    Example
    -------
    ::

        ecology = book.ontology("ecology")
        ecology.relabel("sslam").when("any_bird").above(0.20).as_label("bird_activity")
    """

    def __init__(
        self,
        book: "RouteBook",
        target_ontology: str,
        from_model: str,
    ) -> None:
        self._book = book
        self._target_ontology = target_ontology
        self._from_model = from_model
        self._match: LabelMatch | None = None
        self._min_score: float = 0.0

    def when(self, name_or_label: str) -> "RelabelBuilder":
        """Match a registered group by name, or a single literal label."""
        if name_or_label in self._book._groups:
            self._match = LabelMatch(group=name_or_label)
        else:
            self._match = LabelMatch(labels=frozenset({name_or_label}))
        return self

    def when_group(self, name: str) -> "RelabelBuilder":
        if name not in self._book._groups:
            raise KeyError(f"No group named {name!r} has been defined.")
        self._match = LabelMatch(group=name)
        return self

    def when_label(self, label: str) -> "RelabelBuilder":
        self._match = LabelMatch(labels=frozenset({label}))
        return self

    def when_any(self, labels: Iterable[str]) -> "RelabelBuilder":
        labels = frozenset(labels)
        if not labels:
            raise ValueError(".when_any() requires at least one label.")
        self._match = LabelMatch(labels=labels)
        return self

    def above(self, min_score: float) -> "RelabelBuilder":
        self._min_score = float(min_score)
        return self

    def as_label(self, target_label: str) -> LabelRoute:
        """Finalise the label route and append it to the route book."""
        if self._match is None:
            raise ValueError(
                "Cannot finalise a relabel without .when() / .when_group() / "
                ".when_label() / .when_any()."
            )
        route = LabelRoute(
            from_model=self._from_model,
            match=self._match,
            min_score=self._min_score,
            target_ontology=self._target_ontology,
            target_label=target_label,
        )
        self._book._label_rules.append(route)
        return route


class OntologyBuilder:
    """Namespace handle for declaring label routes that target one ontology.

    Returned by :meth:`RouteBook.ontology`. Lets multiple relabel rules
    sourced from different models all share the same ``target_ontology``
    visually:

    ::

        ecology = book.ontology("ecology")
        ecology.relabel("sslam").when("any_bird").above(0.2).as_label("bird_activity")
        ecology.relabel("sslam").when_label("Speech").above(0.4).as_label("human_voice")
        ecology.relabel("birdnet").when_label("Northern Cardinal").above(0.5).as_label("cardinal")
    """

    def __init__(self, book: "RouteBook", name: str) -> None:
        self._book = book
        self.name = name

    def relabel(self, from_model: str) -> RelabelBuilder:
        """Start declaring a label route from ``from_model`` into this ontology."""
        return RelabelBuilder(self._book, self.name, from_model)

    def __repr__(self) -> str:
        n = sum(1 for r in self._book._label_rules if r.target_ontology == self.name)
        return f"OntologyBuilder(name={self.name!r}, label_rules={n})"


# ---------------------------------------------------------------------------
# RouteBook
# ---------------------------------------------------------------------------


class RouteBook:
    """Container of label groups, ontologies, routing rules, and label routes.

    Designed to outlive any single pipeline instance: define routes once,
    save with :meth:`save`, reload across sessions with :meth:`load`, and
    pass the same book to whichever pipeline executor consumes it.
    """

    def __init__(self) -> None:
        self._groups: dict[str, LabelGroup] = {}
        self._ontologies: set[str] = set()
        self._rules: list[RoutingRule] = []
        self._label_rules: list[LabelRoute] = []

    # ---- groups --------------------------------------------------------

    def define_group(
        self,
        name: str,
        labels: Iterable[str],
        *,
        overwrite: bool = False,
    ) -> LabelGroup:
        """Register a named label group.

        Subsequent rules can reference it via :meth:`RouteBuilder.when` or
        :meth:`RouteBuilder.when_group`. Pass ``overwrite=True`` to replace
        an existing group of the same name (rules that reference it pick up
        the new membership automatically because they store the name).
        """
        if name in self._groups and not overwrite:
            raise KeyError(
                f"Group {name!r} already exists; pass overwrite=True to replace."
            )
        group = LabelGroup(name=name, labels=frozenset(labels))
        self._groups[name] = group
        return group

    def group(self, name: str) -> LabelGroup:
        return self._groups[name]

    @property
    def groups(self) -> Mapping[str, LabelGroup]:
        return self._groups

    # ---- ontologies ----------------------------------------------------

    def define_ontology(self, name: str) -> "OntologyBuilder":
        """Register a custom ontology name.

        Label routes target a named ontology; predictions remapped under
        that name appear as a synthetic child node in the result tree with
        ``model == name``. Calling this method twice with the same name is
        a no-op.
        """
        if not name:
            raise ValueError("Ontology name must be non-empty.")
        self._ontologies.add(name)
        return OntologyBuilder(self, name)

    def ontology(self, name: str) -> "OntologyBuilder":
        """Return a builder for the named ontology, defining it if necessary."""
        if name not in self._ontologies:
            self.define_ontology(name)
        return OntologyBuilder(self, name)

    @property
    def ontologies(self) -> frozenset[str]:
        return frozenset(self._ontologies)

    # ---- rules ---------------------------------------------------------

    def route(self, from_model: str) -> RouteBuilder:
        """Start declaring a model route whose source is ``from_model``."""
        return RouteBuilder(self, from_model)

    @property
    def rules(self) -> tuple[RoutingRule, ...]:
        return tuple(self._rules)

    def rules_for(self, from_model: str) -> tuple[RoutingRule, ...]:
        """Return all model routes whose source is ``from_model``."""
        return tuple(r for r in self._rules if r.from_model == from_model)

    def clear_rules(self) -> None:
        self._rules.clear()

    def remove_rule(self, rule: RoutingRule) -> None:
        self._rules.remove(rule)

    # ---- label rules ---------------------------------------------------

    @property
    def label_rules(self) -> tuple[LabelRoute, ...]:
        return tuple(self._label_rules)

    def label_rules_for(self, from_model: str) -> tuple[LabelRoute, ...]:
        """Return all label routes whose source is ``from_model``."""
        return tuple(r for r in self._label_rules if r.from_model == from_model)

    def clear_label_rules(self) -> None:
        self._label_rules.clear()

    def remove_label_rule(self, route: LabelRoute) -> None:
        self._label_rules.remove(route)

    # ---- persistence ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "groups": [g.to_dict() for g in self._groups.values()],
            "ontologies": sorted(self._ontologies),
            "rules": [r.to_dict() for r in self._rules],
            "label_rules": [r.to_dict() for r in self._label_rules],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "RouteBook":
        book = cls()
        for g in d.get("groups", []):
            group = LabelGroup.from_dict(g)
            book._groups[group.name] = group
        for name in d.get("ontologies", []):
            book._ontologies.add(name)
        for r in d.get("rules", []):
            book._rules.append(RoutingRule.from_dict(r))
        for r in d.get("label_rules", []):
            route = LabelRoute.from_dict(r)
            book._ontologies.add(route.target_ontology)
            book._label_rules.append(route)
        book._validate_references()
        return book

    def save(self, path: Path | str, *, indent: int = 2) -> Path:
        """Persist the route book to ``path`` (JSON). Returns the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=indent, sort_keys=False)
        path.write_text(payload + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RouteBook":
        """Load a route book previously written by :meth:`save`."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- internals -----------------------------------------------------

    def _validate_references(self) -> None:
        missing_groups = sorted(
            {
                r.match.group
                for r in [*self._rules, *self._label_rules]
                if r.match.group is not None and r.match.group not in self._groups
            }
        )
        if missing_groups:
            raise KeyError(
                f"Rules reference undefined group(s): {missing_groups}."
            )

    def __repr__(self) -> str:
        return (
            f"RouteBook(groups={len(self._groups)}, "
            f"ontologies={len(self._ontologies)}, "
            f"rules={len(self._rules)}, "
            f"label_rules={len(self._label_rules)})"
        )


__all__ = [
    "LabelGroup",
    "LabelMatch",
    "LabelRoute",
    "OntologyBuilder",
    "PayloadKind",
    "RelabelBuilder",
    "RouteBook",
    "RouteBuilder",
    "RoutingRule",
]
