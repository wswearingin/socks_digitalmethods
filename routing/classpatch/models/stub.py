"""Stub classifier for end-to-end pipeline tests.

:class:`StubClassifier` implements the :class:`AudioClassifier` contract but
performs no real inference: it returns a fixed list of ``(label, score)``
pairs for every input segment. Useful for exercising routing and executor
logic without depending on a real model.
"""

from __future__ import annotations

from typing import Iterable

from classpatch.base import AudioClassifier, Prediction, Segment


class StubClassifier(AudioClassifier):
    """A no-op classifier that returns canned predictions.

    Parameters
    ----------
    name :
        Identifier used in routing tables (e.g. ``"stub_birdnet"``).
    fixed_predictions :
        ``(label, score)`` pairs returned on every call to :meth:`classify`.
        Each gets the input segment's absolute ``time_range``.
    target_sample_rate :
        Native sample rate (defaults to 16 kHz). The pipeline resamples
        slices to this rate before handing them over.
    """

    def __init__(
        self,
        name: str,
        fixed_predictions: Iterable[tuple[str, float]],
        target_sample_rate: int = 16000,
    ) -> None:
        self.name = name
        self.target_sample_rate = target_sample_rate
        self._fixed = list(fixed_predictions)
        self.labels = [label for label, _ in self._fixed]

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def classify(
        self,
        segment: Segment,
        score_floor: float = 0.0,
        top_k: int | None = None,
    ) -> list[Prediction]:
        preds = [
            Prediction(label=label, score=score, time_range=segment.time_range)
            for label, score in self._fixed
            if score >= score_floor
        ]
        preds.sort(key=lambda p: p.score, reverse=True)
        if top_k is not None:
            preds = preds[:top_k]
        return preds


__all__ = ["StubClassifier"]
