"""Core abstractions for the classpatch recursive audio classification pipeline.

Each concrete classifier wraps a model behind a uniform interface so the
executor can treat every model identically. Models own their own
preprocessing and chunking; the executor shuttles :class:`Segment`s between
them and collects :class:`Prediction`s.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchaudio


AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".ogg"})


@dataclass(frozen=True)
class LineageStep:
    """One hop in a segment's processing history.

    Used by the executor for cycle detection and for the audit trail in the
    final result tree. ``triggering_label`` / ``triggering_score`` are ``None``
    for the root segment (no model has emitted it yet).
    """

    model: str
    triggering_label: str | None
    triggering_score: float | None


@dataclass
class Segment:
    """A chunk of mono audio flowing through the pipeline.

    ``time_range`` is always expressed in *absolute seconds* in the original
    source file, so downstream models and result nodes can report honest
    timestamps regardless of how many hops the segment has taken.
    """

    waveform: torch.Tensor  # 1-D float tensor (samples,)
    sample_rate: int
    source_path: Path
    time_range: tuple[float, float]
    lineage: list[LineageStep] = field(default_factory=list)

    @property
    def duration(self) -> float:
        start, end = self.time_range
        return end - start

    @property
    def num_samples(self) -> int:
        return int(self.waveform.numel())

    def with_lineage(self, step: LineageStep) -> "Segment":
        """Return a copy with one additional lineage step appended."""
        return Segment(
            waveform=self.waveform,
            sample_rate=self.sample_rate,
            source_path=self.source_path,
            time_range=self.time_range,
            lineage=[*self.lineage, step],
        )


@dataclass(frozen=True)
class Prediction:
    """A single label emitted by a model on a sub-window of an input segment.

    ``time_range`` is absolute (relative to ``Segment.source_path``), produced
    by composing the segment's own ``time_range`` with the model's internal
    chunk offsets.
    """

    label: str
    score: float
    time_range: tuple[float, float]


class AudioClassifier(ABC):
    """Uniform contract every model wrapper must satisfy.

    Subclasses are expected to define the following class-level metadata:

    - ``name``: stable identifier used in routing tables (e.g. ``"sslam"``).
    - ``target_sample_rate``: the sample rate the underlying model wants.
    - ``labels``: the model's ordered label vocabulary.

    Heavy weights must be loaded lazily in :meth:`load`. The pipeline can run
    several models in series, so eager loading at construction time can put
    unnecessary pressure on memory in long-running sessions.
    """

    name: str
    target_sample_rate: int
    labels: list[str]

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto the chosen device. Must be idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights. Must be idempotent."""

    @abstractmethod
    def classify(
        self,
        segment: Segment,
        score_floor: float = 0.0,
        top_k: int | None = None,
    ) -> list[Prediction]:
        """Classify ``segment`` and return per-sub-window predictions.

        Implementations:

        - Resample ``segment`` to :attr:`target_sample_rate` if necessary.
        - Split it into the model's native window size internally.
        - Emit one or more :class:`Prediction`s per sub-window, with absolute
          time ranges anchored in ``segment.source_path``.
        - Apply ``score_floor`` and ``top_k`` *per sub-window* before
          returning. Results are sorted by score descending.

        Both filters default to permissive values so the executor can apply
        its own routing thresholds globally.
        """

    def __enter__(self) -> "AudioClassifier":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.unload()
        return False


def get_device() -> torch.device:
    """Pick CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mono_waveform(path: Path | str, target_sample_rate: int) -> torch.Tensor:
    """Load ``path``, resample to ``target_sample_rate``, downmix to mono.

    Returns a 1-D float tensor of shape ``(samples,)``.
    """
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def segment_from_file(path: Path | str, target_sample_rate: int) -> Segment:
    """Build a root :class:`Segment` covering the full duration of ``path``."""
    path = Path(path)
    waveform = load_mono_waveform(path, target_sample_rate)
    duration = waveform.numel() / target_sample_rate
    return Segment(
        waveform=waveform,
        sample_rate=target_sample_rate,
        source_path=path,
        time_range=(0.0, duration),
    )


def resample_segment(segment: Segment, target_sample_rate: int) -> Segment:
    """Return ``segment`` resampled to ``target_sample_rate`` (no-op if equal).

    Lineage and ``time_range`` are preserved; only the waveform and SR change.
    """
    if segment.sample_rate == target_sample_rate:
        return segment
    new_wave = torchaudio.functional.resample(
        segment.waveform.unsqueeze(0),
        segment.sample_rate,
        target_sample_rate,
    ).squeeze(0)
    return Segment(
        waveform=new_wave,
        sample_rate=target_sample_rate,
        source_path=segment.source_path,
        time_range=segment.time_range,
        lineage=list(segment.lineage),
    )


def list_audio_files(audio_dir: Path | str) -> list[Path]:
    """Return audio files in ``audio_dir`` matching :data:`AUDIO_EXTENSIONS`."""
    audio_dir = Path(audio_dir)
    return sorted(
        p
        for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
