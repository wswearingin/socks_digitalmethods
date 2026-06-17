"""classpatch: modular, recursive multi-model audio classification pipeline.

Top-level exports cover the four user-facing layers of the package:

- core types (:class:`Segment`, :class:`Prediction`, :class:`AudioClassifier`),
- routing configuration (:class:`RouteBook` and friends),
- the executor (:class:`Pipeline`), and
- result + aggregation (:class:`ResultNode`, :class:`Survey`).

Concrete model wrappers live under :mod:`classpatch.models`.
"""

from classpatch.base import (
    AUDIO_EXTENSIONS,
    AudioClassifier,
    LineageStep,
    Prediction,
    Segment,
    get_device,
    list_audio_files,
    load_mono_waveform,
    resample_segment,
    segment_from_file,
)
from classpatch.models import SSLAMClassifier, StubClassifier
from classpatch.pipeline import Pipeline, PipelineError
from classpatch.results import (
    ResultNode,
    ResultRow,
    Trigger,
    render,
)
from classpatch.routing import (
    LabelGroup,
    LabelMatch,
    LabelRoute,
    OntologyBuilder,
    PayloadKind,
    RelabelBuilder,
    RouteBook,
    RouteBuilder,
    RoutingRule,
)
from classpatch.survey import (
    Detection,
    Event,
    LabelTally,
    Survey,
    collect_detections,
    merge_detections,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "AudioClassifier",
    "Detection",
    "Event",
    "LabelGroup",
    "LabelMatch",
    "LabelRoute",
    "LabelTally",
    "LineageStep",
    "OntologyBuilder",
    "PayloadKind",
    "Pipeline",
    "PipelineError",
    "Prediction",
    "RelabelBuilder",
    "ResultNode",
    "ResultRow",
    "RouteBook",
    "RouteBuilder",
    "RoutingRule",
    "SSLAMClassifier",
    "Segment",
    "StubClassifier",
    "Survey",
    "Trigger",
    "collect_detections",
    "get_device",
    "list_audio_files",
    "load_mono_waveform",
    "merge_detections",
    "render",
    "resample_segment",
    "segment_from_file",
]
