"""SSLAM wrapper for the AudioSet-2M finetuned checkpoint.

Wraps the preprocessing and inference path documented on the SSLAM model
card behind the :class:`classpatch.base.AudioClassifier` contract so the
executor can treat SSLAM as one node in a routing graph alongside other
models.
"""

from __future__ import annotations

import torch
import torchaudio

from classpatch.base import (
    AudioClassifier,
    Prediction,
    Segment,
    get_device,
    resample_segment,
)
from classpatch.models.audioset_vocab import AUDIOSET_LABEL_LIST, AUDIOSET_LABELS


DEFAULT_MODEL_ID = "ta012/SSLAM_AS2M_Finetuned"
SAMPLE_RATE = 16000
# Per the SSLAM model card: target_length=1024 frames at 10 ms hop == 10.24 s.
TARGET_FRAMES = 1024
WINDOW_SECONDS = 10
NORM_MEAN = -4.268
NORM_STD = 4.569


class SSLAMClassifier(AudioClassifier):
    """SSLAM AudioSet-2M classifier (527 classes).

    Internally chunks input audio into fixed ``window_seconds``-long windows
    (default 10 s) and runs the model on each, returning per-chunk
    :class:`Prediction`s with absolute timestamps.

    Construction is cheap; weights are pulled on the first call to
    :meth:`load` (or implicitly by :meth:`classify`).
    """

    name = "sslam"
    target_sample_rate = SAMPLE_RATE
    labels = AUDIOSET_LABEL_LIST

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        window_seconds: int = WINDOW_SECONDS,
        device: torch.device | str | None = None,
        max_chunks: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.window_seconds = window_seconds
        self.max_chunks = max_chunks
        self._device: torch.device | None = (
            torch.device(device) if isinstance(device, str) else device
        )
        self._model: torch.nn.Module | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = get_device()
        return self._device

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = model.eval().to(self.device)

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify(
        self,
        segment: Segment,
        score_floor: float = 0.0,
        top_k: int | None = None,
    ) -> list[Prediction]:
        self.load()
        seg = resample_segment(segment, self.target_sample_rate)
        chunks, offsets_sec = self._chunk(seg.waveform, seg.sample_rate)
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]
            offsets_sec = offsets_sec[: self.max_chunks]

        seg_start = seg.time_range[0]
        predictions: list[Prediction] = []
        for chunk, offset_sec in zip(chunks, offsets_sec):
            chunk_start = seg_start + offset_sec
            chunk_end = chunk_start + self.window_seconds
            mel = self._waveform_to_mel(chunk, seg.sample_rate).to(self.device)
            with torch.no_grad():
                logits = self._model(mel)  # type: ignore[misc]
            probs = torch.sigmoid(logits).squeeze(0).float().cpu()
            predictions.extend(
                self._predictions_from_probs(
                    probs,
                    time_range=(chunk_start, chunk_end),
                    score_floor=score_floor,
                    top_k=top_k,
                )
            )
        return predictions

    def _chunk(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Split ``waveform`` into non-overlapping ``window_seconds`` chunks.

        Drops any trailing partial window when there is more than one full
        chunk, but keeps the full waveform as a single chunk if it's shorter
        than one window — :meth:`_waveform_to_mel` zero-pads the mel
        spectrogram up to ``target_frames`` to satisfy the model's input
        shape requirement.
        """
        window_samples = self.window_seconds * sample_rate
        if waveform.numel() <= window_samples:
            return [waveform], [0.0]
        n_chunks = waveform.numel() // window_samples
        chunks = [
            waveform[i * window_samples : (i + 1) * window_samples]
            for i in range(n_chunks)
        ]
        offsets = [float(i * self.window_seconds) for i in range(n_chunks)]
        return chunks, offsets

    @staticmethod
    def _waveform_to_mel(
        waveform: torch.Tensor,
        sample_rate: int,
        target_frames: int = TARGET_FRAMES,
    ) -> torch.Tensor:
        """Replicate the SSLAM model card preprocessing exactly."""
        waveform = waveform - waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        ).unsqueeze(0)  # (1, T, 128)

        n_frames = mel.shape[1]
        if n_frames < target_frames:
            mel = torch.nn.ZeroPad2d((0, 0, 0, target_frames - n_frames))(mel)
        else:
            mel = mel[:, :target_frames, :]

        mel = (mel - NORM_MEAN) / (NORM_STD * 2)
        return mel.unsqueeze(0)  # (1, 1, T, 128)

    @staticmethod
    def _predictions_from_probs(
        probs: torch.Tensor,
        time_range: tuple[float, float],
        score_floor: float,
        top_k: int | None,
    ) -> list[Prediction]:
        if top_k is not None:
            k = min(top_k, probs.numel())
            values, indices = torch.topk(probs, k)
            items: list[tuple[int, float]] = [
                (int(idx), float(val)) for idx, val in zip(indices, values)
            ]
        else:
            items = [(idx, float(val)) for idx, val in enumerate(probs.tolist())]

        preds = [
            Prediction(label=AUDIOSET_LABELS[idx], score=score, time_range=time_range)
            for idx, score in items
            if score >= score_floor
        ]
        preds.sort(key=lambda p: p.score, reverse=True)
        return preds
