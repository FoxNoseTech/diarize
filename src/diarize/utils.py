"""Core data structures and utility helpers.

All public data models use Pydantic v2 for validation, serialization,
and schema generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import soundfile as sf

from pydantic import BaseModel, Field, computed_field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "Segment",
    "SpeechSegment",
    "SubSegment",
    "SpeakerEstimationDetails",
    "DiarizeResult",
    "DiarizeArtifacts",
    "get_audio_duration",
    "format_timestamp",
]


# ── Core models ───────────────────────────────────────────────────────────────


class Segment(BaseModel):
    """A single diarization segment with start/end times and speaker label.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.  Must be greater than *start*.
        speaker: Speaker label, e.g. ``"SPEAKER_00"``.

    Example::

        seg = Segment(start=0.5, end=3.2, speaker="SPEAKER_00")
        print(seg.duration)  # 2.7
    """

    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    speaker: str = Field(..., min_length=1, description="Speaker label")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

    @model_validator(mode="after")
    def _validate_times(self) -> Segment:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self

    model_config = {"frozen": True}


class SpeechSegment(BaseModel):
    """A speech segment detected by VAD (no speaker label yet).

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
    """

    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start

    @model_validator(mode="after")
    def _validate_times(self) -> SpeechSegment:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self


class SubSegment(BaseModel):
    """An embedding window within a speech segment.

    Attributes:
        start: Window start time in seconds.
        end: Window end time in seconds.
        parent_idx: Index of the parent :class:`SpeechSegment`.
    """

    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    parent_idx: int = Field(..., ge=0, description="Index of the parent SpeechSegment")


class SpeakerEstimationDetails(BaseModel):
    """Diagnostic details from speaker count estimation.

    Attributes:
        method: Estimation method used (e.g. ``"gmm_bic"``).
        best_k: Estimated number of speakers.
        pca_dim: Number of PCA dimensions used.
        k_bics: Mapping of ``k -> BIC`` values evaluated.
        reason: Short description if estimation was skipped.
        cosine_sim_p10: 10th percentile of pairwise cosine similarities
            (populated when single-speaker pre-check is evaluated).
    """

    method: str = "gmm_bic"
    best_k: int = Field(..., ge=1)
    pca_dim: int | None = None
    k_bics: dict[int, float] = Field(default_factory=dict)
    reason: str | None = None
    cosine_sim_p10: float | None = None


class DiarizeArtifacts(BaseModel):
    """Internal pipeline artifacts, available when ``return_artifacts=True``.

    All fields use plain Python types for clean serialization.

    Attributes:
        embeddings: Speaker embedding vectors (one per subsegment).
        subsegments: Embedding windows used during extraction.
    """

    embeddings: list[list[float]] = Field(
        default_factory=list,
        description="Speaker embedding vectors as nested lists",
    )
    subsegments: list[SubSegment] = Field(
        default_factory=list,
        description="Embedding windows used during extraction",
    )


class DiarizeResult(BaseModel):
    """Result of speaker diarization.

    This is the main object returned by :func:`diarize.diarize`.

    Attributes:
        segments: Diarization segments sorted by start time.
        audio_path: Path to the source audio file.
        audio_duration: Duration of the source audio in seconds.
        estimation_details: Diagnostic info from speaker count estimation.

    Example::

        result = diarize("meeting.wav")
        print(result.num_speakers)      # 3
        print(result.speakers)          # ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
        result.to_rttm("meeting.rttm")  # export RTTM
        result.model_dump()             # full dict serialization
    """

    segments: list[Segment] = Field(default_factory=list)
    audio_path: str = ""
    audio_duration: float = Field(default=0.0, ge=0)
    estimation_details: SpeakerEstimationDetails | None = None
    artifacts: DiarizeArtifacts | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def speakers(self) -> list[str]:
        """Sorted list of unique speaker labels."""
        return sorted({seg.speaker for seg in self.segments})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_speakers(self) -> int:
        """Number of unique speakers detected."""
        return len(self.speakers)

    def to_rttm(self, path: str | Path | None = None) -> str:
        """Export segments as RTTM (Rich Transcription Time Marked) format.

        RTTM is the standard interchange format for diarization results,
        used by evaluation tools like ``pyannote.metrics`` and ``dscore``.

        Args:
            path: If provided, write RTTM to this file path.
                  Otherwise just return the RTTM string.

        Returns:
            RTTM-formatted string.

        Example::

            result.to_rttm("output.rttm")  # write to file
            rttm_str = result.to_rttm()    # get as string
        """
        file_id = Path(self.audio_path).stem if self.audio_path else "audio"
        lines: list[str] = []
        for seg in self.segments:
            dur = seg.end - seg.start
            lines.append(
                f"SPEAKER {file_id} 1 {seg.start:.6f} {dur:.6f} <NA> <NA> {seg.speaker} <NA> <NA>"
            )
        rttm = "\n".join(lines)
        if rttm:
            rttm += "\n"

        if path is not None:
            Path(path).write_text(rttm, encoding="utf-8")
            logger.info("RTTM written to %s", path)

        return rttm

    def to_list(self) -> list[dict[str, float | str]]:
        """Export segments as a list of plain dicts (JSON-friendly).

        Returns:
            List of ``{"start": float, "end": float, "speaker": str}`` dicts.
        """
        return [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in self.segments
        ]

    def __iter__(self) -> Iterator[Segment]:  # type: ignore[override]
        """Iterate over segments directly."""
        return iter(self.segments)

    def __len__(self) -> int:
        """Number of segments."""
        return len(self.segments)

    def __repr__(self) -> str:
        dur = format_timestamp(self.audio_duration) if self.audio_duration else "?"
        return (
            f"DiarizeResult(speakers={self.num_speakers}, "
            f"segments={len(self.segments)}, duration={dur})"
        )


# ── Utility functions ─────────────────────────────────────────────────────────


def get_audio_duration(audio_path: str | Path) -> float:
    """Return audio duration in seconds.

    Tries ``soundfile`` first, falls back to ``torchaudio``.

    Args:
        audio_path: Path to an audio file.

    Returns:
        Duration in seconds, or ``0.0`` if the file cannot be read.
    """
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception as exc:
        logger.debug("soundfile failed for %s: %s", audio_path, exc)
        try:
            import torchaudio

            info = torchaudio.info(str(audio_path))
            return info.num_frames / info.sample_rate
        except Exception as exc2:
            logger.warning("Could not determine duration for %s: %s", audio_path, exc2)
            return 0.0


def format_timestamp(seconds: float) -> str:
    """Format a number of seconds as ``HH:MM:SS`` or ``MM:SS``.

    Args:
        seconds: Time in seconds (non-negative).

    Returns:
        Human-readable timestamp string.

    Examples::

        format_timestamp(45)    # "00:45"
        format_timestamp(3661)  # "01:01:01"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
