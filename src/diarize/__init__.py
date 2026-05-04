"""Speaker diarization for Python — who spoke when.

CPU-only, no GPU required, no API keys, Apache 2.0 licensed.

Example::

    from diarize import diarize

    result = diarize("meeting.wav")
    print(result.num_speakers)  # 3
    print(result.segments)      # [Segment(...), ...]
    result.to_rttm("output.rttm")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .clustering import cluster_speakers, estimate_speakers  # noqa: F401
from .embeddings import extract_embeddings
from .utils import (
    DiarizeArtifacts,
    DiarizeResult,
    Segment,
    SpeakerEstimationDetails,  # noqa: F401
    SpeechSegment,
    SubSegment,
    get_audio_duration,
)
from .vad import run_vad


class _RawSegment(NamedTuple):
    """Intermediate segment used during diarization assembly."""

    start: float
    end: float
    speaker: str


__version__ = "0.1.0"
__all__ = [
    "diarize",
    "DiarizeResult",
    "DiarizeArtifacts",
    "Segment",
    "SpeakerEstimationDetails",
    "estimate_speakers",
    "__version__",
]

logger = logging.getLogger(__name__)


def _build_diarization_segments(
    speech_segments: list[SpeechSegment],
    subsegments: list[SubSegment],
    labels: np.ndarray,
) -> list[Segment]:
    """Assemble diarization segments from subsegments and cluster labels.

    Each subsegment (embedding window) gets its speaker label.  Adjacent
    subsegments from the same speaker are merged.  Short VAD segments
    without embeddings are assigned the nearest speaker.

    Args:
        speech_segments: Original speech segments from VAD.
        subsegments: Embedding windows with parent indices.
        labels: Cluster labels aligned with *subsegments*.

    Returns:
        Merged :class:`Segment` list sorted by start time.
    """
    # Build raw segments from subsegments + labels
    raw_segments: list[_RawSegment] = []
    for sub, label in zip(subsegments, labels):
        raw_segments.append(
            _RawSegment(
                start=sub.start,
                end=sub.end,
                speaker=f"SPEAKER_{int(label):02d}",
            )
        )

    # Add short VAD segments that were skipped during embedding extraction
    covered_indices = {sub.parent_idx for sub in subsegments}
    for idx, seg in enumerate(speech_segments):
        if idx in covered_indices:
            continue
        # Find nearest subsegment by time
        seg_mid = (seg.start + seg.end) / 2
        best_speaker = "SPEAKER_00"
        best_dist = float("inf")
        for sub, label in zip(subsegments, labels):
            sub_mid = (sub.start + sub.end) / 2
            dist = abs(seg_mid - sub_mid)
            if dist < best_dist:
                best_dist = dist
                best_speaker = f"SPEAKER_{int(label):02d}"
        raw_segments.append(_RawSegment(start=seg.start, end=seg.end, speaker=best_speaker))

    # Sort by time
    raw_segments.sort(key=lambda s: s.start)

    # Merge adjacent subsegments of the same speaker
    if not raw_segments:
        return []

    # Use mutable list of lists [start, end, speaker] for merging
    merged: list[list[float | str]] = [
        [raw_segments[0].start, raw_segments[0].end, raw_segments[0].speaker]
    ]
    for seg in raw_segments[1:]:
        prev = merged[-1]
        gap = seg.start - float(prev[1])
        if seg.speaker == prev[2] and gap < 0.7:
            prev[1] = max(float(prev[1]), seg.end)
        else:
            merged.append([seg.start, seg.end, seg.speaker])

    return [Segment(start=float(m[0]), end=float(m[1]), speaker=str(m[2])) for m in merged]


def diarize(
    audio_path: str | Path,
    *,
    min_speakers: int = 1,
    max_speakers: int = 20,
    num_speakers: int | None = None,
    return_artifacts: bool = False,
) -> DiarizeResult:
    """Run the full speaker diarization pipeline on an audio file.

    Pipeline stages:

    1. **Silero VAD** — detect speech segments
    2. **WeSpeaker ResNet34-LM** — extract 256-dim speaker embeddings
    3. **GMM BIC** — estimate number of speakers (unless *num_speakers*
       is provided)
    4. **Spectral Clustering** — assign speaker labels

    Args:
        audio_path: Path to an audio file (wav, mp3, flac, etc.).
        min_speakers: Minimum number of speakers for auto-detection.
        max_speakers: Maximum number of speakers for auto-detection.
        num_speakers: If set, skip auto-detection and use this exact
            number of speakers.
        return_artifacts: If ``True``, populate
            :attr:`DiarizeResult.artifacts` with speaker embeddings and
            subsegments. Defaults to ``False``.

    Returns:
        :class:`DiarizeResult` containing segments, speaker info, and
        export methods.

    Example::

        from diarize import diarize

        result = diarize("meeting.wav")
        print(f"Found {result.num_speakers} speakers")
        for seg in result.segments:
            print(f"  [{seg.start:.1f} - {seg.end:.1f}] {seg.speaker}")
        result.to_rttm("meeting.rttm")
    """
    # ── Input validation ─────────────────────────────────────────────────
    if min_speakers < 1:
        raise ValueError(f"min_speakers must be >= 1, got {min_speakers}")
    if max_speakers < min_speakers:
        raise ValueError(f"max_speakers ({max_speakers}) must be >= min_speakers ({min_speakers})")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError(f"num_speakers must be >= 1, got {num_speakers}")

    audio_path_str = str(audio_path)
    duration = get_audio_duration(audio_path_str)

    logger.info("Diarizing: %s (%.1f seconds)", Path(audio_path_str).name, duration)

    # 1. Voice Activity Detection
    speech_segments: list[SpeechSegment] = run_vad(audio_path_str)
    if not speech_segments:
        logger.warning("No speech detected in %s", audio_path_str)
        return DiarizeResult(audio_path=audio_path_str, audio_duration=duration)

    # 2. Speaker embeddings
    embeddings, subsegments = extract_embeddings(audio_path_str, speech_segments)
    if len(embeddings) == 0:
        logger.warning("Could not extract embeddings from %s", audio_path_str)
        return DiarizeResult(audio_path=audio_path_str, audio_duration=duration)

    # 3. Clustering
    labels, estimation_details = cluster_speakers(
        embeddings,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        num_speakers=num_speakers,
    )

    # 4. Build result
    segments = _build_diarization_segments(speech_segments, subsegments, labels)

    result = DiarizeResult(
        segments=segments,
        audio_path=audio_path_str,
        audio_duration=duration,
        estimation_details=estimation_details,
        artifacts=DiarizeArtifacts(
            embeddings=embeddings.tolist(),
            subsegments=subsegments,
        ) if return_artifacts else None,
    )

    logger.info(
        "Diarization complete: %d speakers, %d segments",
        result.num_speakers,
        len(result.segments),
    )

    return result
