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
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .clustering import cluster_speakers, estimate_speakers  # noqa: F401
from .embeddings import extract_embeddings
from .utils import (
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
    "Segment",
    "SpeakerEstimationDetails",
    "estimate_speakers",
    "__version__",
]

logger = logging.getLogger(__name__)


def _majority_label(labels: list[int]) -> int | None:
    """Return the unique majority label, or ``None`` on ties."""
    if not labels:
        return None

    counts = Counter(labels)
    best_label, best_count = counts.most_common(1)[0]
    if sum(1 for count in counts.values() if count == best_count) > 1:
        return None
    return int(best_label)


def _smooth_window_labels(labels: list[int]) -> list[int]:
    """Apply a 3-window majority filter while preserving ties."""
    if len(labels) < 3:
        return labels

    smoothed: list[int] = []
    for idx, label in enumerate(labels):
        window = labels[max(0, idx - 1) : min(len(labels), idx + 2)]
        majority = _majority_label(window)
        smoothed.append(label if majority is None else majority)

    return smoothed


def _window_boundaries(
    speech_segment: SpeechSegment,
    segment_subsegments: list[SubSegment],
) -> list[tuple[float, float]]:
    """Convert overlapping windows into non-overlapping intervals."""
    if not segment_subsegments:
        return []

    centers = [(sub.start + sub.end) / 2 for sub in segment_subsegments]
    boundaries = [speech_segment.start]
    for left, right in zip(centers, centers[1:]):
        midpoint = (left + right) / 2
        boundaries.append(min(speech_segment.end, max(speech_segment.start, midpoint)))
    boundaries.append(speech_segment.end)

    return list(zip(boundaries[:-1], boundaries[1:]))


def _build_diarization_segments(
    speech_segments: list[SpeechSegment],
    subsegments: list[SubSegment],
    labels: np.ndarray,
) -> list[Segment]:
    """Assemble diarization segments from subsegments and cluster labels.

    Overlapping embedding windows are converted to a non-overlapping
    timeline and smoothed with a local majority filter. VAD segments
    without embeddings are assigned the nearest speaker.

    Args:
        speech_segments: Original speech segments from VAD.
        subsegments: Embedding windows with parent indices.
        labels: Cluster labels aligned with *subsegments*.

    Returns:
        Merged :class:`Segment` list sorted by start time.
    """
    raw_segments: list[_RawSegment] = []
    subsegments_by_parent: dict[int, list[int]] = {}
    for idx, sub in enumerate(subsegments):
        subsegments_by_parent.setdefault(sub.parent_idx, []).append(idx)

    for parent_idx, speech_segment in enumerate(speech_segments):
        indices = subsegments_by_parent.get(parent_idx)
        if not indices:
            continue

        indices.sort(key=lambda idx: subsegments[idx].start)
        parent_labels = _smooth_window_labels([int(labels[idx]) for idx in indices])
        parent_subsegments = [subsegments[idx] for idx in indices]
        windows = _window_boundaries(speech_segment, parent_subsegments)
        for (start, end), label in zip(windows, parent_labels):
            if end <= start:
                continue
            raw_segments.append(
                _RawSegment(
                    start=start,
                    end=end,
                    speaker=f"SPEAKER_{label:02d}",
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
        for raw in raw_segments:
            raw_mid = (raw.start + raw.end) / 2
            dist = abs(seg_mid - raw_mid)
            if dist < best_dist:
                best_dist = dist
                best_speaker = raw.speaker
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
    segments = _build_diarization_segments(
        speech_segments,
        subsegments,
        labels,
    )

    result = DiarizeResult(
        segments=segments,
        audio_path=audio_path_str,
        audio_duration=duration,
        estimation_details=estimation_details,
    )

    logger.info(
        "Diarization complete: %d speakers, %d segments",
        result.num_speakers,
        len(result.segments),
    )

    return result
