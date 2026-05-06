#!/usr/bin/env python3
"""Benchmark diarize against RTTM speaker annotations.

This runner is intentionally dataset-agnostic: point it at an audio directory
and an RTTM directory and it will match files by stem, run diarize, and report
DER plus speaker-count accuracy.

Examples:
    python scripts/benchmark_rttm.py \
        --dataset voxconverse-dev \
        --audio-dir /Users/lukashov/records/benchmark/audio/audio \
        --rttm-dir /Users/lukashov/records/benchmark/rttm_annotations/dev \
        --output /Users/lukashov/records/benchmark/results_voxconverse_dev.json

    python scripts/benchmark_rttm.py \
        --dataset voxconverse-test \
        --audio-dir /Users/lukashov/records/benchmark/audio/test \
        --rttm-dir /Users/lukashov/records/benchmark/rttm_annotations/test \
        --output /Users/lukashov/records/benchmark/results_voxconverse_test.json

    python scripts/benchmark_rttm.py \
        --dataset ami-test \
        --audio-dir /Users/lukashov/records/benchmark/ami/audio/mix-headset/test \
        --rttm-file /path/to/AMI.SpeakerDiarization.Benchmark.test.rttm \
        --output /Users/lukashov/records/benchmark/ami/results_test.json

    # Isolate assignment/clustering quality when the true speaker count is known.
    python scripts/benchmark_rttm.py ... --oracle-speakers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

_TEMP_DIR = Path(tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", str(_TEMP_DIR / "diarize-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_TEMP_DIR / "diarize-matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="'uem' was approximated.*")

_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
_DIARIZE: tuple[Any, Any] | None = None
_PYANNOTE: tuple[Any, Any, Any] | None = None


@dataclass
class BenchmarkFile:
    file_id: str
    audio_path: Path
    rttm_source: str
    rttm_lines: list[str]
    gt_speakers: int


@dataclass
class FileResult:
    file_id: str
    gt_speakers: int
    predicted_speakers: int
    speaker_delta: int
    der: float | None
    elapsed_sec: float
    duration: float
    n_segments: int
    error: str | None = None


def load_pyannote() -> tuple[Any, Any, Any]:
    """Load optional pyannote benchmark dependencies lazily."""
    global _PYANNOTE
    if _PYANNOTE is not None:
        return _PYANNOTE

    try:
        from pyannote.core import Annotation
        from pyannote.core import Segment as PySegment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as exc:  # pragma: no cover - depends on optional local deps.
        raise SystemExit(
            "benchmark_rttm.py requires pyannote.metrics. Install it in your benchmark "
            "environment, for example: pip install pyannote.metrics"
        ) from exc

    _PYANNOTE = (Annotation, PySegment, DiarizationErrorRate)
    return _PYANNOTE


def load_diarize() -> tuple[Any, Any]:
    """Load diarize lazily after benchmark-specific env setup."""
    global _DIARIZE
    if _DIARIZE is not None:
        return _DIARIZE

    from diarize import diarize
    from diarize.utils import get_audio_duration

    _DIARIZE = (diarize, get_audio_duration)
    return _DIARIZE


def parse_rttm_lines_to_annotation(file_id: str, rttm_lines: list[str]) -> Any:
    """Parse a standard RTTM file into a pyannote Annotation."""
    Annotation, PySegment, _ = load_pyannote()
    annotation = Annotation(uri=file_id)
    for line in rttm_lines:
        parts = line.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        annotation[PySegment(start, start + duration)] = speaker
    return annotation


def rttm_lines_speaker_count(rttm_lines: list[str]) -> int:
    """Count unique speakers in an RTTM file."""
    speakers: set[str] = set()
    for line in rttm_lines:
        parts = line.strip().split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            speakers.add(parts[7])
    return len(speakers)


def result_to_annotation(file_id: str, segments) -> Any:
    """Convert diarize result segments to pyannote Annotation."""
    Annotation, PySegment, _ = load_pyannote()
    annotation = Annotation(uri=file_id)
    for segment in segments:
        annotation[PySegment(segment.start, segment.end)] = segment.speaker
    return annotation


def build_audio_index(audio_dir: Path) -> dict[str, Path]:
    """Index audio files recursively by stem and simple AMI-style aliases."""
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue

        aliases = [path.stem]
        if "." in path.stem:
            aliases.append(path.stem.split(".", 1)[0])

        for alias in aliases:
            if alias in index and index[alias] != path:
                duplicates.add(alias)
                continue
            index[alias] = path

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates)[:10])
        print(
            f"Warning: ignored duplicate audio stems ({duplicate_list})",
            file=sys.stderr,
        )
    return index


def load_rttm_groups_from_dir(rttm_dir: Path) -> dict[str, tuple[str, list[str]]]:
    """Load one-RTTM-per-recording annotations from a directory."""
    groups: dict[str, tuple[str, list[str]]] = {}
    for rttm_path in sorted(rttm_dir.rglob("*.rttm")):
        lines = rttm_path.read_text(encoding="utf-8").splitlines()
        groups[rttm_path.stem] = (str(rttm_path), lines)
    return groups


def load_rttm_groups_from_file(rttm_file: Path) -> dict[str, tuple[str, list[str]]]:
    """Load a combined RTTM file and group lines by recording id."""
    groups: dict[str, tuple[str, list[str]]] = {}
    grouped_lines: dict[str, list[str]] = {}
    with rttm_file.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            grouped_lines.setdefault(parts[1], []).append(line.rstrip("\n"))

    for file_id, lines in grouped_lines.items():
        groups[file_id] = (str(rttm_file), lines)
    return groups


def collect_files(
    audio_dir: Path,
    rttm_dir: Path | None,
    rttm_file: Path | None,
    *,
    gt_min: int,
    gt_max: int,
) -> tuple[list[BenchmarkFile], list[str]]:
    """Match RTTM files to audio files by stem."""
    audio_index = build_audio_index(audio_dir)
    if rttm_file is not None:
        rttm_groups = load_rttm_groups_from_file(rttm_file)
    elif rttm_dir is not None:
        rttm_groups = load_rttm_groups_from_dir(rttm_dir)
    else:  # pragma: no cover - parse_args/main validate this.
        raise ValueError("Either rttm_dir or rttm_file is required")

    files: list[BenchmarkFile] = []
    missing_audio: list[str] = []

    for file_id, (source, lines) in sorted(rttm_groups.items()):
        gt_speakers = rttm_lines_speaker_count(lines)
        if gt_speakers < gt_min or gt_speakers > gt_max:
            continue

        audio_path = audio_index.get(file_id)
        if audio_path is None:
            missing_audio.append(file_id)
            continue

        files.append(
            BenchmarkFile(
                file_id=file_id,
                audio_path=audio_path,
                rttm_source=source,
                rttm_lines=lines,
                gt_speakers=gt_speakers,
            )
        )

    return files, missing_audio


def load_existing_results(output_path: Path | None) -> list[FileResult]:
    """Load previous results from a JSON output file."""
    if output_path is None or not output_path.exists():
        return []

    with output_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    rows = payload["files"] if isinstance(payload, dict) and "files" in payload else payload
    return [FileResult(**row) for row in rows]


def write_results(
    output_path: Path | None,
    *,
    dataset: str,
    args: argparse.Namespace,
    results: list[FileResult],
) -> None:
    """Persist intermediate benchmark results."""
    if output_path is None:
        return

    payload = {
        "dataset": dataset,
        "audio_dir": str(args.audio_dir),
        "rttm_dir": str(args.rttm_dir) if args.rttm_dir else None,
        "rttm_file": str(args.rttm_file) if args.rttm_file else None,
        "collar": args.collar,
        "skip_overlap": not args.score_overlap,
        "oracle_speakers": args.oracle_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "files": [asdict(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_file(
    file: BenchmarkFile,
    *,
    args: argparse.Namespace,
    metric: Any,
) -> FileResult:
    """Run diarize and compute metrics for one file."""
    diarize_fn, get_audio_duration_fn = load_diarize()
    start_time = time.time()
    duration = get_audio_duration_fn(file.audio_path)
    num_speakers = file.gt_speakers if args.oracle_speakers else args.num_speakers

    result = diarize_fn(
        file.audio_path,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=num_speakers,
    )
    elapsed = time.time() - start_time

    if not result.segments:
        return FileResult(
            file_id=file.file_id,
            gt_speakers=file.gt_speakers,
            predicted_speakers=0,
            speaker_delta=-file.gt_speakers,
            der=None,
            elapsed_sec=round(elapsed, 2),
            duration=round(duration, 2),
            n_segments=0,
            error="no_segments",
        )

    reference = parse_rttm_lines_to_annotation(file.file_id, file.rttm_lines)
    hypothesis = result_to_annotation(file.file_id, result.segments)
    der = float(metric(reference, hypothesis) * 100)
    predicted_speakers = result.num_speakers

    return FileResult(
        file_id=file.file_id,
        gt_speakers=file.gt_speakers,
        predicted_speakers=predicted_speakers,
        speaker_delta=predicted_speakers - file.gt_speakers,
        der=round(der, 2),
        elapsed_sec=round(elapsed, 2),
        duration=round(duration or result.audio_duration, 2),
        n_segments=len(result.segments),
    )


def print_summary(results: list[FileResult]) -> None:
    """Print aggregate DER and speaker-count metrics."""
    valid = [result for result in results if result.der is not None]
    if not valid:
        print("No valid results.")
        return

    ders = [result.der for result in valid if result.der is not None]
    total_duration = sum(max(result.duration, 0.0) for result in valid)
    weighted_der = (
        sum((result.der or 0.0) * result.duration for result in valid) / total_duration
        if total_duration > 0
        else mean(ders)
    )

    exact = sum(1 for result in valid if result.speaker_delta == 0)
    within_1 = sum(1 for result in valid if abs(result.speaker_delta) <= 1)

    print("\nSummary")
    print(f"  Files:        {len(valid)}")
    print(f"  Weighted DER: {weighted_der:.2f}%")
    print(f"  Mean DER:     {mean(ders):.2f}%")
    print(f"  Median DER:   {median(ders):.2f}%")
    print(f"  Exact count:  {exact}/{len(valid)} ({100 * exact / len(valid):.0f}%)")
    print(f"  Within +/-1:  {within_1}/{len(valid)} ({100 * within_1 / len(valid):.0f}%)")

    by_gt: dict[int, list[FileResult]] = {}
    for result in valid:
        by_gt.setdefault(result.gt_speakers, []).append(result)

    print("\nBy ground-truth speaker count")
    print(f"  {'GT':>3s}  {'N':>3s}  {'DER':>7s}  {'Exact':>7s}  {'Bias':>7s}")
    for gt_speakers, rows in sorted(by_gt.items()):
        row_ders = [row.der for row in rows if row.der is not None]
        row_exact = sum(1 for row in rows if row.speaker_delta == 0)
        row_bias = mean(row.speaker_delta for row in rows)
        print(
            f"  {gt_speakers:>3d}  {len(rows):>3d}  "
            f"{mean(row_ders):>6.2f}%  {row_exact:>3d}/{len(rows):<3d}  "
            f"{row_bias:>+7.2f}"
        )


def print_file_inventory(files: list[BenchmarkFile]) -> None:
    """Print matched-file inventory without running inference."""
    if not files:
        return

    by_gt: dict[int, int] = {}
    for file in files:
        by_gt[file.gt_speakers] = by_gt.get(file.gt_speakers, 0) + 1

    print("\nGround-truth speaker distribution")
    print(f"  {'GT':>3s}  {'N':>3s}")
    for gt_speakers, count in sorted(by_gt.items()):
        print(f"  {gt_speakers:>3d}  {count:>3d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark diarize on audio + RTTM data.")
    parser.add_argument("--dataset", default="dataset", help="Dataset label used in output JSON")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory with audio files")
    parser.add_argument("--rttm-dir", type=Path, default=None, help="Directory with RTTM files")
    parser.add_argument("--rttm-file", type=Path, default=None, help="Combined RTTM file")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0=all)")
    parser.add_argument(
        "--file-id",
        action="append",
        default=[],
        help="Only run a specific file id; may be passed multiple times",
    )
    parser.add_argument(
        "--gt-min",
        type=int,
        default=0,
        help="Only files with at least N speakers",
    )
    parser.add_argument(
        "--gt-max",
        type=int,
        default=999,
        help="Only files with at most N speakers",
    )
    parser.add_argument("--min-speakers", type=int, default=1, help="Minimum auto speaker count")
    parser.add_argument("--max-speakers", type=int, default=20, help="Maximum auto speaker count")
    parser.add_argument("--num-speakers", type=int, default=None, help="Use fixed speaker count")
    parser.add_argument("--list-only", action="store_true", help="Only list matched files")
    parser.add_argument(
        "--oracle-speakers",
        action="store_true",
        help="Use each file's RTTM speaker count as num_speakers",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from --output results")
    parser.add_argument("--collar", type=float, default=0.25, help="DER collar in seconds")
    parser.add_argument(
        "--score-overlap",
        action="store_true",
        help="Score overlapped speech instead of skipping it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.oracle_speakers and args.num_speakers is not None:
        raise SystemExit("--oracle-speakers and --num-speakers are mutually exclusive")
    if (args.rttm_dir is None) == (args.rttm_file is None):
        raise SystemExit("Pass exactly one of --rttm-dir or --rttm-file")
    if not args.audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {args.audio_dir}")
    if args.rttm_dir is not None and not args.rttm_dir.exists():
        raise SystemExit(f"RTTM directory not found: {args.rttm_dir}")
    if args.rttm_file is not None and not args.rttm_file.exists():
        raise SystemExit(f"RTTM file not found: {args.rttm_file}")

    files, missing_audio = collect_files(
        args.audio_dir,
        args.rttm_dir,
        args.rttm_file,
        gt_min=args.gt_min,
        gt_max=args.gt_max,
    )
    if args.file_id:
        requested_ids = set(args.file_id)
        files = [file for file in files if file.file_id in requested_ids]
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Dataset:      {args.dataset}")
    print(f"Audio dir:    {args.audio_dir}")
    if args.rttm_dir is not None:
        print(f"RTTM dir:     {args.rttm_dir}")
    else:
        print(f"RTTM file:    {args.rttm_file}")
    print(f"Matched:      {len(files)} files")
    if missing_audio:
        print(f"Missing audio: {len(missing_audio)} RTTM files")

    if args.list_only:
        print_file_inventory(files)
        return

    results = load_existing_results(args.output) if args.resume else []
    done_ids = {result.file_id for result in results}
    if done_ids:
        print(f"Resuming:     {len(done_ids)} existing results")

    _, _, DiarizationErrorRate = load_pyannote()
    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=not args.score_overlap)
    total = len(files)
    for index, file in enumerate(files, 1):
        if file.file_id in done_ids:
            print(f"[{index:3d}/{total}] {file.file_id} skipped")
            continue

        print(f"[{index:3d}/{total}] {file.file_id} gt={file.gt_speakers}", end=" ", flush=True)
        try:
            result = run_file(file, args=args, metric=metric)
        except Exception as exc:  # noqa: BLE001 - benchmark should continue after bad files.
            result = FileResult(
                file_id=file.file_id,
                gt_speakers=file.gt_speakers,
                predicted_speakers=-1,
                speaker_delta=0,
                der=None,
                elapsed_sec=0.0,
                duration=0.0,
                n_segments=0,
                error=str(exc),
            )
            print(f"error={exc}")
        else:
            der_text = "n/a" if result.der is None else f"{result.der:.2f}%"
            print(
                f"pred={result.predicted_speakers} DER={der_text} time={result.elapsed_sec:.1f}s"
            )

        results.append(result)
        write_results(args.output, dataset=args.dataset, args=args, results=results)

    print_summary(results)


if __name__ == "__main__":
    main()
