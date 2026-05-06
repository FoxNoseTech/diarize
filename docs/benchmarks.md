# Benchmarks

Primary published numbers are evaluated on the
[VoxConverse](https://github.com/joonson/voxconverse) dev set
(216 files, 1--20 speakers per file). We also run preliminary
cross-dataset checks on AMI meetings to track generalisation.

## Speaker Count Estimation

| Metric | Result |
|--------|--------|
| Files | 216 |
| Exact match | 125/216 (58%) |
| Within +/-1 | 178/216 (82%) |

The automatic estimator is usually close, but exact counting remains the
main weak spot. Accuracy drops for many-speaker files --- see
[Limitations](#limitations) below.

## Diarization Error Rate (DER)

DER is the standard metric for speaker diarization, computed with
`collar=0.25` and `skip_overlap=True`.

| System | Weighted DER | Median DER | Notes |
|--------|----------|------------|-------|
| pyannote precision-2 | ~8.5% | -- | Commercial license |
| **diarize** | **~4.8%** | **~2.1%** | **Apache 2.0, CPU-only, no API key** |
| pyannote community-1 | ~11.2% | -- | CC-BY-4.0, needs HF token |
| pyannote 3.1 (legacy) | ~11.2% | -- | MIT, needs HF token |

pyannote DER numbers are self-reported from the
[pyannote benchmark page](https://huggingface.co/pyannote/speaker-diarization-3.1)
on VoxConverse v0.3.

!!! note "Dataset-specific result"
    On this VoxConverse dev evaluation, `diarize` reports lower weighted
    DER than the published pyannote VoxConverse figures, while requiring
    no HuggingFace token or account registration. Treat this as a
    VoxConverse-specific benchmark and compare on your own audio when accuracy
    is the top priority.

## Cross-Dataset Check: AMI

Preliminary AMI test-set evaluation uses 16 Mix-Headset meeting
recordings (4--9 speakers per file), RTTM annotations from the
standard AMI speaker-diarization benchmark, and the same DER settings
(``collar=0.25``, ``skip_overlap=True``).

| Metric | Result |
|--------|--------|
| Files | 16 |
| Weighted DER | 14.96% |
| Mean DER | 14.63% |
| Median DER | 14.18% |
| Speaker count exact match | 4/16 (25%) |
| Speaker count within +/-1 | 8/16 (50%) |

This confirms that meeting-domain audio is a harder case for automatic
speaker counting. The estimator often collapses 6+ speaker meetings to
4--5 speakers, even when aggregate DER remains moderate because some
ground-truth speakers have little speaking time.

## CPU Speed (Real Time Factor)

RTF = processing_time / audio_duration.  Lower is faster; RTF < 1.0 means
faster than real-time.

| System | Mean RTF | Median RTF | Notes |
|--------|----------|------------|-------|
| **diarize** | **0.12** | **0.12** | **~7x faster than community-1** |
| pyannote community-1 | 0.82 | 0.86 | ~2x faster than 3.1 |
| pyannote 3.1 (legacy) | 1.74 | 1.83 | Slower than real-time on CPU |

Measured on VoxConverse dev files on Apple M2 Pro / M2 Max
(CPU only, no GPU).  All systems were warm-started (models pre-loaded).

!!! note "Apples-to-apples"
    All systems ran on the **same files** with `torch.device("cpu")`.
    `diarize` uses ONNX Runtime for speaker embeddings; pyannote uses
    PyTorch neural networks (segmentation + embedding models).

!!! warning "pyannote 3.1 is slower than real-time on CPU"
    With RTF > 1.0, pyannote 3.1 **cannot process audio in real-time**
    on CPU.  A 10-minute recording takes ~18 minutes to diarize vs
    ~1.2 minutes with `diarize`.  Community-1 is faster (RTF ~0.86)
    but still ~7x slower than `diarize`.

## Methodology

- **Dataset:** VoxConverse dev set --- 216 audio files recorded from
  YouTube debates, news shows, and other multi-speaker media.
- **Ground truth:** RTTM annotations from the
  [official repository](https://github.com/joonson/voxconverse).
- **Evaluation:** [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/)
  `DiarizationErrorRate` with standard parameters.
- **Speed benchmark:** 25 files from VoxConverse dev set, stratified by
  duration.  Wall-clock time measured with `time.time()` after model
  warm-up.  RTF = processing_time / audio_duration.
- **Hardware:** Apple M2 Pro, macOS, CPU only (no GPU).

## Reproducing and Extending Benchmarks

The repository includes a dataset-agnostic RTTM runner for local
experiments:

```bash
python scripts/benchmark_rttm.py \
  --dataset voxconverse-dev \
  --audio-dir /path/to/voxconverse/dev/audio \
  --rttm-dir /path/to/voxconverse/rttm_annotations/dev \
  --output results_voxconverse_dev.json
```

It also supports combined RTTM files and targeted diagnostics:

```bash
python scripts/benchmark_rttm.py \
  --dataset ami-test \
  --audio-dir /path/to/ami/mix-headset/test \
  --rttm-file /path/to/AMI.SpeakerDiarization.Benchmark.test.rttm \
  --oracle-speakers \
  --file-id IS1009a
```

Use ``--oracle-speakers`` to isolate speaker assignment and clustering
quality when the true speaker count is known. Use ``--list-only`` to
verify audio/RTTM matching without running inference.

## Limitations

!!! warning "Speaker count > 7"
    The GMM BIC speaker-count estimator with silhouette refinement is
    usually close on VoxConverse dev, but many-speaker files remain the
    hardest case. For **8 or more speakers** it can undercount and
    produce higher DER.
    If you know your audio has many speakers, pass ``num_speakers``
    explicitly:

    ```python
    result = diarize("panel.wav", num_speakers=12)
    ```

**Known limitations:**

- **Many speakers (8+):** Automatic speaker count estimation degrades.
  Use ``num_speakers`` when the speaker count is known.
- **Speaker label switching / fragmentation:** Temporal smoothing reduces
  short label jumps, but on noisy real-world audio one actual speaker can
  still be split across multiple ``SPEAKER_XX`` labels. This is mostly a
  clustering and embedding-assignment limitation, and it is visible in
  transcripts even when aggregate DER looks acceptable.
- **Overlapping speech:** DER is computed with ``skip_overlap=True``.
  The pipeline does not model overlapping speech --- when two people
  talk simultaneously, only one is labelled.
- **Short utterances (<&nbsp;0.4 s):** Segments shorter than 0.4 seconds
  are not embedded directly; they are assigned the label of the nearest
  speaker, which can cause errors at speaker boundaries.

## Future Work

!!! info "Cross-dataset validation in progress"
    VoxConverse remains the primary published benchmark. AMI is now used
    as an additional meeting-domain check, and more datasets are needed
    before making broad accuracy claims.

**Planned evaluation:**

- **Cross-dataset validation** --- DIHARD III, CALLHOME, and other
  standard benchmarks, run in isolated environments with controlled
  CPU/memory limits.
- **Speaker count estimation comparison** --- dedicated benchmarks comparing
  speaker counting accuracy against pyannote and other systems across
  datasets.
- **Broader system comparison** --- benchmark against NeMo, WhisperX, and
  other open-source diarization solutions with verified, reproducible results.

**Planned features:**

- **Streaming / real-time diarization** --- process live audio streams with
  real-time speaker detection and embedding extraction.
- **Speaker identification** --- store and compare speaker embeddings to
  recognise known speakers across sessions.
