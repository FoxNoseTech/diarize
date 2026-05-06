# How It Works

`diarize` answers "who spoke when?" using a four-stage pipeline,
all running on CPU.

## Pipeline Overview

```
Audio File
    |
    v
[1] Silero VAD -----------> Speech segments (start, end)
    |
    v
[2] WeSpeaker ResNet34-LM -> 256-dim speaker embeddings
    |
    v
[3] GMM BIC + silhouette -> Estimated speaker count (k)
    |
    v
[4] Spectral + temporal smoothing -> Speaker timeline
    |
    v
DiarizeResult
```

## Stage 1: Voice Activity Detection

[Silero VAD](https://github.com/snakers4/silero-vad) (MIT licensed)
identifies which portions of the audio contain speech. Non-speech regions
(silence, music, noise) are discarded before embedding extraction.

The VAD returns a list of [`SpeechSegment`](api.md#diarize.utils.SpeechSegment)
objects with start/end timestamps in seconds.

**Key parameters:** threshold (default 0.45), minimum speech duration,
minimum silence duration, padding.

See: [`run_vad()`](api.md#diarize.vad.run_vad)

## Stage 2: Speaker Embedding Extraction

[WeSpeaker](https://github.com/wenet-e2e/wespeaker) ResNet34-LM
(Apache 2.0) extracts a 256-dimensional embedding vector for each speech
segment. Long segments are split using a sliding window (1.2s window,
0.6s step) so each window gets its own embedding. Segments shorter than
0.4s are skipped during extraction and assigned the nearest speaker label
later.

The model runs via ONNX Runtime, so no GPU is needed.

See: [`extract_embeddings()`](api.md#diarize.embeddings.extract_embeddings)

## Stage 3: Speaker Count Estimation

Unless the user provides `num_speakers`, the pipeline estimates how many
speakers are present in three steps:

**Step 1 --- Cosine similarity pre-check.** Compute pairwise cosine
similarities of the L2-normalised embeddings. If the 10th percentile
is above a threshold (0.16), all embeddings belong to the same speaker
and *k = 1* is returned immediately --- skipping the more expensive GMM
fitting entirely.

**Step 2 --- GMM BIC.** If the pre-check does not trigger, the pipeline
uses **Gaussian Mixture Models** with the **Bayesian Information
Criterion (BIC)**:

1. Project to 8 dimensions via PCA (optimal for GMM with full covariance)
2. For each candidate *k* (from `min_speakers` to `max_speakers`), fit
   `GaussianMixture(k, covariance_type="full")`
3. Select *k* with the lowest BIC score

The PCA=8 setting provides a good balance: stable estimation for 2--7
speakers while keeping computational cost low.

**Step 3 --- Silhouette refinement.** BIC is used as an anchor, then a
small neighbourhood around it is scored with silhouette over cosine
distance plus a small logarithmic bonus for larger *k*. The candidate
range is clamped by `min_speakers`, `max_speakers`, and the number of
available embeddings. This catches some BIC undercounts and overcounts
without searching the full range.

!!! warning
    For **8 or more speakers** the estimator can undercount.
    Pass ``num_speakers`` explicitly when the speaker count is known.
    See [Benchmarks --- Limitations](benchmarks.md#limitations).

See: [`estimate_speakers()`](api.md#diarize.clustering.estimate_speakers)
and [`cluster_auto()`](api.md#diarize.clustering.cluster_auto)

## Stage 4: Spectral Clustering

With the number of speakers determined, **Spectral Clustering** groups
the embedding vectors using cosine similarity as the affinity metric.
The cosine similarity matrix is rescaled to [0, 1] and passed to
scikit-learn's `SpectralClustering`.

The initial spectral labels are refined with spherical centroid
reassignment over L2-normalised embeddings. This preserves the selected
speaker count while reducing unstable label assignments.

For long VAD segments, overlapping embedding windows are decoded into
non-overlapping timeline intervals using window-center midpoints. Within
each VAD segment, centroid-scored Viterbi decoding applies a speaker
switch penalty to remove short label flicker. Sustained original label
runs are restored before final island cleanup, so the smoother targets
brief A-B-A fragments rather than real speaker turns. Adjacent intervals
assigned to the same speaker are merged, and short segments that were
skipped during embedding extraction are assigned the label of the nearest
speaker.

See: [`cluster_spectral()`](api.md#diarize.clustering.cluster_spectral)

## Using Individual Stages

Each stage is available as a standalone function for advanced use cases:

```python
from diarize.vad import run_vad
from diarize.embeddings import extract_embeddings
from diarize.clustering import estimate_speakers, cluster_spectral

# Run stages individually
speech_segments = run_vad("meeting.wav")
embeddings, subsegments = extract_embeddings("meeting.wav", speech_segments)
k, details = estimate_speakers(embeddings, min_k=2, max_k=10)
labels = cluster_spectral(embeddings, k)

# Use results
for sub, label in zip(subsegments, labels):
    print(f"[{sub.start:.1f}s - {sub.end:.1f}s] SPEAKER_{label:02d}")
```

## Dependencies

All components are permissively licensed:

| Component | License | Purpose |
|-----------|---------|---------|
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | Voice Activity Detection |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | Apache 2.0 | Speaker Embeddings (ONNX) |
| [scikit-learn](https://scikit-learn.org/) | BSD | Spectral Clustering, GMM, PCA |
| [PyTorch](https://pytorch.org/) | BSD | Audio loading, VAD inference |
| [Pydantic](https://docs.pydantic.dev/) | MIT | Data validation and models |
