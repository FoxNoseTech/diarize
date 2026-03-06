# Quick Start

## Installation

```bash
pip install diarize
```

Requires Python 3.9+. All models (Silero VAD, WeSpeaker) are downloaded
automatically on first use. `diarize` also installs a compatible
`torch/torchaudio` range automatically.

## Basic Usage

```python
from diarize import diarize

result = diarize("meeting.wav")

print(f"Found {result.num_speakers} speakers")
for seg in result.segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")
```

## Specifying Speaker Count

If you know the number of speakers, skip auto-detection:

```python
result = diarize("meeting.wav", num_speakers=3)
```

Or constrain the range:

```python
result = diarize("meeting.wav", min_speakers=2, max_speakers=5)
```

## Exporting Results

### RTTM format

```python
# Write to file
result.to_rttm("output.rttm")

# Get as string
rttm_string = result.to_rttm()
print(rttm_string)
```

### JSON-friendly dicts

```python
for item in result.to_list():
    print(item)
# {"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00"}
```

### Pydantic serialization

```python
data = result.model_dump()
# Full result as a dict, including segments, speakers, estimation_details
```

### Iteration

`DiarizeResult` is iterable:

```python
for seg in result:
    print(seg.speaker, seg.start, seg.end, seg.duration)

print(f"Total segments: {len(result)}")
```

## Speaker Estimation Details

When auto-detection is used, diagnostic information is available:

```python
result = diarize("meeting.wav")
if result.estimation_details:
    details = result.estimation_details
    print(f"Method: {details.method}")
    print(f"Estimated speakers: {details.best_k}")
    print(f"PCA dimensions: {details.pca_dim}")
    print(f"BIC scores: {details.k_bics}")
```

## Logging

`diarize` uses Python's standard `logging` module:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now diarize() will print progress
result = diarize("meeting.wav")
```

Set `level=logging.DEBUG` for detailed diagnostics (BIC scores per k, etc.).
