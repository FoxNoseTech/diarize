# diarize

[![PyPI version](https://img.shields.io/pypi/v/diarize?v=1)](https://pypi.org/project/diarize/)
[![Python versions](https://img.shields.io/pypi/pyversions/diarize?v=1)](https://pypi.org/project/diarize/)
[![License](https://img.shields.io/pypi/l/diarize?v=1)](LICENSE)
[![codecov](https://codecov.io/gh/FoxNoseTech/diarize/graph/badge.svg?v=1)](https://codecov.io/gh/FoxNoseTech/diarize)
[![CI](https://github.com/FoxNoseTech/diarize/actions/workflows/ci.yml/badge.svg)](https://github.com/FoxNoseTech/diarize/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-foxnosetech.github.io-blue)](https://foxnosetech.github.io/diarize/)

**Speaker diarization for Python — answers "who spoke when?" in any audio file.**

Runs on CPU. No GPU, no API keys, no account signup. Apache 2.0 licensed.

```bash
pip install diarize
```

```python
from diarize import diarize

result = diarize("meeting.wav")
for seg in result.segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")
```

**~10.8% DER** on VoxConverse (lower than pyannote's free models). Processes audio **~8x faster than real-time** on CPU. Automatically detects the number of speakers.

> Benchmarked on a single dataset ([VoxConverse](https://github.com/joonson/voxconverse)). Cross-dataset validation is [in progress](#roadmap).

## How diarize compares

| | diarize | pyannote (free) | pyannote (commercial) |
|---|---|---|---|
| License | Apache 2.0 | CC-BY-4.0 | Commercial |
| GPU required | No | No (7x slower on CPU) | No |
| HuggingFace account | No | Yes | Yes |
| Auto speaker count | Yes | Yes | Yes |
| DER (VoxConverse) | **~10.8%** | ~11.2% | ~8.5% |
| CPU speed (RTF) | **0.12** | 0.86 | — |
| Install | `pip install diarize` | `pip install pyannote.audio` | `pip install pyannote.audio` |

DER = Diarization Error Rate (lower is better). RTF = Real-Time Factor (lower is faster).
pyannote numbers are self-reported from their [benchmark page](https://huggingface.co/pyannote/speaker-diarization-3.1). Full methodology: [benchmarks](https://foxnosetech.github.io/diarize/benchmarks/).

## Quick Start

```python
from diarize import diarize

result = diarize("meeting.wav")

print(f"Found {result.num_speakers} speakers")
for seg in result.segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")

# Export to RTTM format
result.to_rttm("meeting.rttm")
```

Requires Python 3.9+. Supports WAV, MP3, FLAC, OGG, and other formats via soundfile/libsndfile.
`diarize` pins a compatible `torch/torchaudio` range during install, so no extra manual pinning is required.

📖 **[Full documentation](https://foxnosetech.github.io/diarize/)** — installation, API reference, architecture, benchmarks.

## API

```python
result = diarize("meeting.wav")                # auto-detect speakers
result = diarize("call.mp3", num_speakers=2)   # known speaker count
result = diarize("panel.flac", min_speakers=3, max_speakers=8)

result.segments      # [Segment(start=0.5, end=4.2, speaker='SPEAKER_00'), ...]
result.num_speakers  # 3
result.speakers      # ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
result.audio_duration  # 324.5

result.to_rttm("output.rttm")  # export to standard RTTM format
result.to_list()                # export as list of dicts (JSON-serializable)
```

Each `Segment` has `.start`, `.end`, `.speaker`, and `.duration` (all in seconds).

Full API reference: [documentation](https://foxnosetech.github.io/diarize/api/)

## How It Works

Four-stage pipeline, all CPU, all open-source:

1. **Silero VAD** (MIT) — detects speech segments
2. **WeSpeaker ResNet34-LM** (Apache 2.0) — extracts 256-dim speaker embeddings via ONNX
3. **GMM BIC** — estimates the number of speakers
4. **Spectral Clustering** (scikit-learn, BSD) — assigns speaker labels

Details: [How It Works](https://foxnosetech.github.io/diarize/how-it-works/)

## Benchmarks

Evaluated on [VoxConverse](https://github.com/joonson/voxconverse) dev set (216 files, 1–20 speakers):

### Diarization Error Rate (DER)

| System | Weighted DER | Notes |
|--------|----------|-------|
| pyannote precision-2 | ~8.5% | Commercial license |
| **diarize** | **~10.8%** | **Apache 2.0, CPU-only, no API key** |
| pyannote community-1 | ~11.2% | CC-BY-4.0, needs HF token |
| pyannote 3.1 (legacy) | ~11.2% | MIT, needs HF token |

### Speaker Count Estimation

| GT Speakers | Files | Exact Match | Within ±1 |
|-------------|-------|-------------|-----------|
| 1 | 22 | 91% | 95% |
| 2 | 44 | 70% | 91% |
| 3 | 35 | 69% | 97% |
| 4 | 24 | 54% | 88% |
| 5 | 31 | 32% | 87% |
| 6–7 | 29 | 45% | 79% |
| 8+ | 31 | 0% | 26% |
| **Overall** | **216** | **51%** | **81%** |

Full benchmark results, speed comparison, and methodology: [benchmarks](https://foxnosetech.github.io/diarize/benchmarks/).

## When to use something else

- **You need <9% DER.** pyannote's commercial model (precision-2) achieves ~8.5%. If accuracy is the top priority and you have budget, use that.
- **Your audio has 8+ speakers.** Automatic speaker count estimation degrades above 7 speakers. You can pass `num_speakers` explicitly, but test carefully.
- **You need overlapping speech detection.** diarize assigns each segment to one speaker. Overlapping speech is not modeled.
- **You need GPU-accelerated throughput.** diarize is CPU-only by design. For processing thousands of hours with GPU infrastructure, NeMo or pyannote on GPU will be faster.

## Roadmap

Current benchmarks are based on VoxConverse dev set only. We are actively working on:

- **Cross-dataset validation** — AMI, DIHARD III, CALLHOME, and other standard benchmarks in isolated environments
- **Speaker count estimation benchmarks** — comparison of speaker counting accuracy against other systems
- **Broader system comparison** — NeMo, WhisperX, and other diarization solutions
- **Streaming / real-time diarization** — live audio streams with real-time speaker detection
- **Speaker identification** — recognise known speakers across sessions using stored embeddings

## Logging

`diarize` uses Python's standard `logging` module:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

All dependencies are permissively licensed:
- Silero VAD: MIT
- WeSpeaker: Apache 2.0
- scikit-learn: BSD
- PyTorch: BSD

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/FoxNoseTech/diarize).
