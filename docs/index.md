# diarize

**Speaker diarization for Python — answers "who spoke when?" in any audio file.**

Runs on CPU. No GPU, no API keys, no account signup. Apache 2.0 licensed.

## Install

```bash
pip install diarize
```

Requires Python 3.9+. All models download automatically on first use.
`diarize` installs a compatible `torch/torchaudio` range automatically.

```python
from diarize import diarize

result = diarize("meeting.wav")
for seg in result.segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}")
```

## How diarize compares

| | diarize | pyannote (free) | pyannote (commercial) |
|---|---|---|---|
| License | Apache 2.0 | CC-BY-4.0 | Commercial |
| GPU required | No | No (7x slower on CPU) | No |
| HuggingFace account | No | Yes | Yes |
| Auto speaker count | Yes | Yes | Yes |
| DER (VoxConverse) | **~10.8%** | ~11.2% | ~8.5% |
| CPU speed (RTF) | **0.12** | 0.86 | --- |

DER and speed numbers for pyannote are from their
[benchmark page](https://huggingface.co/pyannote/speaker-diarization-3.1).
Full methodology: [Benchmarks](benchmarks.md).

## Next Steps

- [How It Works](how-it-works.md) --- pipeline architecture and algorithms
- [Benchmarks](benchmarks.md) --- VoxConverse evaluation, speed comparison, limitations
- [API Reference](api.md) --- full auto-generated API documentation

## License

Apache 2.0 License. All dependencies are permissively licensed (MIT, Apache 2.0, BSD).
