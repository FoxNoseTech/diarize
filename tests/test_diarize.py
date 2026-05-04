"""Tests for the diarize package."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

# ── Segment model ────────────────────────────────────────────────────────────


class TestSegment:
    """Tests for the Segment Pydantic model."""

    def test_create_segment(self):
        from diarize.utils import Segment

        seg = Segment(start=1.0, end=3.5, speaker="SPEAKER_00")
        assert seg.start == 1.0
        assert seg.end == 3.5
        assert seg.speaker == "SPEAKER_00"

    def test_duration(self):
        from diarize.utils import Segment

        seg = Segment(start=1.0, end=3.5, speaker="SPEAKER_00")
        assert seg.duration == pytest.approx(2.5)

    def test_frozen(self):
        from diarize.utils import Segment

        seg = Segment(start=1.0, end=3.5, speaker="SPEAKER_00")
        with pytest.raises(ValidationError):
            seg.start = 2.0  # type: ignore[misc]

    def test_invalid_times(self):
        from diarize.utils import Segment

        with pytest.raises(ValueError, match="end.*must be >= start"):
            Segment(start=5.0, end=2.0, speaker="SPEAKER_00")

    def test_empty_speaker_rejected(self):
        from diarize.utils import Segment

        with pytest.raises(ValidationError):
            Segment(start=0.0, end=1.0, speaker="")

    def test_negative_start_rejected(self):
        from diarize.utils import Segment

        with pytest.raises(ValidationError):
            Segment(start=-1.0, end=1.0, speaker="SPEAKER_00")


# ── SpeechSegment model ──────────────────────────────────────────────────────


class TestSpeechSegment:
    """Tests for the SpeechSegment Pydantic model."""

    def test_create(self):
        from diarize.utils import SpeechSegment

        seg = SpeechSegment(start=0.5, end=3.0)
        assert seg.start == 0.5
        assert seg.end == 3.0
        assert seg.duration == pytest.approx(2.5)

    def test_invalid_times(self):
        from diarize.utils import SpeechSegment

        with pytest.raises(ValueError, match="end.*must be >= start"):
            SpeechSegment(start=5.0, end=2.0)


# ── SubSegment model ─────────────────────────────────────────────────────────


class TestSubSegment:
    """Tests for the SubSegment Pydantic model."""

    def test_create(self):
        from diarize.utils import SubSegment

        sub = SubSegment(start=1.0, end=2.2, parent_idx=3)
        assert sub.start == 1.0
        assert sub.end == 2.2
        assert sub.parent_idx == 3

    def test_negative_parent_idx_rejected(self):
        from diarize.utils import SubSegment

        with pytest.raises(ValidationError):
            SubSegment(start=0.0, end=1.0, parent_idx=-1)


# ── SpeakerEstimationDetails ─────────────────────────────────────────────────


class TestSpeakerEstimationDetails:
    """Tests for the SpeakerEstimationDetails Pydantic model."""

    def test_create(self):
        from diarize.utils import SpeakerEstimationDetails

        d = SpeakerEstimationDetails(method="gmm_bic", best_k=3, pca_dim=8)
        assert d.method == "gmm_bic"
        assert d.best_k == 3
        assert d.pca_dim == 8
        assert d.k_bics == {}
        assert d.reason is None

    def test_with_reason(self):
        from diarize.utils import SpeakerEstimationDetails

        d = SpeakerEstimationDetails(method="gmm_bic", best_k=1, reason="too_few_samples")
        assert d.reason == "too_few_samples"

    def test_serialization(self):
        from diarize.utils import SpeakerEstimationDetails

        d = SpeakerEstimationDetails(
            method="gmm_bic",
            best_k=2,
            pca_dim=8,
            k_bics={1: 100.0, 2: 90.0, 3: 95.0},
        )
        data = d.model_dump()
        assert data["best_k"] == 2
        assert data["k_bics"] == {1: 100.0, 2: 90.0, 3: 95.0}


# ── DiarizeResult ────────────────────────────────────────────────────────────


class TestDiarizeResult:
    """Tests for the DiarizeResult class."""

    def test_empty_result(self):
        from diarize.utils import DiarizeResult

        result = DiarizeResult()
        assert result.num_speakers == 0
        assert result.speakers == []
        assert len(result.segments) == 0

    def test_speakers_property(self):
        from diarize.utils import DiarizeResult, Segment

        segments = [
            Segment(start=0.0, end=1.0, speaker="SPEAKER_01"),
            Segment(start=1.0, end=2.0, speaker="SPEAKER_00"),
            Segment(start=2.0, end=3.0, speaker="SPEAKER_01"),
        ]
        result = DiarizeResult(segments=segments)
        assert result.num_speakers == 2
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]

    def test_to_rttm_string(self):
        from diarize.utils import DiarizeResult, Segment

        segments = [
            Segment(start=0.5, end=3.0, speaker="SPEAKER_00"),
            Segment(start=3.5, end=5.0, speaker="SPEAKER_01"),
        ]
        result = DiarizeResult(segments=segments, audio_path="/tmp/test.wav", audio_duration=5.0)
        rttm = result.to_rttm()

        lines = rttm.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("SPEAKER test 1")
        assert "SPEAKER_00" in lines[0]
        assert "SPEAKER_01" in lines[1]

    def test_to_rttm_no_audio_path(self):
        from diarize.utils import DiarizeResult, Segment

        result = DiarizeResult(segments=[Segment(start=0, end=1, speaker="SPEAKER_00")])
        rttm = result.to_rttm()
        assert "SPEAKER audio 1" in rttm

    def test_to_rttm_empty(self):
        from diarize.utils import DiarizeResult

        result = DiarizeResult()
        rttm = result.to_rttm()
        assert rttm == ""

    def test_to_rttm_file(self):
        from diarize.utils import DiarizeResult, Segment

        segments = [
            Segment(start=0.0, end=1.0, speaker="SPEAKER_00"),
        ]
        result = DiarizeResult(segments=segments, audio_path="audio.wav", audio_duration=1.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as f:
            tmp_path = f.name

        result.to_rttm(tmp_path)
        content = Path(tmp_path).read_text()
        assert "SPEAKER audio 1" in content
        Path(tmp_path).unlink()

    def test_to_list(self):
        from diarize.utils import DiarizeResult, Segment

        segments = [
            Segment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, speaker="SPEAKER_01"),
        ]
        result = DiarizeResult(segments=segments)
        lst = result.to_list()

        assert len(lst) == 2
        assert lst[0] == {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
        assert lst[1] == {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}

    def test_repr(self):
        from diarize.utils import DiarizeResult, Segment

        result = DiarizeResult(
            segments=[Segment(start=0, end=1, speaker="SPEAKER_00")],
            audio_duration=60.0,
        )
        r = repr(result)
        assert "speakers=1" in r
        assert "segments=1" in r

    def test_repr_no_duration(self):
        from diarize.utils import DiarizeResult

        result = DiarizeResult()
        r = repr(result)
        assert "duration=?" in r

    def test_iteration(self):
        from diarize.utils import DiarizeResult, Segment

        segs = [
            Segment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, speaker="SPEAKER_01"),
        ]
        result = DiarizeResult(segments=segs)
        assert len(result) == 2
        assert list(result) == segs

    def test_with_estimation_details(self):
        from diarize.utils import DiarizeResult, SpeakerEstimationDetails

        details = SpeakerEstimationDetails(method="gmm_bic", best_k=3, pca_dim=8)
        result = DiarizeResult(estimation_details=details)
        assert result.estimation_details is not None
        assert result.estimation_details.best_k == 3

    def test_model_dump(self):
        from diarize.utils import DiarizeResult, Segment

        result = DiarizeResult(
            segments=[Segment(start=0.0, end=1.0, speaker="SPEAKER_00")],
            audio_path="test.wav",
            audio_duration=1.0,
        )
        data = result.model_dump()
        assert data["audio_path"] == "test.wav"
        assert len(data["segments"]) == 1
        assert data["segments"][0]["speaker"] == "SPEAKER_00"
        assert data["num_speakers"] == 1
        assert data["speakers"] == ["SPEAKER_00"]


# ── format_timestamp ─────────────────────────────────────────────────────────


class TestFormatTimestamp:
    """Tests for format_timestamp utility."""

    def test_seconds_only(self):
        from diarize.utils import format_timestamp

        assert format_timestamp(45) == "00:45"

    def test_minutes(self):
        from diarize.utils import format_timestamp

        assert format_timestamp(125) == "02:05"

    def test_hours(self):
        from diarize.utils import format_timestamp

        assert format_timestamp(3661) == "01:01:01"


# ── get_audio_duration ───────────────────────────────────────────────────────


class TestGetAudioDuration:
    """Tests for get_audio_duration utility, covering all branches."""

    def test_soundfile_success(self):
        from diarize.utils import get_audio_duration

        mock_info = MagicMock()
        mock_info.duration = 42.5
        with patch("diarize.utils.sf.info", return_value=mock_info):
            assert get_audio_duration("test.wav") == 42.5

    def test_torchaudio_fallback(self):
        from diarize.utils import get_audio_duration

        mock_ta_info = MagicMock()
        mock_ta_info.num_frames = 160000
        mock_ta_info.sample_rate = 16000

        with (
            patch("diarize.utils.sf.info", side_effect=RuntimeError("fail")),
            patch.dict("sys.modules", {"torchaudio": MagicMock()}),
        ):
            import sys

            sys.modules["torchaudio"].info.return_value = mock_ta_info
            assert get_audio_duration("test.wav") == pytest.approx(10.0)

    def test_both_fail_returns_zero(self):
        from diarize.utils import get_audio_duration

        with patch("diarize.utils.sf.info", side_effect=RuntimeError("fail")):
            # torchaudio also fails (import or info call)
            result = get_audio_duration("/nonexistent/path.wav")
            assert result == 0.0


# ── estimate_speakers (GMM BIC) ──────────────────────────────────────────────


class TestEstimateSpeakers:
    """Tests for GMM BIC speaker count estimation."""

    def test_synthetic_2_clusters(self):
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        c1 = rng.randn(50, 256) + np.array([3.0] * 256)
        c2 = rng.randn(50, 256) + np.array([-3.0] * 256)
        embeddings = np.vstack([c1, c2])

        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        assert k == 2
        assert details.best_k == 2
        assert details.pca_dim is not None

    def test_synthetic_3_clusters(self):
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        c1 = rng.randn(40, 256) + np.array([5.0] * 128 + [0.0] * 128)
        c2 = rng.randn(40, 256) + np.array([0.0] * 128 + [5.0] * 128)
        c3 = rng.randn(40, 256) + np.array([-5.0] * 128 + [-5.0] * 128)
        embeddings = np.vstack([c1, c2, c3])

        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        assert k == 3
        assert details.method == "gmm_bic"

    def test_too_few_samples(self):
        from diarize.clustering import estimate_speakers

        embeddings = np.random.randn(3, 256)
        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        assert k == 1
        assert details.reason == "too_few_samples"

    def test_returns_pydantic_model(self):
        from diarize.clustering import estimate_speakers
        from diarize.utils import SpeakerEstimationDetails

        rng = np.random.RandomState(42)
        embeddings = rng.randn(30, 256)

        _, details = estimate_speakers(embeddings, min_k=1, max_k=5)
        assert isinstance(details, SpeakerEstimationDetails)
        assert details.model_dump() is not None

    def test_gmm_exception_returns_fallback(self):
        """When GMM raises an exception for all k values, return min_k with gmm_failed."""
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 256)

        with patch(
            "diarize.clustering.GaussianMixture",
            side_effect=RuntimeError("GMM failed"),
        ):
            k, details = estimate_speakers(embeddings, min_k=2, max_k=5)
            assert k == 2
            assert details.reason == "gmm_failed"

    def test_k_bics_populated(self):
        """Verify k_bics dict is populated with BIC values."""
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        c1 = rng.randn(50, 256) + 5
        c2 = rng.randn(50, 256) - 5
        embeddings = np.vstack([c1, c2])

        _, details = estimate_speakers(embeddings, min_k=1, max_k=5)
        assert len(details.k_bics) > 0
        assert all(isinstance(v, float) for v in details.k_bics.values())


# ── Single-speaker pre-check ─────────────────────────────────────────────────


class TestSingleSpeakerPrecheck:
    """Tests for cosine similarity single-speaker pre-check in estimate_speakers."""

    def test_single_cluster_detected(self):
        """A tight single cluster should return k=1 via cosine pre-check."""
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        # Single tight cluster: all embeddings close together
        embeddings = rng.randn(50, 256) * 0.3 + np.array([1.0] * 256)
        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        assert k == 1
        assert details.reason == "cosine_similarity_single_speaker"
        assert details.cosine_sim_p10 is not None
        assert details.cosine_sim_p10 >= 0.16

    def test_two_clusters_not_overridden(self):
        """Well-separated clusters should NOT trigger the pre-check."""
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        c1 = rng.randn(50, 256) + np.array([3.0] * 256)
        c2 = rng.randn(50, 256) + np.array([-3.0] * 256)
        embeddings = np.vstack([c1, c2])
        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        assert k >= 2
        assert details.reason != "cosine_similarity_single_speaker"

    def test_min_k_2_skips_precheck(self):
        """When min_k=2, the cosine pre-check should not force k=1."""
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        embeddings = rng.randn(50, 256) * 0.3 + np.array([1.0] * 256)
        k, details = estimate_speakers(embeddings, min_k=2, max_k=10)
        assert k >= 2

    def test_precheck_gmm_not_run(self):
        """When pre-check triggers, GMM BIC should not run (k_bics empty)."""
        from diarize.clustering import estimate_speakers

        rng = np.random.RandomState(42)
        embeddings = rng.randn(30, 256) * 0.2 + np.array([2.0] * 256)
        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        if details.reason == "cosine_similarity_single_speaker":
            assert details.k_bics == {}
            assert details.pca_dim is None


# ── cluster_spectral ─────────────────────────────────────────────────────────


class TestClusterSpectral:
    """Tests for Spectral Clustering."""

    def test_spherical_refinement_corrects_noisy_labels(self):
        from diarize.clustering import _refine_labels_spherical

        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        noisy_labels = np.array([0, 0, 0, 1])

        labels = _refine_labels_spherical(embeddings, noisy_labels)
        assert len(set(labels)) == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_basic_clustering(self):
        from diarize.clustering import cluster_spectral

        rng = np.random.RandomState(42)
        c1 = rng.randn(20, 256) + 5
        c2 = rng.randn(20, 256) - 5
        embeddings = np.vstack([c1, c2])

        labels = cluster_spectral(embeddings, k=2)
        assert len(labels) == 40
        assert len(set(labels)) == 2
        assert len(set(labels[:20])) == 1
        assert len(set(labels[20:])) == 1


# ── cluster_speakers ─────────────────────────────────────────────────────────


class TestClusterSpeakers:
    """Tests for the high-level cluster_speakers wrapper."""

    def test_silhouette_candidates_cover_bic_neighbourhood(self):
        from diarize.clustering import _silhouette_candidate_counts

        assert _silhouette_candidate_counts(
            k=5,
            n_embeddings=20,
            min_speakers=1,
            max_speakers=10,
        ) == [
            3,
            4,
            5,
            6,
            7,
            8,
        ]

    def test_silhouette_candidates_skip_single_speaker(self):
        from diarize.clustering import _silhouette_candidate_counts

        assert (
            _silhouette_candidate_counts(
                k=1,
                n_embeddings=20,
                min_speakers=1,
                max_speakers=10,
            )
            == []
        )
        assert (
            _silhouette_candidate_counts(
                k=3,
                n_embeddings=3,
                min_speakers=1,
                max_speakers=10,
            )
            == []
        )

    def test_silhouette_candidates_respect_min_speakers(self):
        from diarize.clustering import _silhouette_candidate_counts

        assert _silhouette_candidate_counts(
            k=5,
            n_embeddings=20,
            min_speakers=4,
            max_speakers=10,
        ) == [4, 5, 6, 7, 8]

    def test_fixed_num_speakers(self):
        from diarize.clustering import cluster_speakers

        rng = np.random.RandomState(42)
        embeddings = rng.randn(30, 256)
        labels, details = cluster_speakers(embeddings, num_speakers=2)
        assert len(labels) == 30
        assert len(set(labels)) == 2
        assert details is None

    def test_auto_returns_details(self):
        from diarize.clustering import cluster_speakers
        from diarize.utils import SpeakerEstimationDetails

        rng = np.random.RandomState(42)
        c1 = rng.randn(30, 256) + 5
        c2 = rng.randn(30, 256) - 5
        embeddings = np.vstack([c1, c2])

        labels, details = cluster_speakers(embeddings, min_speakers=1, max_speakers=5)
        assert len(labels) == 60
        assert isinstance(details, SpeakerEstimationDetails)

    def test_single_embedding(self):
        from diarize.clustering import cluster_speakers

        embeddings = np.random.randn(1, 256)
        labels, details = cluster_speakers(embeddings)
        assert len(labels) == 1
        assert labels[0] == 0
        assert details is None


# ── _build_diarization_segments ──────────────────────────────────────────────


class TestBuildDiarizationSegments:
    """Tests for segment assembly from subsegments and labels."""

    def test_basic_assembly(self):
        from diarize import _build_diarization_segments
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=3.0), SpeechSegment(start=4.0, end=7.0)]
        subs = [
            SubSegment(start=0.0, end=1.2, parent_idx=0),
            SubSegment(start=0.6, end=1.8, parent_idx=0),
            SubSegment(start=4.0, end=5.2, parent_idx=1),
            SubSegment(start=4.6, end=5.8, parent_idx=1),
        ]
        labels = np.array([0, 0, 1, 1])

        segments = _build_diarization_segments(speech, subs, labels)
        assert len(segments) >= 1
        assert all(s.speaker in ("SPEAKER_00", "SPEAKER_01") for s in segments)

    def test_merge_adjacent_same_speaker(self):
        from diarize import _build_diarization_segments
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=5.0)]
        subs = [
            SubSegment(start=0.0, end=1.2, parent_idx=0),
            SubSegment(start=1.0, end=2.2, parent_idx=0),
            SubSegment(start=2.0, end=3.2, parent_idx=0),
        ]
        labels = np.array([0, 0, 0])  # all same speaker

        segments = _build_diarization_segments(speech, subs, labels)
        assert len(segments) == 1  # merged into one
        assert segments[0].speaker == "SPEAKER_00"

    def test_different_speakers_not_merged(self):
        from diarize import _build_diarization_segments
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=5.0)]
        subs = [
            SubSegment(start=0.0, end=1.2, parent_idx=0),
            SubSegment(start=1.0, end=2.2, parent_idx=0),
        ]
        labels = np.array([0, 1])  # different speakers

        segments = _build_diarization_segments(speech, subs, labels)
        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"

    def test_long_segment_windows_become_non_overlapping(self):
        from diarize import _build_diarization_segments
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=8.0)]
        subs = [
            SubSegment(start=0.0, end=1.2, parent_idx=0),
            SubSegment(start=0.6, end=1.8, parent_idx=0),
            SubSegment(start=1.2, end=2.4, parent_idx=0),
            SubSegment(start=1.8, end=3.0, parent_idx=0),
        ]
        labels = np.array([0, 1, 0, 1])

        segments = _build_diarization_segments(speech, subs, labels)
        assert len(segments) >= 2
        assert all(a.end <= b.start for a, b in zip(segments, segments[1:]))

    def test_short_segment_assigned_nearest_speaker(self):
        """VAD segments without embeddings should get the nearest speaker."""
        from diarize import _build_diarization_segments
        from diarize.utils import SpeechSegment, SubSegment

        speech = [
            SpeechSegment(start=0.0, end=2.0),  # has embeddings (idx=0)
            SpeechSegment(start=2.5, end=2.8),  # short, no embeddings (idx=1)
            SpeechSegment(start=5.0, end=7.0),  # has embeddings (idx=2)
        ]
        subs = [
            SubSegment(start=0.0, end=1.2, parent_idx=0),
            SubSegment(start=5.0, end=6.2, parent_idx=2),
        ]
        labels = np.array([0, 1])

        segments = _build_diarization_segments(speech, subs, labels)
        # The short segment at 2.5-2.8 should inherit SPEAKER_00 from the nearest speech.
        covering = [s for s in segments if s.start <= 2.5 and s.end >= 2.8]
        assert len(covering) == 1
        assert covering[0].speaker == "SPEAKER_00"

    def test_no_speech_segments(self):
        """No speech and no subsegments returns empty."""
        from diarize import _build_diarization_segments

        segments = _build_diarization_segments([], [], np.array([]))
        assert segments == []

    def test_gap_prevents_merge(self):
        """Segments with a gap >= 0.5s should not be merged even if same speaker."""
        from diarize import _build_diarization_segments
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=1.0), SpeechSegment(start=2.0, end=3.0)]
        subs = [
            SubSegment(start=0.0, end=1.0, parent_idx=0),
            SubSegment(start=2.0, end=3.0, parent_idx=1),
        ]
        labels = np.array([0, 0])  # same speaker

        segments = _build_diarization_segments(speech, subs, labels)
        # Gap of 1.0s > 0.5s, so should NOT merge
        assert len(segments) == 2


# ── run_vad (mocked) ─────────────────────────────────────────────────────────


class TestRunVad:
    """Tests for run_vad with mocked Silero VAD."""

    def _make_silero_mock(self, timestamps):
        """Create a mock silero_vad module."""
        mock_module = MagicMock()
        mock_module.load_silero_vad.return_value = MagicMock()
        mock_module.read_audio.return_value = MagicMock()
        mock_module.get_speech_timestamps.return_value = timestamps
        return mock_module

    def test_run_vad_returns_speech_segments(self):
        from diarize.utils import SpeechSegment

        mock_timestamps = [
            {"start": 0.5, "end": 2.0},
            {"start": 3.0, "end": 5.5},
        ]
        mock_silero = self._make_silero_mock(mock_timestamps)

        with patch.dict("sys.modules", {"silero_vad": mock_silero}):
            from diarize.vad import run_vad

            segments = run_vad("test.wav")

        assert len(segments) == 2
        assert isinstance(segments[0], SpeechSegment)
        assert segments[0].start == 0.5
        assert segments[0].end == 2.0
        assert segments[1].start == 3.0
        assert segments[1].end == 5.5

    def test_run_vad_empty(self):
        mock_silero = self._make_silero_mock([])

        with patch.dict("sys.modules", {"silero_vad": mock_silero}):
            from diarize.vad import run_vad

            segments = run_vad("test.wav")

        assert segments == []

    def test_run_vad_custom_params(self):
        mock_silero = self._make_silero_mock([])

        with patch.dict("sys.modules", {"silero_vad": mock_silero}):
            from diarize.vad import run_vad

            run_vad("test.wav", threshold=0.6, min_speech_duration_ms=500)
            kwargs = mock_silero.get_speech_timestamps.call_args
            assert kwargs[1]["threshold"] == 0.6
            assert kwargs[1]["min_speech_duration_ms"] == 500


# ── extract_embeddings (mocked) ──────────────────────────────────────────────


class TestExtractEmbeddings:
    """Tests for extract_embeddings with mocked WeSpeaker."""

    def _mock_wespeaker(self):
        """Create a mock WeSpeaker model returning 256-dim embeddings."""
        mock_model = MagicMock()
        mock_model.extract_embedding.return_value = np.random.randn(256)
        return mock_model

    def test_basic_extraction(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        speech = [SpeechSegment(start=0.0, end=2.0)]
        audio = np.random.randn(32000).astype(np.float32)  # 2s at 16kHz

        mock_model = self._mock_wespeaker()
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        assert embeddings.shape[1] == 256
        assert len(subs) > 0

    def test_short_segment_skipped(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        # Segment shorter than MIN_SEGMENT_DURATION (0.4s)
        speech = [SpeechSegment(start=0.0, end=0.2)]
        audio = np.random.randn(3200).astype(np.float32)

        mock_model = self._mock_wespeaker()
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        assert len(embeddings) == 0
        assert subs == []

    def test_long_segment_windowed(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        # Segment > EMBEDDING_WINDOW * 1.5 = 1.8s -> will be windowed
        speech = [SpeechSegment(start=0.0, end=5.0)]
        audio = np.random.randn(80000).astype(np.float32)  # 5s at 16kHz

        mock_model = self._mock_wespeaker()
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        # 5s segment with 1.2s window, 0.6s step -> multiple windows
        assert embeddings.shape[0] > 1
        assert all(s.parent_idx == 0 for s in subs)

    def test_stereo_to_mono(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        speech = [SpeechSegment(start=0.0, end=1.5)]
        stereo_audio = np.random.randn(24000, 2).astype(np.float32)

        mock_model = self._mock_wespeaker()
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(stereo_audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        assert embeddings.shape[1] == 256

    def test_embedding_extraction_failure(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        speech = [SpeechSegment(start=0.0, end=2.0)]
        audio = np.random.randn(32000).astype(np.float32)

        mock_model = MagicMock()
        mock_model.extract_embedding.side_effect = RuntimeError("model error")
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        assert len(embeddings) == 0
        assert subs == []

    def test_2d_embedding_squeezed(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        speech = [SpeechSegment(start=0.0, end=1.5)]
        audio = np.random.randn(24000).astype(np.float32)

        mock_model = MagicMock()
        # Return 2D embedding (1, 256) which should be squeezed to (256,)
        mock_model.extract_embedding.return_value = np.random.randn(1, 256)
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        assert embeddings.shape == (1, 256)

    def test_none_embedding_skipped(self):
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        speech = [SpeechSegment(start=0.0, end=1.5)]
        audio = np.random.randn(24000).astype(np.float32)

        mock_model = MagicMock()
        mock_model.extract_embedding.return_value = None
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
            patch("diarize.embeddings.sf.write"),
        ):
            embeddings, subs = extract_embeddings("test.wav", speech)

        assert len(embeddings) == 0
        assert subs == []


# ── diarize() pipeline (mocked) ──────────────────────────────────────────────


class TestDiarizePipeline:
    """Tests for the main diarize() function with mocked components."""

    def test_full_pipeline(self):
        from diarize import diarize
        from diarize.utils import SpeakerEstimationDetails, SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=3.0), SpeechSegment(start=4.0, end=7.0)]
        subs = [
            SubSegment(start=0.0, end=1.2, parent_idx=0),
            SubSegment(start=4.0, end=5.2, parent_idx=1),
        ]
        embeddings = np.random.randn(2, 256)
        labels = np.array([0, 1])
        details = SpeakerEstimationDetails(method="gmm_bic", best_k=2, pca_dim=8)

        with (
            patch("diarize.get_audio_duration", return_value=10.0),
            patch("diarize.run_vad", return_value=speech),
            patch("diarize.extract_embeddings", return_value=(embeddings, subs)),
            patch("diarize.cluster_speakers", return_value=(labels, details)),
        ):
            result = diarize("test.wav")

        assert result.num_speakers == 2
        assert len(result.segments) >= 2
        assert result.audio_duration == 10.0
        assert result.estimation_details is not None
        assert result.estimation_details.best_k == 2

    def test_no_speech_detected(self):
        from diarize import diarize

        with (
            patch("diarize.get_audio_duration", return_value=5.0),
            patch("diarize.run_vad", return_value=[]),
        ):
            result = diarize("silence.wav")

        assert result.num_speakers == 0
        assert result.segments == []
        assert result.audio_duration == 5.0

    def test_no_embeddings_extracted(self):
        from diarize import diarize
        from diarize.utils import SpeechSegment

        speech = [SpeechSegment(start=0.0, end=0.2)]  # too short for embeddings

        with (
            patch("diarize.get_audio_duration", return_value=5.0),
            patch("diarize.run_vad", return_value=speech),
            patch("diarize.extract_embeddings", return_value=(np.array([]), [])),
        ):
            result = diarize("short_speech.wav")

        assert result.num_speakers == 0
        assert result.segments == []

    def test_with_fixed_num_speakers(self):
        from diarize import diarize
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=5.0)]
        subs = [SubSegment(start=0.0, end=1.2, parent_idx=0)]
        embeddings = np.random.randn(1, 256)
        labels = np.array([0])

        with (
            patch("diarize.get_audio_duration", return_value=5.0),
            patch("diarize.run_vad", return_value=speech),
            patch("diarize.extract_embeddings", return_value=(embeddings, subs)),
            patch("diarize.cluster_speakers", return_value=(labels, None)),
        ):
            result = diarize("test.wav", num_speakers=3)

        assert result.estimation_details is None

    def test_path_object_accepted(self):
        from diarize import diarize
        from diarize.utils import SpeechSegment, SubSegment

        speech = [SpeechSegment(start=0.0, end=2.0)]
        subs = [SubSegment(start=0.0, end=1.2, parent_idx=0)]
        embeddings = np.random.randn(1, 256)

        with (
            patch("diarize.get_audio_duration", return_value=2.0),
            patch("diarize.run_vad", return_value=speech),
            patch("diarize.extract_embeddings", return_value=(embeddings, subs)),
            patch("diarize.cluster_speakers", return_value=(np.array([0]), None)),
        ):
            result = diarize(Path("test.wav"))

        assert result.audio_path == "test.wav"


# ── Public API imports ───────────────────────────────────────────────────────


class TestImports:
    """Test that the public API is properly exposed."""

    def test_import_diarize_function(self):
        from diarize import diarize

        assert callable(diarize)

    def test_import_result_classes(self):
        from diarize import DiarizeResult, Segment

        assert DiarizeResult is not None
        assert Segment is not None

    def test_version(self):
        from diarize import __version__

        assert __version__ == "0.1.0"

    def test_import_pydantic_models(self):
        from diarize.utils import (
            SpeakerEstimationDetails,
            SpeechSegment,
            SubSegment,
        )

        assert SpeechSegment is not None
        assert SubSegment is not None
        assert SpeakerEstimationDetails is not None


# ── Edge-case fixes (Codex review) ───────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases found during Codex code review."""

    def test_estimate_speakers_empty_embeddings(self):
        """P2: estimate_speakers should handle empty embedding array."""
        from diarize.clustering import estimate_speakers

        embeddings = np.empty((0, 256))
        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        assert k == 1
        assert details.reason == "no_embeddings"

    def test_cluster_spectral_k_greater_than_n(self):
        """P1: cluster_spectral should clamp k to number of embeddings."""
        from diarize.clustering import cluster_spectral

        embeddings = np.random.RandomState(42).randn(2, 256)
        labels = cluster_spectral(embeddings, k=5)
        assert len(labels) == 2
        assert len(set(labels)) <= 2

    def test_cluster_speakers_num_speakers_exceeds_n(self):
        """P1: cluster_speakers should not crash when num_speakers > N."""
        from diarize.clustering import cluster_speakers

        embeddings = np.random.RandomState(42).randn(3, 256)
        labels, details = cluster_speakers(embeddings, num_speakers=10)
        assert len(labels) == 3
        assert details is None

    def test_cluster_speakers_min_speakers_exceeds_n(self):
        """P1: cluster_speakers with min_speakers > N should not crash."""
        from diarize.clustering import cluster_speakers

        embeddings = np.random.RandomState(42).randn(4, 256)
        labels, details = cluster_speakers(embeddings, min_speakers=10, max_speakers=12)
        assert len(labels) == 4

    def test_cluster_spectral_empty_embeddings(self):
        """P1: cluster_spectral should handle empty input."""
        from diarize.clustering import cluster_spectral

        labels = cluster_spectral(np.empty((0, 256)), k=3)
        assert len(labels) == 0

    def test_cluster_spectral_k_equals_1(self):
        """cluster_spectral with k=1 should return all zeros."""
        from diarize.clustering import cluster_spectral

        embeddings = np.random.RandomState(42).randn(10, 256)
        labels = cluster_spectral(embeddings, k=1)
        assert len(labels) == 10
        assert set(labels) == {0}

    def test_extract_embeddings_empty_returns_2d(self):
        """P3: extract_embeddings should return (0, 256) shape when no embeddings."""
        from diarize.embeddings import extract_embeddings
        from diarize.utils import SpeechSegment

        # All segments too short to embed (< 0.4s)
        speech = [SpeechSegment(start=0.0, end=0.2)]
        audio = np.random.randn(3200).astype(np.float32)  # 0.2s at 16kHz

        mock_model = MagicMock()
        mock_rt = MagicMock()
        mock_rt.Speaker.return_value = mock_model

        with (
            patch.dict("sys.modules", {"wespeakerruntime": mock_rt}),
            patch("diarize.embeddings.sf.read", return_value=(audio, 16000)),
        ):
            embeddings, subsegments = extract_embeddings("test.wav", speech)
            assert embeddings.ndim == 2
            assert embeddings.shape == (0, 256)
            assert subsegments == []
