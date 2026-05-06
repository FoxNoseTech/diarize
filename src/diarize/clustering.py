"""Speaker clustering: GMM BIC for speaker count estimation + Spectral Clustering.

The module provides two main capabilities:

1. **Speaker count estimation** — :func:`estimate_speakers` uses Gaussian
   Mixture Models with the Bayesian Information Criterion (BIC) to
   automatically determine how many speakers are present.
2. **Spectral clustering** — :func:`cluster_spectral` groups embedding
   vectors into *k* clusters using cosine-similarity affinity.

High-level convenience wrappers :func:`cluster_auto` and
:func:`cluster_speakers` combine both steps.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from .utils import SpeakerEstimationDetails

logger = logging.getLogger(__name__)

_SILHOUETTE_K_BONUS = 0.04

__all__ = [
    "estimate_speakers",
    "cluster_spectral",
    "cluster_auto",
    "cluster_speakers",
]


# ── Speaker count estimation ─────────────────────────────────────────────────


def estimate_speakers(
    embeddings: np.ndarray,
    min_k: int = 1,
    max_k: int = 20,
) -> tuple[int, SpeakerEstimationDetails]:
    """Estimate the number of speakers using GMM BIC.

    Algorithm:

    1. L2-normalise embeddings.
    2. PCA projection to 8 dimensions (optimal for GMM with full covariance).
    3. For each *k* from *min_k* to *max_k*, fit
       ``GaussianMixture(k, covariance_type="full")``.
    4. Select *k* with the minimum BIC.

    Args:
        embeddings: Speaker embeddings of shape ``(N, D)``.
        min_k: Minimum number of speakers to consider.
        max_k: Maximum number of speakers to consider.

    Returns:
        A ``(best_k, details)`` tuple where *best_k* is the estimated
        speaker count and *details* is a
        :class:`~diarize.utils.SpeakerEstimationDetails` instance with
        diagnostic information.

    Example::

        k, details = estimate_speakers(embeddings, min_k=1, max_k=10)
        print(f"Estimated {k} speakers (PCA dim={details.pca_dim})")
    """
    n = embeddings.shape[0]

    if n == 0:
        return max(1, min_k), SpeakerEstimationDetails(
            method="gmm_bic",
            best_k=max(1, min_k),
            reason="no_embeddings",
        )

    emb = normalize(embeddings, norm="l2")

    if n < 4:
        return max(1, min_k), SpeakerEstimationDetails(
            method="gmm_bic",
            best_k=max(1, min_k),
            reason="too_few_samples",
        )

    # ── Single-speaker pre-check (cosine similarity) ─────────────────────
    # If the 10th percentile of pairwise cosine similarities is high,
    # all embeddings likely belong to a single speaker.  This catches
    # cases where BIC overfits by splitting one speaker's irregular
    # embedding distribution into multiple Gaussians.
    _single_speaker_sim_p10: float = 0.16
    sim_matrix = cosine_similarity(emb)  # emb is already L2-normalised
    mask = ~np.eye(n, dtype=bool)
    sim_p10 = float(np.percentile(sim_matrix[mask], 10))

    if sim_p10 >= _single_speaker_sim_p10 and min_k <= 1:
        logger.info(
            "Cosine similarity p10=%.3f >= %.2f — single speaker detected",
            sim_p10,
            _single_speaker_sim_p10,
        )
        return 1, SpeakerEstimationDetails(
            method="gmm_bic",
            best_k=1,
            reason="cosine_similarity_single_speaker",
            cosine_sim_p10=round(sim_p10, 4),
        )

    # ── Parameters ────────────────────────────────────────────────────────
    n_pca: int = 8  # PCA dimensions (optimal for GMM full cov)
    gmm_n_init: int = 5  # number of GMM initialisations
    gmm_max_iter: int = 300  # max EM iterations

    # ── PCA projection ────────────────────────────────────────────────────
    actual_pca = min(n_pca, n - 1, emb.shape[1])
    emb_pca = PCA(n_components=actual_pca, random_state=42).fit_transform(emb)

    # ── Sweep k and compute BIC ───────────────────────────────────────────
    # Upper bound: at most n // 2 components (need ≥2 samples per cluster
    # on average), but never more than max_k.
    k_upper = max(min_k + 1, min(max_k + 1, n // 2 + 1))
    k_range = range(min_k, k_upper)
    k_to_bic: dict[int, float] = {}

    for k in k_range:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                n_init=gmm_n_init,
                max_iter=gmm_max_iter,
            )
            gmm.fit(emb_pca)
            bic = gmm.bic(emb_pca)
            k_to_bic[k] = bic

            delta_str = ""
            if k > min_k and (k - 1) in k_to_bic:
                delta = bic - k_to_bic[k - 1]
                delta_str = f"  delta={delta:+.0f}"
            logger.debug("k=%d: BIC=%.0f%s", k, bic, delta_str)
        except Exception:
            logger.debug("GMM failed for k=%d, skipping", k)
            continue

    if not k_to_bic:
        return min_k, SpeakerEstimationDetails(
            method="gmm_bic",
            best_k=min_k,
            reason="gmm_failed",
        )

    # ── Select optimal k ──────────────────────────────────────────────────
    best_k = min(k_to_bic, key=k_to_bic.get)  # type: ignore[arg-type]

    details = SpeakerEstimationDetails(
        method="gmm_bic",
        best_k=best_k,
        pca_dim=actual_pca,
        k_bics={k: round(b, 1) for k, b in sorted(k_to_bic.items())},
    )

    logger.info("GMM BIC (PCA=%d) -> k=%d", actual_pca, best_k)
    return best_k, details


# ── Spectral Clustering ──────────────────────────────────────────────────────


def _refine_labels_spherical(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 8,
) -> np.ndarray:
    """Refine cluster labels with spherical centroid reassignment."""
    n = len(embeddings)
    if n == 0:
        return labels

    unique_labels = sorted({int(label) for label in labels})
    k = len(unique_labels)
    if k <= 1:
        return np.zeros(n, dtype=int)

    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    refined = np.array([label_map[int(label)] for label in labels], dtype=int)
    emb = normalize(embeddings, norm="l2")

    for _ in range(max_iter):
        centroids = np.zeros((k, emb.shape[1]), dtype=float)
        valid = np.zeros(k, dtype=bool)
        for label in range(k):
            members = emb[refined == label]
            if len(members) == 0:  # pragma: no cover - defensive against future label changes.
                continue
            centroid = members.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm == 0:
                continue
            centroids[label] = centroid / norm
            valid[label] = True

        if not np.all(valid):
            break

        scores = emb @ centroids.T
        updated = np.argmax(scores, axis=1)
        if len(set(updated)) < k:
            break
        if np.array_equal(updated, refined):
            break
        refined = updated

    return refined


def cluster_spectral(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Cluster embeddings into *k* speakers using Spectral Clustering.

    Uses cosine similarity as the affinity metric, rescaled to [0, 1].
    The spectral assignment is then refined with a few spherical
    centroid-reassignment iterations, which reduces noisy window labels
    while preserving the selected number of speakers.

    Args:
        embeddings: Speaker embeddings of shape ``(N, D)``.
        k: Number of clusters (speakers).

    Returns:
        Integer label array of shape ``(N,)``.

    Example::

        labels = cluster_spectral(embeddings, k=3)
        print(set(labels))  # {0, 1, 2}
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)

    # Clamp k to the number of available embeddings
    k = min(k, n)

    if k == 1:
        return np.zeros(n, dtype=int)

    affinity = (cosine_similarity(embeddings) + 1) / 2
    np.fill_diagonal(affinity, 1.0)
    affinity = np.maximum(affinity, 0)

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_init=10,
    )
    labels: np.ndarray = sc.fit_predict(affinity)
    labels = _refine_labels_spherical(embeddings, labels)
    logger.debug("Spectral clustering: %d clusters", k)
    return labels


# ── High-level wrappers ──────────────────────────────────────────────────────


def _silhouette_candidate_counts(
    k: int,
    n_embeddings: int,
    min_speakers: int,
    max_speakers: int,
) -> list[int]:
    """Return valid speaker counts for silhouette refinement."""
    if k < 2 or n_embeddings < 4:
        return []

    lower = max(2, min_speakers, k - 2)
    upper = min(max_speakers, n_embeddings - 1, k + 3)
    if upper < lower:
        return []
    return list(range(lower, upper + 1))


def _speaker_count_score(silhouette: float, k: int) -> float:
    """Score speaker-count candidates from silhouette plus a small k prior.

    Raw silhouette tends to prefer fewer, broader clusters on noisy
    speaker embeddings. The logarithmic k bonus counteracts that bias
    without overwhelming the separation score.
    """
    return silhouette + _SILHOUETTE_K_BONUS * np.log(max(k, 1))


def cluster_auto(
    embeddings: np.ndarray,
    min_speakers: int = 1,
    max_speakers: int = 20,
) -> tuple[np.ndarray, SpeakerEstimationDetails]:
    """Automatically determine speaker count and cluster embeddings.

    Combines :func:`estimate_speakers` and :func:`cluster_spectral`
    in a single call.

    Args:
        embeddings: Speaker embeddings of shape ``(N, D)``.
        min_speakers: Minimum number of speakers.
        max_speakers: Maximum number of speakers.

    Returns:
        A ``(labels, details)`` tuple where *labels* is an integer array
        of shape ``(N,)`` and *details* is
        :class:`~diarize.utils.SpeakerEstimationDetails`.
    """
    k, details = estimate_speakers(embeddings, min_speakers, max_speakers)
    n = len(embeddings)

    # Silhouette refinement: use BIC as an anchor, then score a small
    # neighbourhood around it. This catches both undercounts and overcounts.
    if k >= 2:
        candidates = _silhouette_candidate_counts(k, n, min_speakers, max_speakers)
        if len(candidates) > 1:
            distance = np.maximum(1 - (cosine_similarity(embeddings) + 1) / 2, 0)
            best_k, best_labels, best_score = k, None, -1.0
            for c in candidates:
                labels_c = cluster_spectral(embeddings, c)
                sil = silhouette_score(distance, labels_c, metric="precomputed")
                score = _speaker_count_score(sil, c)
                logger.debug(
                    "Silhouette refinement: k=%d  sil=%.4f  score=%.4f",
                    c,
                    sil,
                    score,
                )
                if score > best_score:
                    best_k, best_labels, best_score = c, labels_c, score
            if best_k != k:
                logger.info(
                    "Silhouette refinement: BIC k=%d -> k=%d (score=%.4f)",
                    k,
                    best_k,
                    best_score,
                )
                details.best_k = best_k
            return best_labels, details  # type: ignore[return-value]

    logger.info("Estimated %d speakers (auto — GMM BIC)", k)
    return cluster_spectral(embeddings, k), details


def cluster_speakers(
    embeddings: np.ndarray,
    min_speakers: int = 1,
    max_speakers: int = 20,
    num_speakers: int | None = None,
) -> tuple[np.ndarray, SpeakerEstimationDetails | None]:
    """Cluster speaker embeddings into groups.

    If *num_speakers* is provided, uses that exact number.  Otherwise
    automatically estimates the number of speakers via GMM BIC.

    Args:
        embeddings: Speaker embeddings of shape ``(N, D)``.
        min_speakers: Minimum number of speakers for auto-detection.
        max_speakers: Maximum number of speakers for auto-detection.
        num_speakers: If set, skip auto-detection and use this exact number.

    Returns:
        A ``(labels, details)`` tuple.  *details* is ``None`` when
        *num_speakers* is explicitly provided (no estimation performed).

    Example::

        labels, details = cluster_speakers(embeddings, num_speakers=3)
        # or
        labels, details = cluster_speakers(embeddings, min_speakers=2, max_speakers=10)
    """
    # ── Input validation ─────────────────────────────────────────────────
    if min_speakers < 1:
        raise ValueError(f"min_speakers must be >= 1, got {min_speakers}")
    if max_speakers < min_speakers:
        raise ValueError(f"max_speakers ({max_speakers}) must be >= min_speakers ({min_speakers})")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError(f"num_speakers must be >= 1, got {num_speakers}")

    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int), None

    if num_speakers is not None:
        logger.info("Clustering with fixed num_speakers=%d", num_speakers)
        return cluster_spectral(embeddings, num_speakers), None

    return cluster_auto(embeddings, min_speakers, max_speakers)
