"""Joint K-means scoring for one subject's Pre, Peak, and Post clone rows.

Each row is one unique clone observation in one timepoint. Repeated embedding
vectors must remain repeated rows when they represent different clones or
timepoints. Counts/read abundance never supplies a sample weight here.

The latest paper supplies K=500 and the sum-of-ratios score. Epsilon=1.0,
seed=20260919, n_init=10, unscaled Euclidean K-means++, and four threads are
explicit provisional implementation choices, not confirmed paper settings.
UMAP coordinates must not be supplied. The application checks the expected
480-dimensional AbLang2 representation; this numeric component permits smaller
dimensions for independently checkable synthetic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence
import warnings

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


TIMEPOINTS = ("Pre", "Peak", "Post")


@dataclass(frozen=True)
class KMeansResult:
    """Labels and scores retain the input clone-row order."""

    labels: np.ndarray
    scores: np.ndarray
    clusters: list[dict[str, int | float]]
    metadata: dict[str, object]


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _epsilon(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("epsilon must be a finite positive real number.")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("epsilon must be a finite positive real number.") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError("epsilon must be a finite positive real number.")
    return result


def score_cluster_counts(
    n_pre: int, n_peak: int, n_post: int, *, epsilon: float = 1.0
) -> float:
    """Return peak/(pre+epsilon) + peak/(post+epsilon).

    Counts are nonnegative numbers of unique clone observations. Epsilon=1.0
    is provisional. This is a sum, not a product or a requirement that both
    ratios exceed a threshold. Zero Pre/Post counts are allowed.
    """
    pre = _integer(n_pre, "n_pre", minimum=0)
    peak = _integer(n_peak, "n_peak", minimum=0)
    post = _integer(n_post, "n_post", minimum=0)
    epsilon = _epsilon(epsilon)
    try:
        result = peak / (pre + epsilon) + peak / (post + epsilon)
    except OverflowError:
        raise ValueError("Clone counts exceed the supported numeric range.") from None
    if not math.isfinite(result):
        raise ValueError("The cluster score is not finite; review epsilon and counts.")
    return float(result)


def run_kmeans(
    embeddings: np.ndarray,
    timepoints: Sequence[str],
    *,
    n_clusters: int = 500,
    epsilon: float = 1.0,
    seed: int = 20260919,
    n_init: int = 10,
    threads: int = 4,
) -> KMeansResult:
    """Fit pooled, unscaled embeddings and assign a cluster score to each row.

    All three timepoints must be present. The caller owns single-subject and
    unique-clone construction, as well as the use of original AbLang2 rather
    than UMAP coordinates. Rows are not deduplicated or weighted. Data are
    copied as float64, and the caller's array is left unchanged.

    K is never silently reduced: too few observations, too few distinct
    embedding vectors, convergence warnings, and empty clusters are errors.
    The seed may be zero and must fit NumPy's 32-bit random-state range.
    Labels are arbitrary cluster identifiers, not biological categories.
    """
    n_clusters = _integer(n_clusters, "n_clusters")
    n_init = _integer(n_init, "n_init")
    threads = _integer(threads, "threads")
    seed = _integer(seed, "seed", minimum=0)
    if seed > 2**32 - 1:
        raise ValueError("seed must fit the unsigned 32-bit random-state range.")
    epsilon = _epsilon(epsilon)

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional NumPy array.")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have at least one row and one dimension.")
    if embeddings.dtype.kind not in "iuf":
        raise ValueError("embeddings must contain real numeric values, excluding booleans.")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values.")
    if isinstance(timepoints, (str, bytes)):
        raise ValueError("timepoints must provide one label per embedding row.")
    try:
        points = tuple(timepoints)
    except TypeError:
        raise ValueError("timepoints must provide one label per embedding row.") from None
    if len(points) != len(embeddings):
        raise ValueError("timepoints and embeddings must have the same number of rows.")
    if any(not isinstance(point, str) or point not in TIMEPOINTS for point in points):
        raise ValueError("Timepoint labels must be exactly Pre, Peak, or Post.")
    if set(points) != set(TIMEPOINTS):
        raise ValueError("Pre, Peak, and Post must each contain at least one clone.")
    if n_clusters > len(embeddings):
        raise ValueError("n_clusters exceeds the number of clone observations; choose K explicitly.")

    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(embeddings, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(values).all():
        raise ValueError("embeddings exceed the supported float64 numeric range.")
    n_unique_vectors = len(np.unique(values, axis=0))
    if n_clusters > n_unique_vectors:
        raise ValueError("n_clusters exceeds distinct embedding vectors; choose K explicitly.")

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=seed,
        algorithm="lloyd",
        copy_x=True,
    )
    try:
        with threadpool_limits(limits=threads), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            labels = model.fit_predict(values)
    except ConvergenceWarning:
        raise ValueError(
            "K-means reported a convergence warning; review embeddings and K before continuing."
        ) from None

    if len(np.unique(labels)) != n_clusters:
        raise ValueError("K-means produced empty clusters; review embeddings and K before continuing.")
    if not math.isfinite(float(model.inertia_)):
        raise ValueError("K-means produced non-finite distances; review the embedding numeric range.")

    timepoint_indices = np.fromiter((TIMEPOINTS.index(point) for point in points), dtype=np.int64)
    counts = np.zeros((n_clusters, len(TIMEPOINTS)), dtype=np.int64)
    np.add.at(counts, (labels, timepoint_indices), 1)
    clusters: list[dict[str, int | float]] = []
    cluster_scores = np.empty(n_clusters, dtype=np.float64)
    for cluster_id, (n_pre, n_peak, n_post) in enumerate(counts):
        score = score_cluster_counts(n_pre, n_peak, n_post, epsilon=epsilon)
        cluster_scores[cluster_id] = score
        clusters.append(
            {
                "cluster_id": cluster_id,
                "n_pre": int(n_pre),
                "n_peak": int(n_peak),
                "n_post": int(n_post),
                "score": score,
            }
        )

    return KMeansResult(
        labels=np.array(labels, dtype=np.int64, copy=True),
        scores=cluster_scores[labels],
        clusters=clusters,
        metadata={
            "method": "pooled_kmeans",
            "distance": "euclidean",
            "scaling": "none",
            "sample_weight": "none; one point per unique clone per timepoint",
            "n_clusters": n_clusters,
            "epsilon": epsilon,
            "seed": seed,
            "n_init": n_init,
            "init": "k-means++",
            "algorithm": "lloyd",
            "max_iter": model.max_iter,
            "tol": model.tol,
            "thread_limit": threads,
            "observations": len(values),
            "unique_embedding_vectors": n_unique_vectors,
            "dimensions": values.shape[1],
            "input_dtype": str(embeddings.dtype),
            "fit_dtype": str(values.dtype),
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "unconfirmed_paper_settings": [
                "epsilon", "seed", "n_init", "init", "distance", "scaling", "algorithm",
                "max_iter", "tol", "thread_limit",
            ],
        },
    )
