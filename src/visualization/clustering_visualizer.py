"""
Clustering visualizations (2D & 3D) with PCA projection.

All functions accept any array-like feature matrix (X) and a 1D array of labels.
Noise points (label == -1) are rendered in light gray.

Logging:
- Uses module-level logger for informative messages.
- Saves figures if `save_path` is provided (directories are created automatically).
- Uses a safe PCA reducer that pads missing dimensions with zeros when data is too small, preventing ValueError.
"""

import os
import logging
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------- #
# Internal helper utilities
# ---------------------------- #

def _to_numpy_2d(X: Iterable) -> np.ndarray:
    """
    Ensure input features are a clean 2D numpy array with finite values only.
    Non-finite rows are filtered out (NaN/Inf), with a warning log.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # keep only finite rows
    finite_mask = np.all(np.isfinite(X), axis=1)
    if not np.all(finite_mask):
        removed = int((~finite_mask).sum())
        logger.warning("Dropped %d rows with NaN/Inf before plotting.", removed)
        X = X[finite_mask]

    return X


def _labels_to_numpy(labels: Iterable, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert labels to a numpy array. If labels length mismatches X rows,
    it trims to the smaller size and logs a warning.
    Returns (labels, valid_mask) where valid_mask indicates rows that remain.
    """
    y = np.asarray(labels)
    if y.ndim != 1:
        y = y.ravel()

    if len(y) != n_rows:
        k = min(len(y), n_rows)
        logger.warning(
            "Labels length (%d) != X rows (%d). Trimming to %d.", len(y), n_rows, k
        )
        y = y[:k]
        valid_mask = np.zeros(n_rows, dtype=bool)
        valid_mask[:k] = True
        return y, valid_mask
    else:
        return y, np.ones(n_rows, dtype=bool)


def _maybe_standardize(X: np.ndarray, standardize: bool) -> np.ndarray:
    """
    Optionally standardize features to zero-mean/unit-variance before PCA.
    Useful when scales differ significantly across dimensions.
    """
    if standardize:
        X = StandardScaler().fit_transform(X)
    return X


def _safe_pca_reduce(X: np.ndarray, target_dims: int, random_state: int = 42) -> np.ndarray:
    """
    Reduce X with PCA to up to `target_dims` components safely.
    If the data does not have enough samples/features to compute the requested
    number of components, it will:
      - fit PCA with the maximum feasible components, and
      - pad remaining dimensions with zeros so that the returned array has
        shape (n_samples, target_dims).

    This avoids ValueError like:
      "n_components=k must be between 0 and min(n_samples, n_features)=m"
    """
    n_samples, n_features = X.shape
    feasible_dims = min(target_dims, n_samples, n_features)
    if feasible_dims <= 0:
        logger.error(
            "Insufficient data for PCA (n_samples=%d, n_features=%d). "
            "Cannot project to %dD.",
            n_samples, n_features, target_dims
        )
        raise ValueError("Not enough data to perform PCA.")

    pca = PCA(n_components=feasible_dims, random_state=random_state)
    reduced = pca.fit_transform(X)

    # Pad with zeros if we couldn't reach the target dimensionality
    if feasible_dims < target_dims:
        pad = np.zeros((n_samples, target_dims - feasible_dims), dtype=reduced.dtype)
        reduced = np.hstack([reduced, pad])
        logger.warning(
            "PCA could only produce %d component(s); padded to %dD for plotting.",
            feasible_dims, target_dims
        )
    return reduced


def _make_colors(unique_labels: np.ndarray):
    """
    Pick a visually distinct colormap given the number of clusters.
    Noise (-1) will be handled outside and always drawn in light gray.
    """
    n = len(unique_labels)
    # remove -1 from palette count if present (noise is fixed color)
    effective_n = n - (1 if -1 in unique_labels else 0)
    # Use tab20 up to 20 distinct colors, then fallback to HSV for more
    if effective_n <= 20:
        cmap = cm.get_cmap("tab20", max(effective_n, 1))
        def color_fn(i):
            return cmap(i)
    else:
        cmap = cm.get_cmap("hsv", effective_n)
        def color_fn(i):
            return cmap(i)
    return color_fn


def _ensure_dir(path: str):
    """Create parent directory for a file path if it does not exist."""
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------- #
# Public plotting functions
# ---------------------------- #

def plot_clusters_2d(
    embeddings,
    labels,
    title: str = "2D Clustering Result",
    save_path: Optional[str] = None,
    standardize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.85,
    point_size: int = 40,
):
    """
    Project features to 2D by PCA and scatter-plot colored by cluster labels.

    Parameters
    ----------
    embeddings : array-like, shape (n_samples, n_features)
        Feature matrix to visualize.
    labels : array-like, shape (n_samples,)
        Cluster labels. Noise points should be labeled -1 (if any).
    title : str
        Figure title.
    save_path : Optional[str]
        If provided, saves the figure (PNG) to this path.
    standardize : bool
        If True, standardize features before PCA.
    figsize : tuple
        Figure size in inches.
    alpha : float
        Point transparency for scatter.
    point_size : int
        Size of scatter points.
    """
    X = _to_numpy_2d(embeddings)
    y, valid_mask = _labels_to_numpy(labels, X.shape[0])
    X = X[valid_mask]

    X = _maybe_standardize(X, standardize)
    reduced = _safe_pca_reduce(X, target_dims=2, random_state=42)

    unique_labels = np.unique(y)
    color_fn = _make_colors(unique_labels)

    plt.figure(figsize=figsize)

    # Plot noise first (so clusters draw on top)
    noise_mask = (y == -1)
    if noise_mask.any():
        plt.scatter(
            reduced[noise_mask, 0],
            reduced[noise_mask, 1],
            label="Noise",
            s=point_size,
            c=["lightgray"],
            edgecolors="none",
            alpha=alpha,
        )

    # Plot each cluster
    cluster_ids = [lab for lab in unique_labels if lab != -1]
    for i, lab in enumerate(cluster_ids):
        mask = (y == lab)
        if mask.any():
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                label=f"Cluster {lab}",
                s=point_size,
                color=[color_fn(i)],
                edgecolors="none",
                alpha=alpha,
            )

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster", loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved 2D clustering plot to %s", save_path)
    plt.show()


def plot_clusters_3d(
    embeddings,
    labels,
    title: str = "3D Clustering Result",
    save_path: Optional[str] = None,
    standardize: bool = False,
    figsize: Tuple[int, int] = (10, 7),
    alpha: float = 0.85,
    point_size: int = 40,
    elev: int = 25,
    azim: int = 135,
):
    """
    Project features to 3D by PCA and scatter-plot colored by cluster labels.

    Parameters
    ----------
    embeddings : array-like, shape (n_samples, n_features)
        Feature matrix to visualize.
    labels : array-like, shape (n_samples,)
        Cluster labels. Noise points should be labeled -1 (if any).
    title : str
        Figure title.
    save_path : Optional[str]
        If provided, saves the figure (PNG) to this path.
    standardize : bool
        If True, standardize features before PCA.
    figsize : tuple
        Figure size in inches.
    alpha : float
        Point transparency for scatter.
    point_size : int
        Size of scatter points.
    elev : int
        Elevation angle for 3D view.
    azim : int
        Azimuth angle for 3D view.
    """
    X = _to_numpy_2d(embeddings)
    y, valid_mask = _labels_to_numpy(labels, X.shape[0])
    X = X[valid_mask]

    X = _maybe_standardize(X, standardize)
    reduced = _safe_pca_reduce(X, target_dims=3, random_state=42)

    unique_labels = np.unique(y)
    color_fn = _make_colors(unique_labels)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot noise first
    noise_mask = (y == -1)
    if noise_mask.any():
        ax.scatter(
            reduced[noise_mask, 0],
            reduced[noise_mask, 1],
            reduced[noise_mask, 2],
            label="Noise",
            s=point_size,
            color="lightgray",
            edgecolors="none",
            alpha=alpha,
        )

    # Plot clusters
    cluster_ids = [lab for lab in unique_labels if lab != -1]
    for i, lab in enumerate(cluster_ids):
        mask = (y == lab)
        if mask.any():
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                reduced[mask, 2],
                label=f"Cluster {lab}",
                s=point_size,
                color=color_fn(i),
                edgecolors="none",
                alpha=alpha,
            )

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.legend(title="Cluster", loc="best")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved 3D clustering plot to %s", save_path)
    plt.show()


def plot_elbow_method(
    scores: Sequence[float],
    ks: Optional[Sequence[int]] = None,
    title: str = "Elbow Method - WCSS vs K",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
):
    """
    Plot the elbow curve for KMeans (WCSS vs K).

    Parameters
    ----------
    scores : sequence of float
        Inertia/Within-Cluster Sum of Squares for each K.
    ks : sequence of int, optional
        The corresponding K values. If None, assumes range(2, 2+len(scores)).
    title : str
        Plot title.
    save_path : Optional[str]
        If provided, saves the figure (PNG) to this path.
    figsize : tuple
        Figure size in inches.
    """
    scores = list(scores)
    if ks is None:
        ks = list(range(2, 2 + len(scores)))

    if len(ks) != len(scores):
        raise ValueError("Length of ks must match length of scores.")

    plt.figure(figsize=figsize)
    plt.plot(ks, scores, marker="o", linestyle="--")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved elbow plot to %s", save_path)
    plt.show()
