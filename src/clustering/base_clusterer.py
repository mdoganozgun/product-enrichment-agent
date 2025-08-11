"""
Abstract base class for clustering implementations (feature-agnostic).

This package treats input vectors generically as 'features' rather than 'embeddings'.
"""

from abc import ABC, abstractmethod
import pandas as pd
from . import logger


class BaseClusterer(ABC):
    """
    Base interface for clustering classes.

    Expected DataFrame schema (minimum):
      - 'CustomerID' (or a relevant identifier)
      - 'features' (array-like numeric vector per row)

    Concrete classes must implement:
      - load_features(...)
      - cluster(...)
      - evaluate()
      - save_results(...)
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.labels_ = None

    @abstractmethod
    def load_features(self, path: str = "../data/customer_features.pkl") -> None:
        """Load feature vectors into `self.df`. Implement auto-mapping from legacy 'embedding' to 'features'."""
        raise NotImplementedError

    @abstractmethod
    def cluster(self, **kwargs) -> None:
        """Run the clustering algorithm; set `self.labels_` and a 'cluster' column in `self.df`."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> None:
        """Compute and log clustering quality metrics (e.g., silhouette)."""
        raise NotImplementedError

    @abstractmethod
    def save_results(self, path: str = "../data/customer_segments.csv") -> None:
        """Persist clustering outputs to disk."""
        raise NotImplementedError

    # Common validation helpers
    def _ensure_features_loaded(self) -> None:
        if self.df is None:
            logger.error("DataFrame is not loaded. Call `load_features()` first.")
            raise ValueError("DataFrame is not loaded. Call `load_features()` first.")
        if "features" not in self.df.columns:
            logger.error("Column 'features' not found. Ensure loader maps vectors to 'features'.")
            raise ValueError("Column 'features' not found. Ensure loader maps vectors to 'features'.")