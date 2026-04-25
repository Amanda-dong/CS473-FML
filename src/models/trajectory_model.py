"""Clustering model for neighborhood regime discovery with proper validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryClusteringModel:
    """Unsupervised neighborhood phase discovery with cluster validation.

    Supports k-means and GMM. When ``n_clusters`` is None, automatically
    selects k via silhouette score (k-means) or BIC (GMM).
    """

    algorithm: str = "kmeans"
    n_clusters: int | None = 3
    random_state: int = 42
    fitted_: bool = field(default=False, init=False)
    cluster_labels_: list[str] = field(default_factory=list, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    scaler_: StandardScaler | None = field(default=None, init=False)
    model_: KMeans | GaussianMixture | None = field(default=None, init=False)
    diagnostics_: dict = field(default_factory=dict, init=False)

    def _select_numeric_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        numeric = feature_matrix.select_dtypes(include=["number"]).copy()
        return numeric.fillna(0.0)

    def fit(self, feature_matrix: pd.DataFrame) -> "TrajectoryClusteringModel":
        numeric = self._select_numeric_features(feature_matrix)
        if numeric.empty:
            raise ValueError("feature_matrix must contain at least one numeric column.")

        self.feature_columns_ = list(numeric.columns)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(numeric)

        # Auto-select k if not specified
        k = self.n_clusters
        if k is None:
            k = self._auto_select_k(scaled)
            logger.info(
                "Auto-selected k=%d via %s",
                k,
                "BIC" if self.algorithm == "gmm" else "silhouette",
            )

        if self.algorithm == "gmm":
            self.model_ = GaussianMixture(
                n_components=k,
                random_state=self.random_state,
            )
            self.model_.fit(scaled)
        else:
            self.model_ = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init="auto",
            )
            self.model_.fit(scaled)

        # Store diagnostics
        labels = (
            self.model_.predict(scaled)
            if self.algorithm == "gmm"
            else self.model_.labels_
        )
        n_unique = len(set(labels))
        if 1 < n_unique < len(scaled):
            self.diagnostics_["silhouette"] = float(silhouette_score(scaled, labels))
        if self.algorithm == "gmm":
            self.diagnostics_["bic"] = float(self.model_.bic(scaled))
            self.diagnostics_["aic"] = float(self.model_.aic(scaled))
        elif hasattr(self.model_, "inertia_"):
            self.diagnostics_["inertia"] = float(self.model_.inertia_)

        self.diagnostics_["n_clusters"] = k
        self.diagnostics_["n_samples"] = len(scaled)

        self.fitted_ = True
        self.cluster_labels_ = [f"cluster_{index}" for index in range(k)]
        return self

    def _auto_select_k(self, scaled: np.ndarray, k_range: range | None = None) -> int:
        """Select optimal k using silhouette (k-means) or BIC (GMM)."""
        if k_range is None:
            max_k = min(8, len(scaled) - 1)
            k_range = range(2, max_k + 1)

        if self.algorithm == "gmm":
            best_k, best_bic = 2, float("inf")
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=self.random_state)
                gmm.fit(scaled)
                bic = gmm.bic(scaled)
                if bic < best_bic:
                    best_bic, best_k = bic, k
            return best_k
        else:
            best_k, best_sil = 2, -1.0
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
                labels = km.fit_predict(scaled)
                if len(set(labels)) < 2:
                    continue
                sil = silhouette_score(scaled, labels)
                if sil > best_sil:
                    best_sil, best_k = sil, k
            return best_k

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        assert self.scaler_ is not None
        assert self.model_ is not None

        numeric = feature_matrix[self.feature_columns_].copy().fillna(0.0)
        scaled = self.scaler_.transform(numeric)
        raw_labels = self.model_.predict(scaled)
        labels = [self.cluster_labels_[label] for label in raw_labels]
        return pd.Series(labels, name="trajectory_cluster")

    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Convenience helper for exploratory notebooks."""
        return self.fit(feature_matrix).predict(feature_matrix)

    def describe_clusters(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Return feature means by predicted cluster for inspection."""
        labeled = feature_matrix.copy()
        labeled["trajectory_cluster"] = self.predict(feature_matrix)
        numeric_cols = labeled.select_dtypes(include=["number"]).columns.tolist()
        return labeled.groupby("trajectory_cluster")[numeric_cols].mean(
            numeric_only=True
        )

    def cluster_stability(self, scaled_data: np.ndarray, n_runs: int = 10) -> float:
        """Measure clustering stability via Adjusted Rand Index across runs.

        Fits the model ``n_runs`` times with different seeds and computes
        pairwise ARI. High mean ARI (>0.8) indicates stable clusters.
        """
        all_labels = []
        for i in range(n_runs):
            seed = self.random_state + i
            k = self.diagnostics_.get("n_clusters", self.n_clusters or 3)
            if self.algorithm == "gmm":
                m = GaussianMixture(n_components=k, random_state=seed)
                m.fit(scaled_data)
                all_labels.append(m.predict(scaled_data))
            else:
                m = KMeans(n_clusters=k, random_state=seed, n_init="auto")
                all_labels.append(m.fit_predict(scaled_data))

        ari_scores = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))
        mean_ari = float(np.mean(ari_scores)) if ari_scores else 0.0
        self.diagnostics_["stability_ari"] = mean_ari
        return mean_ari

    def sweep_k(
        self, feature_matrix: pd.DataFrame, k_range: range | None = None
    ) -> pd.DataFrame:
        """Evaluate multiple k values and return diagnostics for each.

        Returns DataFrame with columns: k, silhouette, inertia (or bic/aic for GMM).
        """
        numeric = self._select_numeric_features(feature_matrix)
        scaled = (
            self.scaler_.fit_transform(numeric)
            if self.scaler_
            else StandardScaler().fit_transform(numeric)
        )

        if k_range is None:
            max_k = min(8, len(scaled) - 1)
            k_range = range(2, max_k + 1)

        records = []
        for k in k_range:
            row: dict = {"k": k}
            if self.algorithm == "gmm":
                m = GaussianMixture(n_components=k, random_state=self.random_state)
                m.fit(scaled)
                labels = m.predict(scaled)
                row["bic"] = float(m.bic(scaled))
                row["aic"] = float(m.aic(scaled))
            else:
                m = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
                labels = m.fit_predict(scaled)
                row["inertia"] = float(m.inertia_)

            if len(set(labels)) > 1:
                row["silhouette"] = float(silhouette_score(scaled, labels))
            else:
                row["silhouette"] = -1.0
            records.append(row)

        return pd.DataFrame(records)
