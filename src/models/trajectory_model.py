"""Clustering model scaffold for neighborhood regime discovery."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class TrajectoryClusteringModel:
    """Scaffold for unsupervised neighborhood phase discovery."""

    algorithm: str = "kmeans"
    n_clusters: int = 3
    random_state: int = 42
    fitted_: bool = field(default=False, init=False)
    cluster_labels_: list[str] = field(default_factory=list, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    scaler_: StandardScaler | None = field(default=None, init=False)
    model_: KMeans | GaussianMixture | None = field(default=None, init=False)

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

        if self.algorithm == "gmm":
            self.model_ = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
            )
            self.model_.fit(scaled)
        else:
            self.model_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
            self.model_.fit(scaled)

        self.fitted_ = True
        self.cluster_labels_ = [f"cluster_{index}" for index in range(self.n_clusters)]
        return self

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        assert self.scaler_ is not None
        assert self.model_ is not None

        numeric = feature_matrix[self.feature_columns_].copy().fillna(0.0)
        scaled = self.scaler_.transform(numeric)
        if self.algorithm == "gmm":
            raw_labels = self.model_.predict(scaled)
        else:
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
        return labeled.groupby("trajectory_cluster")[numeric_cols].mean(numeric_only=True)
