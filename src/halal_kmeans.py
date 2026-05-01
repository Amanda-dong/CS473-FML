from __future__ import annotations

import numpy as np
import pandas as pd


class HalalKMeans:
    def __init__(self, k: int = 4, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def _assign(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        init_idx = rng.choice(len(X), size=self.k, replace=False)
        self.centroids_ = X[init_idx].copy()

        for i in range(1, self.max_iter + 1):
            labels = self._assign(X)
            new_centroids = self.centroids_.copy()
            for cluster_id in range(self.k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = X[mask].mean(axis=0)
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.labels_ = labels
            self.n_iter_ = i
            if shift < self.tol:
                break

        dists = np.linalg.norm(X - self.centroids_[self.labels_], axis=1) ** 2
        self.inertia_ = float(np.sum(dists))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(X)


def run_kmeans(df: pd.DataFrame, feature_cols: list[str], k: int = 4):
    work = df.dropna(subset=feature_cols).copy()
    means = work[feature_cols].mean()
    stds = work[feature_cols].std().replace(0, 1.0)
    X = ((work[feature_cols] - means) / stds).to_numpy(dtype=float)

    km = HalalKMeans(k=k, random_state=42)
    km.fit(X)
    work["cluster_id"] = km.labels_

    centroid_raw = (
        work.groupby("cluster_id", as_index=False)[feature_cols]
        .mean()
    )
    centroid_raw["market_type"] = ""

    remaining = set(centroid_raw["cluster_id"].astype(int).tolist())

    rank_col = "gap_score" if "gap_score" in centroid_raw.columns else "demand_score"
    high_opp_candidates = centroid_raw[centroid_raw["demand_score"] > 0.5]
    if high_opp_candidates.empty:
        high_opp_row = centroid_raw.sort_values([rank_col, "demand_score"], ascending=False).iloc[0]
    else:
        high_opp_row = high_opp_candidates.sort_values([rank_col, "demand_score"], ascending=False).iloc[0]
    high_opp_id = int(high_opp_row["cluster_id"])
    centroid_raw.loc[centroid_raw["cluster_id"] == high_opp_id, "market_type"] = "High Opportunity"
    remaining.discard(high_opp_id)

    if remaining:
        established_row = (
            centroid_raw[centroid_raw["cluster_id"].isin(remaining)]
            .sort_values(["halal_supply_rate", "demand_score"], ascending=False)
            .iloc[0]
        )
        established_id = int(established_row["cluster_id"])
        centroid_raw.loc[centroid_raw["cluster_id"] == established_id, "market_type"] = "Established Hub"
        remaining.discard(established_id)

    if remaining:
        ordered_remaining = (
            centroid_raw[centroid_raw["cluster_id"].isin(remaining)]
            .sort_values("demand_score", ascending=False)
        )
        names = ["Growing Market", "Low Demand"]
        for row, name in zip(ordered_remaining.itertuples(index=False), names):
            centroid_raw.loc[centroid_raw["cluster_id"] == int(row.cluster_id), "market_type"] = name
            remaining.discard(int(row.cluster_id))

    for cid in remaining:
        centroid_raw.loc[centroid_raw["cluster_id"] == cid, "market_type"] = "Other"

    cluster_to_name = dict(zip(centroid_raw["cluster_id"].astype(int), centroid_raw["market_type"]))
    work["market_type"] = work["cluster_id"].map(cluster_to_name)

    counts = work["market_type"].value_counts()
    print(f"Inertia: {km.inertia_:.4f}")
    print(f"Iterations to convergence: {km.n_iter_}")
    print("Cluster size counts:")
    print(counts.to_string())

    return work, km
