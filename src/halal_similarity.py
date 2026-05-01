from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity(
    df: pd.DataFrame, feature_cols: list[str], top_n: int = 3
) -> pd.DataFrame:
    work = df.dropna(subset=feature_cols).copy()
    means = work[feature_cols].mean()
    stds = work[feature_cols].std().replace(0, 1.0)
    X = ((work[feature_cols] - means) / stds).to_numpy(dtype=float)

    sim = cosine_similarity(X)
    nta_ids = work["nta_id"].tolist()
    mapping: dict[str, str] = {}

    for i, nta in enumerate(nta_ids):
        order = sim[i].argsort()[::-1]
        neighbors = [nta_ids[j] for j in order if j != i][:top_n]
        mapping[nta] = ",".join(neighbors)

    out = df.copy()
    out["similar_ntas"] = out["nta_id"].map(mapping)
    return out
