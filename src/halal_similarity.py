from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import CFG


def build_similarity(
    df: pd.DataFrame, feature_cols: list[str] = None, top_n: int = None, cfg=CFG
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = list(cfg.similarity_features)
    if top_n is None:
        top_n = cfg.similarity_top_n

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
