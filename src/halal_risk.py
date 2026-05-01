from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from src.config import CFG


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "output"
INSPECTIONS = PROCESSED / "inspections.parquet"
PHASE1 = OUT_DIR / "phase1_cluster_assignments.csv"

GMM_FEATURES = [
    "critical_rate",
    "grade_a_rate",
    "inspection_frequency",
    "demand_score",
    "halal_supply_rate",
]


def _load_inspection_agg() -> pd.DataFrame:
    inspections = pd.read_parquet(INSPECTIONS)
    inspections["inspection_date"] = pd.to_datetime(
        inspections["inspection_date"], errors="coerce"
    )
    inspections["year"] = inspections["inspection_date"].dt.year
    inspections = inspections[
        (inspections["year"] >= 2020) & (inspections["year"] <= 2025)
    ].copy()
    inspections = inspections.dropna(subset=["nta_id"]).copy()

    critical = inspections["critical_flag"].fillna("").astype(str).str.upper()
    grade = inspections["grade"].fillna("").astype(str).str.upper()
    inspections["is_critical"] = critical.eq("CRITICAL").astype(int)
    inspections["is_grade_a"] = grade.eq("A").astype(int)

    agg = inspections.groupby("nta_id", as_index=False).agg(
        critical_rate=("is_critical", "mean"),
        grade_a_rate=("is_grade_a", "mean"),
        inspection_count=("restaurant_id", "count"),
        restaurant_count=("restaurant_id", pd.Series.nunique),
    )
    agg["inspection_frequency"] = (
        agg["inspection_count"] / agg["restaurant_count"].replace(0, pd.NA)
    ).fillna(0.0)
    agg["risk_confidence"] = agg["inspection_count"].apply(
        lambda n: "High confidence" if n >= 10 else "Low confidence"
    )
    return agg


def _zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        std = out[c].std()
        if pd.isna(std) or std == 0:
            std = 1.0
        out[c] = (out[c] - out[c].mean()) / std
    return out


def build_viability(cfg=CFG) -> pd.DataFrame:
    agg = _load_inspection_agg()

    tmp = _zscore(agg[['grade_a_rate', 'critical_rate']], ['grade_a_rate', 'critical_rate'])
    raw = tmp['grade_a_rate'] - tmp['critical_rate']
    min_r = raw.min()
    max_r = raw.max()
    agg["viability_score"] = (raw - min_r) / (max_r - min_r) if max_r != min_r else 0.5

    agg["risk_bucket"] = agg["viability_score"].apply(
        lambda v: "Low" if v >= cfg.viability_low_threshold else ("Medium" if v >= cfg.viability_medium_threshold else "High")
    )

    return agg[
        [
            "nta_id",
            "critical_rate",
            "grade_a_rate",
            "inspection_frequency",
            "viability_score",
            "risk_bucket",
        ]
    ].copy()


def build_gmm_risk(cfg=CFG):
    agg = _load_inspection_agg()

    phase1 = pd.read_csv(PHASE1)[["nta_id", "demand_score", "halal_supply_rate"]].copy()
    merged = (
        agg.merge(phase1, on="nta_id", how="inner").dropna(subset=GMM_FEATURES).copy()
    )

    z = _zscore(merged[GMM_FEATURES], GMM_FEATURES)

    bic_rows = []
    for n in [2, 3, 4]:
        g = GaussianMixture(n_components=n, covariance_type="full", random_state=cfg.gmm_random_state)
        g.fit(z.to_numpy())
        bic_rows.append({"n_components": n, "bic": g.bic(z.to_numpy())})

    gmm = GaussianMixture(n_components=cfg.gmm_n_components, covariance_type="full", random_state=cfg.gmm_random_state)
    gmm.fit(z.to_numpy())
    probs = gmm.predict_proba(z.to_numpy())
    labels = gmm.predict(z.to_numpy())

    merged["component"] = labels
    means_table = (
        merged.groupby("component", as_index=False)[GMM_FEATURES]
        .mean()
        .rename(columns={c: f"{c}_mean" for c in GMM_FEATURES})
    )
    means_table["nta_count"] = merged["component"].value_counts().sort_index().to_list()

    ranked_components = (
        means_table.sort_values("critical_rate_mean", ascending=False)["component"]
        .astype(int)
        .tolist()
    )
    risk_name_map = {
        ranked_components[0]: "High Risk",
        ranked_components[1]: "Medium-High Risk",
        ranked_components[2]: "Medium-Low Risk",
        ranked_components[3]: "Low Risk",
    }
    high_risk_components = ranked_components[:2]
    merged["high_risk_prob"] = probs[:, high_risk_components].sum(axis=1)
    merged["risk_component"] = merged["component"].map(risk_name_map)

    def bucket(p: float) -> str:
        if p > cfg.risk_high_threshold:
            return "High"
        if p > cfg.risk_medium_threshold:
            return "Medium"
        return "Low"

    merged["risk_bucket"] = merged["high_risk_prob"].apply(bucket)
    sil = silhouette_score(z.to_numpy(), labels)

    means_table["risk_component"] = means_table["component"].map(risk_name_map)
    means_table = means_table.sort_values(
        "critical_rate_mean", ascending=False
    ).reset_index(drop=True)

    component_inspection_table = merged.groupby("component", as_index=False).agg(
        nta_count=("nta_id", "count"),
        total_inspections=("inspection_count", "sum"),
        mean_inspections=("inspection_count", "mean"),
        low_confidence_ntas=("inspection_count", lambda s: int((s < 10).sum())),
    )
    component_inspection_table["risk_component"] = component_inspection_table[
        "component"
    ].map(risk_name_map)
    component_inspection_table = component_inspection_table.sort_values(
        "total_inspections", ascending=False
    ).reset_index(drop=True)

    low_cov_mask = component_inspection_table["risk_component"].eq(
        "Medium-Low Risk"
    ) & (component_inspection_table["mean_inspections"] < 10)
    component_inspection_table["coverage_note"] = ""
    component_inspection_table.loc[low_cov_mask, "coverage_note"] = (
        "low inspection coverage — risk assessment unreliable"
    )

    risk_df = merged[
        ["nta_id", "high_risk_prob", "risk_bucket", "risk_component", "risk_confidence"]
    ].copy()
    diagnostics = {
        "cluster_means": means_table,
        "component_inspections": component_inspection_table,
        "silhouette": sil,
        "bic_table": pd.DataFrame(bic_rows),
        "high_risk_component": high_risk_components,
    }
    return risk_df, diagnostics
