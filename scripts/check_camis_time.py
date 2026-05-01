from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    df = pd.read_csv(RAW / "restaurant_hygiene.csv")
    df["INSPECTION DATE"] = pd.to_datetime(df["INSPECTION DATE"], errors="coerce")

    print("Date range:", df["INSPECTION DATE"].min(), "to", df["INSPECTION DATE"].max())
    print(
        "Years available:",
        sorted(df["INSPECTION DATE"].dt.year.dropna().unique().tolist()),
    )

    halal_cuisines = {
        "Halal",
        "Middle Eastern",
        "Pakistani",
        "Bangladeshi",
        "Afghan",
        "Egyptian",
        "Turkish",
        "Moroccan",
        "Lebanese",
        "Persian/Iranian",
    }
    halal_df = df[df["CUISINE DESCRIPTION"].isin(halal_cuisines)]
    print(f"Total halal restaurant records: {len(halal_df)}")
    print(f"Unique halal CAMIS: {halal_df['CAMIS'].nunique()}")
    print("Halal records per year:")
    print(halal_df["INSPECTION DATE"].dt.year.value_counts().sort_index())

    print("\n--- inspections.parquet 2022 vs 2024 validation ---")
    insp = pd.read_parquet(PROCESSED / "inspections.parquet")
    insp["inspection_date"] = pd.to_datetime(insp["inspection_date"], errors="coerce")
    insp["year"] = insp["inspection_date"].dt.year
    insp = insp[insp["year"].isin([2022, 2024])].dropna(subset=["nta_id"]).copy()
    insp["is_grade_a"] = (
        insp["grade"].fillna("").astype(str).str.upper().eq("A").astype(int)
    )

    agg = insp.groupby(["nta_id", "year"], as_index=False).agg(
        grade_a_rate=("is_grade_a", "mean"), n_inspections=("restaurant_id", "count")
    )

    g2022 = agg[agg["year"] == 2022].rename(
        columns={
            "grade_a_rate": "grade_a_rate_2022",
            "n_inspections": "n_inspections_2022",
        }
    )[["nta_id", "grade_a_rate_2022", "n_inspections_2022"]]
    g2024 = agg[agg["year"] == 2024].rename(
        columns={
            "grade_a_rate": "grade_a_rate_2024",
            "n_inspections": "n_inspections_2024",
        }
    )[["nta_id", "grade_a_rate_2024", "n_inspections_2024"]]
    both = g2022.merge(g2024, on="nta_id", how="inner")
    both_valid = both[
        (both["n_inspections_2022"] >= 5) & (both["n_inspections_2024"] >= 5)
    ].copy()

    print("Total NTAs with valid 2022 data:", g2022["nta_id"].nunique())
    print("Total NTAs with valid 2024 data:", g2024["nta_id"].nunique())
    print("Total NTAs with both years and n >= 5:", both_valid["nta_id"].nunique())
    print(
        "Mean grade_a_rate 2022 vs 2024:",
        round(both_valid["grade_a_rate_2022"].mean(), 4),
        "vs",
        round(both_valid["grade_a_rate_2024"].mean(), 4),
    )

    print("\n--- Yelp/Gemini 2022 vs 2023 validation ---")
    reviews = pd.read_csv(RAW / "yelp_reviews_with_zones.csv")
    gemini = pd.read_csv(RAW / "gemini_labels_full.csv")

    join_key = next(
        (
            c
            for c in ["review_id", "restaurant_id", "business_id"]
            if c in reviews.columns and c in gemini.columns
        ),
        None,
    )
    label_col = next(
        (
            c
            for c in [
                "halal_label",
                "label",
                "gemini_label",
                "category",
                "halal_relevance",
            ]
            if c in gemini.columns
        ),
        None,
    )

    if join_key is None:
        raise ValueError("No join key found for Yelp/Gemini validation.")
    if label_col is None:
        raise ValueError("No halal label column found for Yelp/Gemini validation.")

    joined = reviews.merge(
        gemini[[join_key, label_col]].drop_duplicates(subset=[join_key]),
        on=join_key,
        how="left",
    )
    joined["year"] = pd.to_datetime(joined["review_date"], errors="coerce").dt.year
    joined["is_halal"] = (
        joined[label_col]
        .fillna("")
        .astype(str)
        .str.contains("halal", case=False, regex=False)
        .astype(int)
    )

    agg = (
        joined.dropna(subset=["nta", "year"])
        .groupby(["nta", "year"], as_index=False)
        .agg(total_reviews=("review_id", "count"), halal_count=("is_halal", "sum"))
    )
    agg["halal_related_share"] = agg["halal_count"] / agg["total_reviews"]

    y2022 = agg[(agg["year"] == 2022) & (agg["total_reviews"] >= 5)].rename(
        columns={
            "total_reviews": "total_reviews_2022",
            "halal_related_share": "halal_related_share_2022",
        }
    )[["nta", "total_reviews_2022", "halal_related_share_2022"]]
    y2023 = agg[(agg["year"] == 2023) & (agg["total_reviews"] >= 5)].rename(
        columns={
            "total_reviews": "total_reviews_2023",
            "halal_related_share": "halal_related_share_2023",
        }
    )[["nta", "total_reviews_2023", "halal_related_share_2023"]]
    both_yelp = y2022.merge(y2023, on="nta", how="inner")

    print("NTAs with valid 2022 data (n >= 5):", y2022["nta"].nunique())
    print("NTAs with valid 2023 data (n >= 5):", y2023["nta"].nunique())
    print("NTAs with both years (n >= 5 each):", both_yelp["nta"].nunique())
    print(
        "Mean halal_related_share 2022 vs 2023:",
        round(both_yelp["halal_related_share_2022"].mean(), 4),
        "vs",
        round(both_yelp["halal_related_share_2023"].mean(), 4),
    )

    print("\n--- New halal restaurants per NTA/year validation ---")
    halal_new = halal_df.dropna(subset=["CAMIS", "NTA", "INSPECTION DATE"]).copy()
    halal_new = halal_new[
        halal_new["INSPECTION DATE"].dt.year.between(2010, 2025)
    ].copy()
    halal_new["year"] = halal_new["INSPECTION DATE"].dt.year

    first_seen = (
        halal_new.groupby("CAMIS", as_index=False)["INSPECTION DATE"]
        .min()
        .rename(columns={"INSPECTION DATE": "first_seen_date"})
    )
    first_seen["first_year"] = first_seen["first_seen_date"].dt.year

    camis_nta = halal_new.sort_values("INSPECTION DATE").drop_duplicates(
        subset=["CAMIS"]
    )[["CAMIS", "NTA"]]
    new_halal = camis_nta.merge(
        first_seen[["CAMIS", "first_year"]], on="CAMIS", how="inner"
    )
    new_halal_count = (
        new_halal.groupby(["NTA", "first_year"], as_index=False)["CAMIS"]
        .nunique()
        .rename(columns={"CAMIS": "new_halal_count"})
    )

    yearly_totals = (
        new_halal_count.groupby("first_year", as_index=False)["new_halal_count"]
        .sum()
        .sort_values("first_year")
    )
    print("Total new halal restaurants per year across all NTAs:")
    print(yearly_totals.to_string(index=False))

    needed_years = {2022, 2023, 2024}
    nta_year_sets = new_halal_count.groupby("NTA")["first_year"].apply(set)
    ntas_all_three = sorted(
        [nta for nta, years in nta_year_sets.items() if needed_years.issubset(years)]
    )
    print("\nNTAs that have new_halal_count data in 2022, 2023, and 2024:")
    print(ntas_all_three)
    print("Count of those NTAs:", len(ntas_all_three))
    if ntas_all_three:
        three_year_subset = new_halal_count[
            new_halal_count["NTA"].isin(ntas_all_three)
            & new_halal_count["first_year"].isin([2022, 2023, 2024])
        ].copy()
        mean_by_year = (
            three_year_subset.groupby("first_year", as_index=False)["new_halal_count"]
            .mean()
            .sort_values("first_year")
        )
        print("Mean new_halal_count per year for those NTAs:")
        print(mean_by_year.to_string(index=False))
    else:
        print("Mean new_halal_count per year for those NTAs: none")


if __name__ == "__main__":
    main()
