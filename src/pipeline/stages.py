"""Named stages for the end-to-end workflow."""

PIPELINE_STAGES = (
    "audit_sources",
    "build_feature_tables",
    "build_microzones",
    "fit_trajectory_model",
    "fit_survival_model",
    "aggregate_review_labels",
    "rank_candidate_zones",
)
