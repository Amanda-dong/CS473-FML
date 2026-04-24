# API Contract

Base URL (local): `http://127.0.0.1:8000`

Content type: `application/json`

Error format (FastAPI default):

```json
{
  "detail": []
}
```

## GET `/health`

`200`:

```json
{
  "status": "ok"
}
```

## GET `/datasets`

`200` (example):

```json
[
  {
    "name": "permits",
    "owner": "data",
    "spatial_unit": "nta",
    "time_grain": "year",
    "earliest_year": null,
    "status": "planned",
    "notes": ""
  }
]
```

## POST `/predict/cmf`

Request:

```json
{
  "concept_subtype": "ramen",
  "price_tier": "mid",
  "borough": "Any",
  "risk_tolerance": "balanced",
  "zone_type": "",
  "limit": 5
}
```

Field notes:

- `concept_subtype`: string, default `healthy_indian`
- `price_tier`: `budget | mid | premium`
- `borough`: `Any | Manhattan | Brooklyn | Queens | Bronx | Staten Island` or `null`
- `risk_tolerance`: `conservative | balanced | aggressive`
- `zone_type`: optional string filter, empty means all
- `limit`: int in `[1, 20]`

`200` (shape):

```json
{
  "query": {
    "concept_subtype": "ramen",
    "zone_type": "all",
    "borough": "Any"
  },
  "recommendations": [
    {
      "zone_id": "bk-tandon",
      "zone_name": "NYU Tandon / MetroTech 10-minute campus walkshed",
      "concept_subtype": "ramen",
      "opportunity_score": 0.8123,
      "confidence_bucket": "high",
      "healthy_gap_summary": "string",
      "positives": ["string"],
      "risks": ["string"],
      "freshness_note": "string",
      "feature_contributions": {
        "healthy_review_share": 0.1182
      },
      "survival_risk": 0.25,
      "model_version": "xgboost_v1",
      "scoring_path": "learned",
      "label_quality": 1.0
    }
  ]
}
```

Notes:

- `opportunity_score` and `survival_risk` are in `[0, 1]`
- `scoring_path` is one of `learned`, `heuristic`, `heuristic_fallback`

Errors:

- `422`: request validation error
- `500`: unexpected server error

## POST `/predict/trajectory`

Request:

```json
{
  "concept_subtype": "ramen",
  "price_tier": "mid",
  "risk_tolerance": "balanced",
  "zone_type": "campus_walkshed"
}
```

`200`:

```json
{
  "concept_subtype": "ramen",
  "zone_type": "campus_walkshed",
  "trajectory_cluster": "emerging"
}
```

`trajectory_cluster`:

- `emerging`
- `gentrifying`
- `stable`
- `declining`

Errors:

- `422`: request validation error
- `500`: unexpected server error
