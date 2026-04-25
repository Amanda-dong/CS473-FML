.PHONY: install etl train-survival train-scoring train pipeline api ui run test all

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv pip install -r requirements.txt

# ── Data pipeline ────────────────────────────────────────────────────────────

etl:
	uv run python -m run_full_pipeline --etl-only

etl-small:
	uv run python -m run_full_pipeline --etl-only --limit 5000

# ── Model training ───────────────────────────────────────────────────────────

train-survival:
	uv run python -m src.models.train_survival

train-scoring:
	uv run python -m src.models.train_scoring

train:
	uv run python -m run_full_pipeline --train-only

# ── Full pipeline (ETL + train) ───────────────────────────────────────────────

pipeline:
	uv run python -m run_full_pipeline

# ── Servers ──────────────────────────────────────────────────────────────────

api:
	uv run python -m uvicorn src.api.main:app --reload --port 8000

ui:
	uv run python -m streamlit run frontend/app.py

run:
	uv run python -m uvicorn src.api.main:app --port 8000 &
	uv run python -m streamlit run frontend/app.py

# ── Quality ──────────────────────────────────────────────────────────────────

test:
	uv run python -m pytest -v

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

# ── Combined ─────────────────────────────────────────────────────────────────

all: install pipeline
