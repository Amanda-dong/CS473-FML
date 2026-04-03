"""Dataset metadata endpoints."""

from fastapi import APIRouter

from src.data.audit import build_default_audit_rows

router = APIRouter(tags=["datasets"])


@router.get("/datasets")
async def list_datasets() -> list[dict[str, str | int | None]]:
    """Expose the dataset audit inventory for frontend and QA work."""

    return [row.model_dump() for row in build_default_audit_rows()]
