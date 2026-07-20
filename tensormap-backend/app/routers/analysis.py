"""Analysis router for post-training interpretability.

Provides endpoints for:
- Confusion matrices and classification reports
- Feature importance analysis
- Paginated prediction results

Week 8 scaffolding: All routes return 501 Not Implemented.
Full implementation in Week 9.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.database import get_db
from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model/analysis", tags=["analysis"])


def _verify_job_completed(job_id: str, session: Session) -> TrainingJob:
    """Verify that a training job exists and is completed.

    Args:
        job_id: Training job ID
        session: Database session

    Returns:
        TrainingJob instance

    Raises:
        HTTPException: 404 if job not found, 400 if job not completed
    """
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis only available for completed jobs. Current status: {job.status}",
        )

    return job


@router.get("/{job_id}/confusion-matrix")
async def get_confusion_matrix(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Returns confusion matrix and classification report.

    Args:
        job_id: Training job ID

    Returns:
        {
            "confusion_matrix": [[int, ...], ...],
            "classification_report": {
                "class_0": {"precision": float, "recall": float, "f1-score": float},
                ...
            },
            "accuracy": float,
            "cached": bool
        }

    Status:
        - 200: Analysis available (from cache or computed)
        - 202: Analysis in progress (check back later)
        - 400: Job not completed
        - 404: Job not found
        - 501: Not implemented (Week 8 scaffolding)
    """
    # Verify job exists and is completed
    _verify_job_completed(job_id, db)

    # Week 8: Return 501 Not Implemented
    logger.info(f"Confusion matrix requested for job {job_id} - not implemented yet")
    raise HTTPException(
        status_code=501,
        detail="Confusion matrix analysis will be implemented in Week 9",
    )

    # Week 9 implementation will be:
    # service = InterpretabilityService()
    #
    # # Check cache first
    # cached = service.cache.get_cached(job_id, "confusion_matrix", db)
    # if cached:
    #     return JSONResponse(status_code=200, content={**cached, "cached": True})
    #
    # # Compute and cache
    # result = service.compute_confusion_matrix(job_id, db)
    # return JSONResponse(status_code=200, content={**result, "cached": False})


@router.get("/{job_id}/feature-importance")
async def get_feature_importance(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Returns permutation feature importance.

    Computes feature importance by permuting each feature and measuring
    the impact on model performance.

    Args:
        job_id: Training job ID

    Returns:
        {
            "feature_importance": [
                {"feature": str, "importance": float, "std": float},
                ...
            ],
            "sorted_by": "importance",
            "cached": bool
        }

    Status:
        - 200: Analysis available (from cache or computed)
        - 202: Analysis in progress (check back later)
        - 400: Job not completed
        - 404: Job not found
        - 501: Not implemented (Week 8 scaffolding)
    """
    # Verify job exists and is completed
    _verify_job_completed(job_id, db)

    # Week 8: Return 501 Not Implemented
    logger.info(f"Feature importance requested for job {job_id} - not implemented yet")
    raise HTTPException(
        status_code=501,
        detail="Feature importance analysis will be implemented in Week 9",
    )

    # Week 9 implementation will be:
    # service = InterpretabilityService()
    #
    # # Check cache first
    # cached = service.cache.get_cached(job_id, "feature_importance", db)
    # if cached:
    #     return JSONResponse(status_code=200, content={**cached, "cached": True})
    #
    # # Return 202 if computation not started yet, start async computation
    # # For Week 9, we'll compute synchronously first, then optimize with async
    # result = service.compute_feature_importance(job_id, db)
    # return JSONResponse(status_code=200, content={**result, "cached": False})


@router.get("/{job_id}/predictions")
async def get_predictions(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Returns paginated predictions for test data.

    Args:
        job_id: Training job ID
        offset: Pagination offset (default: 0)
        limit: Number of predictions to return (default: 25, max: 100)

    Returns:
        {
            "predictions": [
                {
                    "index": int,
                    "true_label": int,
                    "predicted_label": int,
                    "confidence": float,
                    "correct": bool
                },
                ...
            ],
            "total": int,
            "offset": int,
            "limit": int
        }

    Status:
        - 200: Predictions available
        - 400: Job not completed
        - 404: Job not found
        - 501: Not implemented (Week 8 scaffolding)
    """
    # Verify job exists and is completed
    _verify_job_completed(job_id, db)

    # Week 8: Return 501 Not Implemented
    logger.info(f"Predictions requested for job {job_id} (offset={offset}, limit={limit}) - not implemented yet")
    raise HTTPException(
        status_code=501,
        detail="Predictions endpoint will be implemented in Week 9",
    )

    # Week 9 implementation will be:
    # service = InterpretabilityService()
    # result = service.get_predictions(job_id, offset, limit, db)
    # return JSONResponse(status_code=200, content=result)
