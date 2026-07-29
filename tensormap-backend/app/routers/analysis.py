"""Analysis router for post-training interpretability.

Provides endpoints for:
- Confusion matrices and classification reports
- Feature importance analysis
- Paginated prediction results

Routes to different analysis types based on model_basic.model_type:
- classification (1): confusion matrix, classification report, feature importance
- regression (2): residual plot, error histogram, feature importance (MSE-based)
- image_classification (3): confusion matrix, classification report, per-class grid
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.interpretability import InterpretabilityService
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model/analysis", tags=["analysis"])

# Singleton service instance — stateless, safe to share across requests.
_service = InterpretabilityService()


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


def get_analysis_type(job_id: str, session: Session) -> str:
    """Returns 'classification', 'regression', or 'image_classification'.

    Resolves the problem type by following training_job → model_basic → model_type.

    Args:
        job_id: Training job ID
        session: Database session

    Returns:
        String identifying the analysis type

    Raises:
        HTTPException: 404 if job or model not found
    """
    from app.models.ml import ModelBasic

    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    model = session.get(ModelBasic, job.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    type_map = {
        ProblemType.CLASSIFICATION: "classification",
        ProblemType.REGRESSION: "regression",
        ProblemType.IMAGE_CLASSIFICATION: "image_classification",
    }

    return type_map.get(model.model_type, "classification")


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
            "overall_accuracy": float,
            "class_names": [str, ...],
            "n_samples": int,
            "analysis_type": "classification",
            "cached": bool
        }

    Status:
        - 200: Analysis available (from cache or computed)
        - 400: Job not completed or regression model
        - 404: Job not found
    """
    # Verify job exists and is completed
    _verify_job_completed(job_id, db)

    # Task-type routing: confusion matrix not available for regression
    analysis_type = get_analysis_type(job_id, db)
    if analysis_type == "regression":
        raise HTTPException(
            status_code=400,
            detail="Confusion matrix not available for regression models. Use /residual-plot instead.",
        )

    # Check if result was already cached
    cached = _service.cache.get_cached(job_id, "confusion_matrix", db)
    if cached:
        return JSONResponse(status_code=200, content={**cached, "cached": True})

    # Compute and cache
    try:
        result = await _service.compute_confusion_matrix_async(job_id, db)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.exception("Confusion matrix computation failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Analysis computation failed: {exc}") from exc
    return JSONResponse(status_code=200, content={**result, "cached": False})


@router.get("/{job_id}/feature-importance")
async def get_feature_importance(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Returns permutation feature importance.

    Uses 202 polling pattern: first request starts background computation
    and returns 202. Subsequent requests return 202 while computing, and
    200 with data once complete.

    Args:
        job_id: Training job ID

    Returns:
        202: {"status": "computing"} — computation in progress
        200: {
            "features": [str, ...],
            "importances_mean": [float, ...],
            "importances_std": [float, ...],
            "n_samples_used": int,
            "n_repeats": int,
            "analysis_type": "feature_importance",
            "cached": bool
        }

    Status:
        - 200: Analysis available (from cache or computed)
        - 202: Analysis in progress (check back later)
        - 400: Job not completed
        - 404: Job not found
    """
    # Verify job exists and is completed
    _verify_job_completed(job_id, db)

    result = await _service.compute_feature_importance_async(job_id, db)
    if result is None:
        return JSONResponse(status_code=202, content={"status": "computing"})
    return JSONResponse(status_code=200, content={**result, "cached": True})


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
        - 501: Not implemented (Week 10)
    """
    # Verify job exists and is completed
    _verify_job_completed(job_id, db)

    # Week 10: Return 501 Not Implemented
    logger.info(f"Predictions requested for job {job_id} (offset={offset}, limit={limit}) - not implemented yet")
    raise HTTPException(
        status_code=501,
        detail="Predictions endpoint will be implemented in Week 10",
    )
