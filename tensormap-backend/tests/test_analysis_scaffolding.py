"""Tests for Phase 4 analysis scaffolding (Week 8).

These tests verify that the analysis routes exist and return expected
status codes during the scaffolding phase.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus

client = TestClient(app)


@pytest.fixture
def completed_job(db_session: Session):
    """Create a completed training job for testing."""
    # Create model
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    # Create completed job
    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    return job_id


def test_analysis_routes_exist(completed_job):
    """All 3 analysis routes should exist and return 501 (not 404)."""
    job_id = completed_job

    # Confusion matrix
    response = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
    assert response.status_code == 501
    assert "week 9" in response.json()["detail"].lower()

    # Feature importance
    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert response.status_code == 501
    assert "week 9" in response.json()["detail"].lower()

    # Predictions
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert response.status_code == 501
    assert "week 9" in response.json()["detail"].lower()


def test_analysis_requires_completed_job(db_session: Session):
    """Analysis routes should return 400 for non-completed jobs."""
    # Create model
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    # Create running job
    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=1,
        status=TrainingStatus.RUNNING,
        hyperparams={},
        started_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    # All routes should return 400 (job not completed)
    response = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
    assert response.status_code == 400
    assert "completed" in response.json()["detail"].lower()

    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert response.status_code == 400

    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert response.status_code == 400


def test_analysis_returns_404_for_nonexistent_job(db_session: Session):
    """Analysis routes should return 404 for non-existent jobs."""
    nonexistent_job_id = str(uuid4())

    # All routes should return 404
    response = client.get(f"/api/v1/model/analysis/{nonexistent_job_id}/confusion-matrix")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

    response = client.get(f"/api/v1/model/analysis/{nonexistent_job_id}/feature-importance")
    assert response.status_code == 404

    response = client.get(f"/api/v1/model/analysis/{nonexistent_job_id}/predictions")
    assert response.status_code == 404


def test_predictions_accepts_pagination_params(completed_job):
    """Predictions route should accept offset and limit query parameters."""
    job_id = completed_job

    # Test with default params
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert response.status_code == 501  # Not implemented yet, but route exists

    # Test with custom params
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions?offset=10&limit=50")
    assert response.status_code == 501

    # Test with invalid params (limit too high)
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions?limit=200")
    # Route declares limit: int = Query(25, ge=1, le=100), so FastAPI rejects with 422
    assert response.status_code == 422


def test_interpretability_service_not_implemented():
    """InterpretabilityService methods should raise NotImplementedError."""
    from app.services.interpretability import InterpretabilityService

    service = InterpretabilityService()

    with pytest.raises(NotImplementedError, match="Week 9"):
        service.compute_confusion_matrix("job_id", None)

    with pytest.raises(NotImplementedError, match="Week 9"):
        service.compute_feature_importance("job_id", None)

    with pytest.raises(NotImplementedError, match="Week 9"):
        service.get_predictions("job_id", 0, 25, None)


def test_analysis_cache_operations(db_session: Session):
    """AnalysisCache should be able to get and set cached results."""
    from app.services.interpretability import AnalysisCache

    # Create a completed job
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    cache = AnalysisCache()

    # Initially, no cache
    result = cache.get_cached(job_id, "confusion_matrix", db_session)
    assert result is None

    # Set cache
    test_data = {"matrix": [[10, 2], [1, 15]], "accuracy": 0.89}
    cache.set_cached(job_id, "confusion_matrix", test_data, db_session)

    # Retrieve cache
    result = cache.get_cached(job_id, "confusion_matrix", db_session)
    assert result is not None
    assert result["matrix"] == [[10, 2], [1, 15]]
    assert result["accuracy"] == 0.89

    # Different analysis type should not be cached
    result = cache.get_cached(job_id, "feature_importance", db_session)
    assert result is None
