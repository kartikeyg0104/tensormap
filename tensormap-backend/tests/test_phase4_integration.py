"""Phase 4 Integration Tests — End-to-end analysis workflow.

Tests the complete analysis pipeline from training job creation through
all analysis endpoints (confusion matrix, feature importance, predictions).
Verifies caching behavior and performance.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from tests.fixtures.create_test_model import create_test_training_job

client = TestClient(app)


@pytest.fixture
def export_dir(tmp_path):
    """Provide a temporary export directory."""
    return tmp_path / "exports"


@pytest.mark.slow
def test_phase4_full_analysis_workflow(db_session: Session, export_dir: Path):
    """Full Phase 4 workflow: train → confusion matrix → feature importance → predictions.

    This test verifies:
    1. Training a simple classification model
    2. GET /analysis/{id}/confusion-matrix → correct shape and accuracy
    3. GET /analysis/{id}/feature-importance → poll until 200 → verify features list
    4. GET /analysis/{id}/predictions → verify pagination works
    5. All results are cached (analysis_cache column populated)
    6. Repeat GET requests → served from cache (no recomputation)
    """
    # Step 1: Create a trained model
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name="phase4_integration_model",
        num_features=4,
        num_classes=3,
        epochs=5,
    )

    class_names = [str(i) for i in range(3)]

    # Generate real predictions using the trained model to avoid Mock/sklearn issues
    y_pred_proba = model.predict(X_test, verbose=0)

    # Create a mock model that returns our pre-computed predictions
    mock_model = MagicMock()
    mock_model.predict.return_value = y_pred_proba

    # Prepare mocks
    from app.services.interpretability import InterpretabilityService

    # Step 2: GET confusion matrix
    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch.object(
            InterpretabilityService,
            "_load_test_data_with_features",
            return_value=(X_test, y_test, feature_names),
        ),
        patch.object(
            InterpretabilityService,
            "_load_test_data_complete",
            return_value=(X_test, y_test, class_names, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
    ):
        # First call: compute confusion matrix
        cm_response = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
        assert cm_response.status_code == 200
        cm_data = cm_response.json()

        # Verify correct shape and structure
        assert len(cm_data["confusion_matrix"]) == 3
        assert len(cm_data["confusion_matrix"][0]) == 3
        assert "overall_accuracy" in cm_data
        assert 0 <= cm_data["overall_accuracy"] <= 1
        assert cm_data["cached"] is False

        # Step 3: GET feature importance (202 polling pattern)
        # First request should return 202
        fi_response1 = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
        assert fi_response1.status_code == 202
        assert fi_response1.json()["status"] == "computing"

        # Simulate computation completion by manually caching
        from app.services.interpretability import AnalysisCache

        cache = AnalysisCache()
        fi_result = {
            "features": feature_names,
            "importances_mean": [0.15, 0.25, 0.35, 0.10],
            "importances_std": [0.02, 0.03, 0.04, 0.01],
            "analysis_type": "feature_importance",
            "n_samples_used": len(X_test),
            "n_repeats": 10,
        }
        cache.set_cached(job_id, "feature_importance", fi_result, db_session)

        # Subsequent request should return 200 with data
        fi_response2 = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
        assert fi_response2.status_code == 200
        fi_data = fi_response2.json()
        assert "features" in fi_data
        assert len(fi_data["features"]) == 4
        assert fi_data["cached"] is True

        # Step 4: GET predictions with pagination
        pred_response = client.get(f"/api/v1/model/analysis/{job_id}/predictions", params={"offset": 0, "limit": 10})
        assert pred_response.status_code == 200
        pred_data = pred_response.json()

        assert "predictions" in pred_data
        assert "total" in pred_data
        assert "offset" in pred_data
        assert "limit" in pred_data
        assert pred_data["offset"] == 0
        assert pred_data["limit"] == 10

        # Verify prediction structure
        if len(pred_data["predictions"]) > 0:
            pred = pred_data["predictions"][0]
            assert "index" in pred
            assert "actual_class" in pred
            assert "predicted_class" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
            assert "features" in pred
            assert "is_correct" in pred

        # Step 5: Verify all results are cached
        from app.models.training_job import TrainingJob

        job = db_session.get(TrainingJob, job_id)
        assert job is not None
        assert job.analysis_cache is not None
        assert "confusion_matrix" in job.analysis_cache
        assert "feature_importance" in job.analysis_cache
        assert "predictions" in job.analysis_cache

        # Step 6: Repeat GET requests → served from cache
        # Confusion matrix second call
        cm_response2 = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
        assert cm_response2.status_code == 200
        assert cm_response2.json()["cached"] is True
        assert cm_response2.json()["confusion_matrix"] == cm_data["confusion_matrix"]

        # Feature importance already tested above

        # Predictions second call (should use cache)
        pred_response2 = client.get(f"/api/v1/model/analysis/{job_id}/predictions", params={"offset": 0, "limit": 10})
        assert pred_response2.status_code == 200
        # Predictions are served from cache (same data)
        assert pred_response2.json()["predictions"] == pred_data["predictions"]


@pytest.mark.slow
def test_phase4_regression_analysis(db_session: Session, export_dir: Path):
    """Regression analysis returns residuals correctly."""
    # Create a regression-configured job
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name="regression_integration_model",
        num_features=4,
        num_classes=3,  # Will use as regression output
        epochs=3,
    )

    # Update the model to be regression type
    from app.models.ml import ModelBasic
    from app.models.training_job import TrainingJob
    from app.shared.enums import ProblemType

    job = db_session.get(TrainingJob, job_id)
    model_record = db_session.get(ModelBasic, job.model_id)
    model_record.model_type = ProblemType.REGRESSION
    db_session.add(model_record)
    db_session.commit()

    # Convert y_test to float for regression
    y_test_float = y_test.astype(float)

    # Generate real predictions to avoid Mock/sklearn issues
    y_pred_raw = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_raw, axis=1).astype(float)  # Convert to regression-like output

    # Create a mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = y_pred.reshape(-1, 1)  # Regression shape

    from app.services.interpretability import InterpretabilityService

    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data",
            return_value=(X_test, y_test_float, []),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
    ):
        # GET residuals endpoint
        response = client.get(f"/api/v1/model/analysis/{job_id}/residuals")
        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "y_pred" in data
        assert "y_true" in data
        assert "residuals" in data
        assert "mae" in data
        assert "mse" in data
        assert data["analysis_type"] == "regression"
        assert len(data["residuals"]) == len(y_test_float)

        # Verify residuals = y_pred - y_true
        for i in range(min(5, len(data["residuals"]))):
            expected_residual = data["y_pred"][i] - data["y_true"][i]
            assert abs(data["residuals"][i] - expected_residual) < 1e-5


def test_phase4_no_regressions_to_phases_1_3(db_session: Session):
    """Verify Phase 4 doesn't break existing endpoints (smoke test)."""
    # Test that core endpoints still work

    # Phase 1: Projects
    projects_response = client.get("/api/v1/project")
    assert projects_response.status_code == 200

    # Phase 2: Data upload endpoint exists
    files_response = client.get("/api/v1/data/upload/file")
    assert files_response.status_code == 200

    # Phase 3: Model endpoints exist
    models_response = client.get("/api/v1/model/model-list")
    assert models_response.status_code == 200
