"""Tests for model deletion cascade behavior (Week 8)."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.deep_learning import delete_model_service


@pytest.fixture
def mock_model_path():
    """Mock the model path validation to avoid file system checks."""
    with patch("app.services.deep_learning._validate_model_path") as mock:
        mock.return_value = "/tmp/model.json"
        yield mock


def test_delete_model_cancels_running_jobs(db_session: Session, tmp_path, mock_model_path):
    """Deleting a model should delete all associated training jobs via CASCADE."""
    # Create model
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    # Create jobs in different states
    running_job = TrainingJob(
        id=str(uuid4()),
        model_id=1,
        status=TrainingStatus.RUNNING,
        hyperparams={},
        started_at=datetime.now(UTC),
    )
    pending_job = TrainingJob(
        id=str(uuid4()),
        model_id=1,
        status=TrainingStatus.PENDING,
        hyperparams={},
    )
    completed_job = TrainingJob(
        id=str(uuid4()),
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC),
    )

    db_session.add(running_job)
    db_session.add(pending_job)
    db_session.add(completed_job)
    db_session.commit()

    running_job_id = running_job.id
    pending_job_id = pending_job.id
    completed_job_id = completed_job.id

    # Mock export deletion
    with patch("app.services.model_export.delete_model_exports", return_value=0):
        # Delete model
        response, status_code = delete_model_service(db_session, model_id=1)

    assert status_code == 200
    assert response["success"] is True

    # Verify all jobs were deleted via CASCADE
    assert db_session.get(TrainingJob, running_job_id) is None
    assert db_session.get(TrainingJob, pending_job_id) is None
    assert db_session.get(TrainingJob, completed_job_id) is None


def test_delete_model_deletes_associated_jobs(db_session: Session, tmp_path, mock_model_path):
    """Training job records should be deleted via CASCADE when model is deleted."""
    # Create model
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    # Create a running job
    job = TrainingJob(
        id=str(uuid4()),
        model_id=1,
        status=TrainingStatus.RUNNING,
        hyperparams={},
    )
    db_session.add(job)
    db_session.commit()

    job_id = job.id

    # Mock export deletion
    with patch("app.services.model_export.delete_model_exports", return_value=0):
        # Delete model
        response, status_code = delete_model_service(db_session, model_id=1)

    assert status_code == 200

    # Job record should be deleted (CASCADE)
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    deleted_job = db_session.exec(stmt).first()
    assert deleted_job is None


def test_delete_model_deletes_exports(db_session: Session, tmp_path, mock_model_path):
    """Deleting a model should delete all associated export directories."""
    # Create model
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    # Create jobs
    job1_id = str(uuid4())
    job2_id = str(uuid4())

    job1 = TrainingJob(id=job1_id, model_id=1, status=TrainingStatus.COMPLETED, hyperparams={})
    job2 = TrainingJob(id=job2_id, model_id=1, status=TrainingStatus.COMPLETED, hyperparams={})

    db_session.add(job1)
    db_session.add(job2)
    db_session.commit()

    # Create export directories
    exports_base = tmp_path / "exports"
    job1_dir = exports_base / job1_id
    job2_dir = exports_base / job2_id

    job1_dir.mkdir(parents=True)
    job2_dir.mkdir(parents=True)
    (job1_dir / "model.keras").write_text("job1")
    (job2_dir / "model.keras").write_text("job2")

    # Delete model
    with patch("app.services.model_export.EXPORTS_BASE", exports_base):
        response, status_code = delete_model_service(db_session, model_id=1)

    assert status_code == 200

    # Export directories should be deleted
    assert not job1_dir.exists()
    assert not job2_dir.exists()


def test_delete_model_handles_completed_jobs(db_session: Session, mock_model_path):
    """Deleting a model with completed jobs should succeed and delete all jobs via CASCADE."""
    # Create model with only completed jobs
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    completed_job = TrainingJob(
        id=str(uuid4()),
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC),
    )
    db_session.add(completed_job)
    db_session.commit()

    completed_job_id = completed_job.id

    # Mock export deletion
    with patch("app.services.model_export.delete_model_exports", return_value=1):
        # Delete model
        response, status_code = delete_model_service(db_session, model_id=1)

    assert status_code == 200
    assert response["success"] is True

    # Completed job should be deleted via CASCADE
    assert db_session.get(TrainingJob, completed_job_id) is None


def test_delete_model_continues_on_export_failure(db_session: Session, mock_model_path):
    """Model deletion should continue even if export cleanup fails."""
    # Create model
    model = ModelBasic(id=1, model_name="test_model", graph_ir={})
    db_session.add(model)
    db_session.commit()

    # Mock export deletion to raise an exception
    with patch("app.services.model_export.delete_model_exports", side_effect=Exception("Export cleanup failed")):
        # Delete model - should succeed despite export failure
        response, status_code = delete_model_service(db_session, model_id=1)

    # Model deletion should succeed
    assert status_code == 200
    assert response["success"] is True

    # Model should be deleted from database
    deleted_model = db_session.get(ModelBasic, 1)
    assert deleted_model is None


def test_delete_model_cascades_training_metrics(db_session: Session, mock_model_path):
    """Deleting a model should cascade-delete training_metric rows via training_job."""
    from app.models.training_metric import TrainingMetric

    # Create model → training job → training metric
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

    # Add metric rows for this job
    metric1 = TrainingMetric(job_id=job_id, epoch=1, metric_name="loss", metric_value=0.5)
    metric2 = TrainingMetric(job_id=job_id, epoch=1, metric_name="accuracy", metric_value=0.85)
    metric3 = TrainingMetric(job_id=job_id, epoch=2, metric_name="loss", metric_value=0.3)
    db_session.add_all([metric1, metric2, metric3])
    db_session.commit()

    metric1_id = metric1.id
    metric2_id = metric2.id
    metric3_id = metric3.id

    # Verify metrics exist before deletion
    assert db_session.get(TrainingMetric, metric1_id) is not None
    assert db_session.get(TrainingMetric, metric2_id) is not None
    assert db_session.get(TrainingMetric, metric3_id) is not None

    # Delete model — should cascade through training_job to training_metric
    with patch("app.services.model_export.delete_model_exports", return_value=0):
        response, status_code = delete_model_service(db_session, model_id=1)

    assert status_code == 200
    assert response["success"] is True

    # Verify all metric rows were cascade-deleted
    assert db_session.get(TrainingMetric, metric1_id) is None
    assert db_session.get(TrainingMetric, metric2_id) is None
    assert db_session.get(TrainingMetric, metric3_id) is None

    # Verify the training job itself is also gone
    assert db_session.get(TrainingJob, job_id) is None
