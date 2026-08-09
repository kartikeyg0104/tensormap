"""Integration tests for the tuning pipeline.

These tests exercise the full tuning flow against a real (in-memory) database
but with model training mocked out so they complete in seconds.  Real training
integration tests would be marked @pytest.mark.slow and need a configured
dataset.

Marked @pytest.mark.slow for CI — run with ``pytest -m slow``.
"""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.models.tuning_session import TuningSession
from app.models.tuning_session import TuningStatus as TuningSessionStatus

BASE = "/api/v1/model"


def _seed_model(session: Session, name: str = "int_tuning_model") -> ModelBasic:
    """Create a fully-configured model ready to train."""
    project = Project(name="int_proj")
    session.add(project)
    session.commit()
    session.refresh(project)

    data_file = DataFile(file_name="d.csv", file_type="csv", disk_name="d.csv", project_id=project.id)
    session.add(data_file)
    session.commit()
    session.refresh(data_file)

    model = ModelBasic(
        model_name=name,
        file_id=data_file.id,
        project_id=project.id,
        model_type=1,
        target_field="label",
        training_split=80,
        optimizer="adam",
        metric="accuracy",
        epochs=2,
        batch_size=32,
        loss="sparse_categorical_crossentropy",
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


@pytest.fixture(autouse=True)
def _no_background_tuning(mocker):
    """Stop the tuning endpoint from launching real tuning."""
    return mocker.patch("app.routers.tuning.launch_tuning_session")


@pytest.mark.slow
class TestTuningIntegrationFlow:
    """End-to-end tuning flow with mocked training."""

    def test_full_tuning_flow(self, client, db_session, monkeypatch):
        """Create model → POST tuning → verify jobs → GET → apply-best."""
        model = _seed_model(db_session, "int_tuning_model")

        # 1. Start tuning with 4 random trials.
        resp = client.post(
            f"{BASE}/tuning/int_tuning_model",
            json={
                "model_name": "int_tuning_model",
                "strategy": "random",
                "search_space": {
                    "optimizer": ["adam", "sgd"],
                    "batch_size": [16, 32],
                },
                "max_trials": 4,
                "metric": "val_accuracy",
                "direction": "maximize",
            },
        )
        assert resp.status_code == 202
        data = resp.json()["data"]
        tuning_id = data["tuning_id"]
        assert data["n_trials"] == 4
        assert data["estimated_seconds"] > 0

        # 2. Simulate: create 4 training_job rows linked to this session.
        now = datetime.now(UTC)
        best_metric = 0.0
        best_job_id = None
        for i in range(4):
            hp = {"optimizer": "adam" if i % 2 == 0 else "sgd", "batch_size": 16 if i < 2 else 32}
            job = TrainingJob(
                model_id=model.id,
                status=TrainingStatus.COMPLETED,
                hyperparams=hp,
                tuning_session_id=tuning_id,
                started_at=now - timedelta(seconds=60 - i * 10),
                completed_at=now - timedelta(seconds=50 - i * 10),
            )
            db_session.add(job)
            db_session.commit()
            db_session.refresh(job)

            # Add a metric for each job.
            metric_val = 0.7 + i * 0.05
            db_session.add(
                TrainingMetric(
                    job_id=job.id,
                    epoch=1,
                    metric_name="val_accuracy",
                    metric_value=metric_val,
                )
            )
            db_session.commit()

            if metric_val > best_metric:
                best_metric = metric_val
                best_job_id = job.id

        # Update tuning session status.
        ts = db_session.get(TuningSession, tuning_id)
        ts.status = TuningSessionStatus.COMPLETED
        ts.completed_trials = 4
        ts.best_job_id = best_job_id
        ts.completed_at = now
        db_session.add(ts)
        db_session.commit()

        # 3. Verify 4 training_job rows created with correct tuning_session_id.
        jobs = db_session.exec(select(TrainingJob).where(TrainingJob.tuning_session_id == tuning_id)).all()
        assert len(jobs) == 4

        # 4. GET tuning session → verify best_job_id set.
        resp = client.get(f"{BASE}/tuning/{tuning_id}")
        assert resp.status_code == 200
        detail = resp.json()["data"]
        assert detail["status"] == "completed"
        assert detail["best_job_id"] == best_job_id
        assert len(detail["trials"]) == 4
        assert detail["completed_trials"] == 4

        # 5. POST apply-best → verify model config updated.
        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        resp = client.post(f"{BASE}/tuning/{tuning_id}/apply-best")
        assert resp.status_code == 200
        applied = resp.json()["data"]["applied_hyperparams"]
        assert applied["optimizer"] in ["adam", "sgd"]

        # Verify model was updated.
        db_session.refresh(model)
        assert model.optimizer == applied["optimizer"]

    def test_grid_search_creates_correct_trials(self, client, db_session):
        """Grid search with 2×2 space creates 4 trial combinations."""
        _seed_model(db_session, "grid_model")

        resp = client.post(
            f"{BASE}/tuning/grid_model",
            json={
                "model_name": "grid_model",
                "strategy": "grid",
                "search_space": {
                    "optimizer": ["adam", "sgd"],
                    "batch_size": [16, 32],
                },
            },
        )
        assert resp.status_code == 202
        data = resp.json()["data"]
        assert data["n_trials"] == 4

    def test_cancel_and_recheck(self, client, db_session):
        """Cancel a running session, then verify GET shows cancelled."""
        _seed_model(db_session, "cancel_model")

        resp = client.post(
            f"{BASE}/tuning/cancel_model",
            json={
                "model_name": "cancel_model",
                "strategy": "random",
                "search_space": {"optimizer": ["adam", "sgd"]},
                "max_trials": 10,
            },
        )
        tuning_id = resp.json()["data"]["tuning_id"]

        # Set to RUNNING manually (normally done by the background task).
        ts = db_session.get(TuningSession, tuning_id)
        ts.status = TuningSessionStatus.RUNNING
        db_session.add(ts)
        db_session.commit()

        # Cancel it.
        resp = client.delete(f"{BASE}/tuning/{tuning_id}")
        assert resp.status_code == 204

        # Verify status via GET.
        resp = client.get(f"{BASE}/tuning/{tuning_id}")
        assert resp.json()["data"]["status"] == "cancelled"
