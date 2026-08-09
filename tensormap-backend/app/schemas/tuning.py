"""Request/response schemas for the tuning API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TuningStartRequest(BaseModel):
    """Request body for starting a hyperparameter tuning session.

    ``model_name`` and ``search_space`` are required; the rest have sensible
    defaults matching the spec (random search, 20 trials, maximise val_accuracy).
    """

    model_name: str = Field(min_length=1)
    strategy: str = Field(default="random", pattern="^(grid|random)$")
    search_space: dict = Field(
        ...,
        description=(
            "Parameter definitions. Lists are discrete choices, dicts with "
            "'type':'log_uniform'/'uniform' define continuous ranges."
        ),
    )
    max_trials: int = Field(default=20, gt=0)
    metric: str = Field(default="val_accuracy", min_length=1)
    direction: str = Field(default="maximize", pattern="^(maximize|minimize)$")
    early_stop_threshold: float | None = None


class TuningSessionResponse(BaseModel):
    """Full detail for a tuning session."""

    tuning_id: str
    model_id: int
    strategy: str
    search_space: dict
    max_trials: int
    metric: str
    direction: str
    early_stop_threshold: float | None = None
    best_job_id: str | None = None
    status: str
    total_trials: int
    completed_trials: int
    early_stopped: bool
    created_at: datetime | None = None
    completed_at: datetime | None = None
    trials: list[TuningTrialSummary] = []
    estimated_seconds: int | None = None


class TuningTrialSummary(BaseModel):
    """Summary of a single trial within a tuning session."""

    job_id: str
    status: str
    hyperparams: dict | None = None
    metric_value: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TuningStartResponse(BaseModel):
    """202 response from starting a tuning session."""

    tuning_id: str
    status: str
    estimated_seconds: int
    n_trials: int


class TuningProgressEvent(BaseModel):
    """Socket.IO progress event payload."""

    type: str = "tuning_progress"
    trial: int
    total: int
    hyperparams: dict
    metric: float | None = None
    best_metric: float | None = None
    best_job_id: str | None = None


# Rebuild forward refs for TuningSessionResponse.trials
TuningSessionResponse.model_rebuild()
