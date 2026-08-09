"""Hyperparameter tuning session model.

A tuning session orchestrates multiple training-job trials with different
hyperparameter combinations.  Each trial creates a ``TrainingJob`` row so
metrics are independently queryable.  The session tracks overall progress,
the best-performing trial, and supports early stopping.
"""

import uuid as uuid_pkg
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, SQLModel


class TuningStrategy(StrEnum):
    """Search strategy for hyperparameter exploration."""

    GRID = "grid"
    RANDOM = "random"


class TuningStatus(StrEnum):
    """Lifecycle states for a tuning session.

    A session moves PENDING -> RUNNING -> COMPLETED on the happy path, or to
    FAILED (unrecoverable error) / CANCELLED (user requested a stop).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _tuning_status_values(enum_cls) -> list[str]:
    return [member.value for member in enum_cls]


def _tuning_strategy_values(enum_cls) -> list[str]:
    return [member.value for member in enum_cls]


class TuningSession(SQLModel, table=True):
    """A hyperparameter tuning session that orchestrates multiple training trials.

    Each session defines a search space and strategy (grid or random), spawns
    training jobs for each trial, tracks the best result, and optionally stops
    early when a threshold is reached.
    """

    __tablename__ = "tuning_session"

    id: str = Field(
        default_factory=lambda: str(uuid_pkg.uuid4()),
        primary_key=True,
        max_length=36,
    )
    model_id: int = Field(
        sa_column=Column(
            "model_id",
            ForeignKey("model_basic.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )
    strategy: TuningStrategy = Field(
        sa_column=Column(
            SAEnum(TuningStrategy, native_enum=False, length=20, values_callable=_tuning_strategy_values),
            nullable=False,
        )
    )
    # Search space definition.  Format:
    # {
    #   "optimizer": ["adam", "sgd", "rmsprop"],           # discrete list
    #   "learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2},
    #   "batch_size": [16, 32, 64],                       # discrete list
    #   "epochs": [20, 50]                                # discrete list
    # }
    search_space: dict = Field(sa_column=Column(JSON, nullable=False))
    max_trials: int = Field(default=20)
    metric: str = Field(default="val_accuracy", max_length=50)
    direction: str = Field(default="maximize", max_length=20)  # "maximize" or "minimize"
    early_stop_threshold: float | None = Field(default=None, nullable=True)
    best_job_id: str | None = Field(
        default=None,
        sa_column=Column(
            String(36),
            ForeignKey("training_job.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    status: TuningStatus = Field(
        default=TuningStatus.PENDING,
        sa_column=Column(
            SAEnum(TuningStatus, native_enum=False, length=20, values_callable=_tuning_status_values),
            nullable=False,
            index=True,
        ),
    )
    total_trials: int = Field(default=0)
    completed_trials: int = Field(default=0)
    early_stopped: bool = Field(default=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime, nullable=False),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
    )
