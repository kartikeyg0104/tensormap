"""re-add phase 4 columns to training_job

Re-adds analysis_cache, last_export_download_at, and tuning_session_id
columns that were removed earlier but are now needed for Phase 4
(interpretability) and export tracking features in Week 8.

Revision ID: h6i7j8k9l0m1
Revises: g5h6i7j8k9l0
Create Date: 2026-07-20 12:50:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "h6i7j8k9l0m1"
down_revision = "g5h6i7j8k9l0"
branch_labels = None
depends_on = None


def upgrade():
    """Re-add Phase 4 columns to training_job table."""
    op.add_column("training_job", sa.Column("analysis_cache", postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column("training_job", sa.Column("last_export_download_at", sa.DateTime(), nullable=True))
    op.add_column("training_job", sa.Column("tuning_session_id", sa.String(length=36), nullable=True))


def downgrade():
    """Remove Phase 4 columns from training_job table."""
    op.drop_column("training_job", "tuning_session_id")
    op.drop_column("training_job", "last_export_download_at")
    op.drop_column("training_job", "analysis_cache")
