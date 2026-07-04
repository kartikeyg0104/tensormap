"""remove_unused_training_job_columns

Revision ID: 209c98a79bb7
Revises: b2c3d4e5f6a7
Create Date: 2026-06-29 21:02:01.807763

Remove speculative columns (analysis_cache, last_export_download_at,
tuning_session_id) that were reserved for future phases but unused in the
current implementation. This avoids coupling the schema to unfinalized designs.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '209c98a79bb7'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade():
    # Drop unused forward-looking columns
    op.drop_column('training_job', 'tuning_session_id')
    op.drop_column('training_job', 'last_export_download_at')
    op.drop_column('training_job', 'analysis_cache')


def downgrade():
    # Restore columns if downgrading
    op.add_column('training_job', sa.Column('analysis_cache', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column('training_job', sa.Column('last_export_download_at', sa.DateTime(), nullable=True))
    op.add_column('training_job', sa.Column('tuning_session_id', sa.String(length=36), nullable=True))
