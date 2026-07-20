"""add cascade delete to training_job foreign key

Adds ON DELETE CASCADE to training_job.model_id FK constraint so that
deleting a model automatically deletes its associated training_job records.

Revision ID: g5h6i7j8k9l0
Revises: 209c98a79bb7
Create Date: 2026-07-19 10:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "g5h6i7j8k9l0"
down_revision = "209c98a79bb7"
branch_labels = None
depends_on = None


def upgrade():
    """Add ON DELETE CASCADE to training_job.model_id FK constraint."""
    # Drop the existing FK constraint (no cascade)
    op.drop_constraint("training_job_model_id_fkey", "training_job", type_="foreignkey")

    # Re-create it with ON DELETE CASCADE
    op.create_foreign_key(
        "training_job_model_id_fkey",
        "training_job",
        "model_basic",
        ["model_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    """Remove ON DELETE CASCADE from training_job.model_id FK constraint."""
    # Drop the CASCADE FK constraint
    op.drop_constraint("training_job_model_id_fkey", "training_job", type_="foreignkey")

    # Re-create it without CASCADE
    op.create_foreign_key(
        "training_job_model_id_fkey",
        "training_job",
        "model_basic",
        ["model_id"],
        ["id"],
    )
