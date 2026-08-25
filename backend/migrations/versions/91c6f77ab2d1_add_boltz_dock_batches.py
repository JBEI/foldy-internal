"""Add persisted Boltz docking batches and matrix results.

Revision ID: 91c6f77ab2d1
Revises: 3f8a9c1d2e4b
Create Date: 2026-08-25
"""

import sqlalchemy as sa
from alembic import op

revision = "91c6f77ab2d1"
down_revision = "3f8a9c1d2e4b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "boltz_dock_batches",
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("source_fold_id", sa.Integer(), nullable=False),
        sa.Column("campaign_round_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("comparison_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["campaign_round_id"],
            ["campaign_rounds.id"],
            onupdate="CASCADE",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_fold_id"], ["roles.id"], onupdate="CASCADE", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], onupdate="CASCADE", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_boltz_dock_batches_source_fold_id", "boltz_dock_batches", ["source_fold_id"]
    )
    op.create_index(
        "ix_boltz_dock_batches_campaign_round_id",
        "boltz_dock_batches",
        ["campaign_round_id"],
    )
    op.create_index("ix_boltz_dock_batches_user_id", "boltz_dock_batches", ["user_id"])

    op.create_table(
        "boltz_dock_results",
        sa.Column("batch_id", sa.Integer(), nullable=False),
        sa.Column("fold_id", sa.Integer(), nullable=False),
        sa.Column("seq_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Text(), nullable=False),
        sa.Column("ligand_name", sa.String(length=120), nullable=False),
        sa.Column("ligand_smiles", sa.Text(), nullable=False),
        sa.Column("state_data", sa.JSON(), nullable=True),
        sa.Column("score_data", sa.JSON(), nullable=True),
        sa.Column("graded_at", sa.DateTime(), nullable=True),
        sa.Column("setup_error", sa.Text(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["boltz_dock_batches.id"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["fold_id"], ["roles.id"], onupdate="CASCADE", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "batch_id", "seq_id", "ligand_name", name="uq_boltz_dock_result_matrix_cell"
        ),
        sa.UniqueConstraint("fold_id", name="uq_boltz_dock_results_fold_id"),
    )
    op.create_index("ix_boltz_dock_results_batch_id", "boltz_dock_results", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_boltz_dock_results_batch_id", table_name="boltz_dock_results")
    op.drop_table("boltz_dock_results")
    op.drop_index("ix_boltz_dock_batches_user_id", table_name="boltz_dock_batches")
    op.drop_index("ix_boltz_dock_batches_campaign_round_id", table_name="boltz_dock_batches")
    op.drop_index("ix_boltz_dock_batches_source_fold_id", table_name="boltz_dock_batches")
    op.drop_table("boltz_dock_batches")
