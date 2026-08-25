"""Add foreign-key lookup indexes and logical uniqueness constraints.

Revision ID: 3f8a9c1d2e4b
Revises: 7c9c3b1f4a2e
Create Date: 2026-07-11
"""

from alembic import op

revision = "3f8a9c1d2e4b"
down_revision = "7c9c3b1f4a2e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_invokation_fold_id", "invokation", ["fold_id"])

    op.create_index(
        "ix_docking_receptor_fold_id_ligand_name",
        "docking",
        ["receptor_fold_id", "ligand_name"],
    )
    op.create_index("ix_docking_invokation_id", "docking", ["invokation_id"])

    op.create_unique_constraint("uq_logits_fold_id_name", "logits", ["fold_id", "name"])
    op.create_index("ix_logits_invokation_id", "logits", ["invokation_id"])

    op.create_index("ix_embeddings_fold_id", "embeddings", ["fold_id"])
    op.create_index("ix_embeddings_invokation_id", "embeddings", ["invokation_id"])

    op.create_unique_constraint(
        "uq_fold_evolution_fold_id_name", "fold_evolution", ["fold_id", "name"]
    )
    op.create_index("ix_fold_evolution_invokation_id", "fold_evolution", ["invokation_id"])

    op.create_unique_constraint("uq_campaigns_fold_id_name", "campaigns", ["fold_id", "name"])

    op.create_unique_constraint(
        "uq_campaign_rounds_campaign_id_round_number",
        "campaign_rounds",
        ["campaign_id", "round_number"],
    )
    op.create_index(
        "ix_campaign_rounds_naturalness_run_id", "campaign_rounds", ["naturalness_run_id"]
    )
    op.create_index("ix_campaign_rounds_few_shot_run_id", "campaign_rounds", ["few_shot_run_id"])


def downgrade() -> None:
    op.drop_index("ix_campaign_rounds_few_shot_run_id", table_name="campaign_rounds")
    op.drop_index("ix_campaign_rounds_naturalness_run_id", table_name="campaign_rounds")
    op.drop_constraint(
        "uq_campaign_rounds_campaign_id_round_number", "campaign_rounds", type_="unique"
    )

    op.drop_constraint("uq_campaigns_fold_id_name", "campaigns", type_="unique")

    op.drop_index("ix_fold_evolution_invokation_id", table_name="fold_evolution")
    op.drop_constraint("uq_fold_evolution_fold_id_name", "fold_evolution", type_="unique")

    op.drop_index("ix_embeddings_invokation_id", table_name="embeddings")
    op.drop_index("ix_embeddings_fold_id", table_name="embeddings")

    op.drop_index("ix_logits_invokation_id", table_name="logits")
    op.drop_constraint("uq_logits_fold_id_name", "logits", type_="unique")

    op.drop_index("ix_docking_invokation_id", table_name="docking")
    op.drop_index("ix_docking_receptor_fold_id_ligand_name", table_name="docking")

    op.drop_index("ix_invokation_fold_id", table_name="invokation")
