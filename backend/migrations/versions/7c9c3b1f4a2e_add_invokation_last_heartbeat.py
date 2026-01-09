"""add invokation last heartbeat

Revision ID: 7c9c3b1f4a2e
Revises: c2b1f3d4e5a6
Create Date: 2025-02-14 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "7c9c3b1f4a2e"
down_revision = "c2b1f3d4e5a6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("invokation", schema=None) as batch_op:
        batch_op.add_column(sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    with op.batch_alter_table("invokation", schema=None) as batch_op:
        batch_op.drop_column("last_heartbeat")
