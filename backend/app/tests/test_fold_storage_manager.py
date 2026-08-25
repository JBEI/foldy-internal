import random
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from flask import Flask
from werkzeug.exceptions import BadRequest

from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.jobs.evolve_jobs import run_few_shot_prediction
from app.models import Evolution, Fold, Invokation, User


def test_paginated_fold_query_loads_only_summary_relationships(app, test_fold):
    """List queries use a fixed query count and do not hydrate detail collections."""
    with app.app_context():
        fold = db.session.get(Fold, test_fold.id)
        assert fold is not None
        db.session.add_all(
            [
                Invokation(fold_id=fold.id, type="boltz", state="finished"),
                Invokation(fold_id=fold.id, type="annotate", state="queued"),
            ]
        )
        db.session.commit()
        db.session.expire_all()

        statements: list[str] = []

        def record_statement(
            conn, cursor, statement, parameters, context, executemany  # noqa: ARG001
        ) -> None:
            statements.append(statement)

        from sqlalchemy import event

        event.listen(db.engine, "before_cursor_execute", record_statement)
        try:
            result = FoldStorageManager().get_folds_with_pagination(
                filter=None,
                tag=None,
                only_public=False,
                page=1,
                per_page=25,
            )
        finally:
            event.remove(db.engine, "before_cursor_execute", record_statement)

        assert len(result["data"]) == 1
        listed_fold = result["data"][0]
        assert {job.type for job in listed_fold.jobs} == {"boltz", "annotate"}
        assert "docks" not in listed_fold.__dict__
        assert "naturalness_runs" not in listed_fold.__dict__
        assert "embeddings" not in listed_fold.__dict__
        assert "few_shots" not in listed_fold.__dict__

        # count, page, and select-in-loaded jobs; this stays constant as page size grows.
        select_statements = [
            statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
        ]
        assert len(select_statements) == 3


def test_write_fastas(app, client, tmp_path, test_fold):
    fsm = FoldStorageManager()
    fsm.setup()
    fsm.write_fastas(test_fold.id, test_fold.yaml_config)
