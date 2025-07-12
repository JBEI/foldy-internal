"""Defines API endpoints related to protein language models."""

import logging
from typing import List

from flask import (
    request,
)
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from rq import Callback
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.authorization import verify_has_edit_access
from app.helpers.rq_helpers import (
    add_meta_to_job,
    get_queue,
    send_failure_email,
    send_success_email,
)
from app.jobs import esm_jobs
from app.models import Embedding, Fold, Logit
from app.util import get_job_type_replacement
from app.views.other_views import embedding_fields, logit_fields

ns = Namespace("esm_views", decorators=[jwt_required(fresh=True)])

ALLOWED_ESM_MODELS: List[str] = [
    "esmc_600m",
    "esmc_300m",
    "esm3-open",
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
    "esm1v",
]

ALLOWED_LOGITS_MODELS: List[str] = ALLOWED_ESM_MODELS + ["esm1v_t33_650M_UR90S_ensemble"]


@ns.route("/embeddings")
class CalculateEmbeddingsResource(Resource):
    @verify_has_edit_access
    @ns.expect(embedding_fields)
    def post(self) -> bool:
        """Create a new embedding calculation job for a fold.

        Args:
            fold_id: ID of the fold to create embeddings for

        Returns:
            True if the embedding job was successfully created

        Raises:
            BadRequest: If embedding model is not allowed or fold doesn't exist
        """
        req = request.get_json()

        fold_id: int = req["fold_id"]
        embedding_name: str = req["name"]
        embedding_model: str = req["embedding_model"]
        extra_seq_ids_str: str = req.get("extra_seq_ids", "")
        dms_starting_seq_ids_str: str = req.get("dms_starting_seq_ids", "")
        extra_layers_str: str = req.get("extra_layers", "")

        extra_seq_ids: list[str] = [
            seq_id.strip() for seq_id in extra_seq_ids_str.split(",") if seq_id.strip()
        ]
        dms_starting_seq_ids: list[str] = [
            seq_id.strip() for seq_id in dms_starting_seq_ids_str.split(",") if seq_id.strip()
        ]
        extra_layers: list[str] = [
            layer.strip() for layer in extra_layers_str.split(",") if layer.strip()
        ]

        if embedding_model not in ALLOWED_ESM_MODELS:
            raise BadRequest(
                f"Invalid embedding model {embedding_model}: must be one of {ALLOWED_ESM_MODELS}"
            )

        fold = Fold.get_by_id(fold_id)

        if not fold:
            raise BadRequest(f"Fold with ID {fold_id} not found")

        new_invokation_id = get_job_type_replacement(fold, f"embed_{embedding_name}")

        embed_record = Embedding.create(
            name=embedding_name,
            fold_id=fold_id,
            embedding_model=embedding_model,
            extra_seq_ids=",".join(extra_seq_ids),
            dms_starting_seq_ids=",".join(dms_starting_seq_ids),
            extra_layers=",".join(extra_layers),
            invokation_id=new_invokation_id,
        )

        esm_q = get_queue("esm")
        enqueued_job = esm_q.enqueue(
            esm_jobs.get_esm_embeddings,
            embed_record.id,
            job_timeout="24h",
            result_ttl=48 * 60 * 60,  # 2 days
            on_success=Callback(send_success_email, timeout="5s"),
            on_failure=Callback(send_failure_email, timeout="5s"),
        )
        add_meta_to_job(enqueued_job, fold, "embed", embed_record.id)

        logging.info(
            f"Queued embedding job {enqueued_job.id} for fold {fold_id}, model {embedding_model}"
        )
        return True


@ns.route("/startlogits/<int:fold_id>")
class StartLogitsResource(Resource):
    @verify_has_edit_access
    @ns.expect(logit_fields)
    @ns.marshal_with(logit_fields)
    def post(self, fold_id: int) -> Logit:
        """Create a new logit calculation job for a fold.

        Args:
            fold_id: ID of the fold to create logits for

        Returns:
            The created Logit record

        Raises:
            BadRequest: If logit model is not allowed or fold doesn't exist
        """
        req = request.get_json()

        name: str = req["name"]
        logit_model: str = req["logit_model"]
        use_structure: bool = req.get("use_structure", False)
        get_depth_two_logits: bool = req.get("get_depth_two_logits", False)

        if logit_model not in ALLOWED_LOGITS_MODELS:
            raise BadRequest(
                f"Invalid logit model {logit_model}: must be one of {ALLOWED_LOGITS_MODELS}"
            )

        fold = Fold.get_by_id(fold_id)

        if not fold:
            raise BadRequest(f"Fold with ID {fold_id} not found")

        existing_logit = Logit.query.filter(Logit.name == name, Logit.fold_id == fold_id).first()
        if existing_logit:
            logging.info(f"Deleting existing logit job {existing_logit.id} for {name}")
            existing_logit.delete()

        new_invokation_id: int = get_job_type_replacement(fold, f"logits_{name}")

        logit_record: Logit = Logit.create(
            name=name,
            fold_id=fold_id,
            logit_model=logit_model,
            use_structure=use_structure,
            get_depth_two_logits=get_depth_two_logits,
            invokation_id=new_invokation_id,
        )

        esm_q = get_queue("esm")
        enqueued_job = esm_q.enqueue(
            esm_jobs.get_esm_logits,
            logit_record.id,
            job_timeout="24h",
            result_ttl=48 * 60 * 60,  # 2 days
            on_success=Callback(send_success_email, timeout="5s"),
            on_failure=Callback(send_failure_email, timeout="5s"),
        )
        add_meta_to_job(enqueued_job, fold, "logits", logit_record.id)

        logging.info(
            f"Queued logit job {enqueued_job.id} for fold {fold_id}, model {logit_model}, "
            f"use_structure={use_structure}, get_depth_two_logits={get_depth_two_logits}"
        )

        return logit_record
