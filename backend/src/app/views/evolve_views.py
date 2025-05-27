import json
import logging
from pathlib import Path
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields
from google.cloud.storage import Blob
from google.cloud.storage.blob import BlobReader
from rq import Callback
from rq.job import Job
from sklearn.ensemble import RandomForestRegressor
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest

from app.authorization import verify_has_edit_access
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager, LocalBlob
from app.helpers.rq_helpers import (
    add_meta_to_job,
    get_queue,
    send_failure_email,
    send_success_email,
)
from app.helpers.sequence_util import (
    maybe_get_seq_id_error_message,
)
from app.jobs import esm_jobs, evolve_jobs
from app.models import Evolution, Fold, Invokation
from app.util import get_job_type_replacement
from app.views.other_views import evolution_fields
from folde.few_shot_models import is_valid_few_shot_model_name

ns = Namespace("evolve_views", decorators=[jwt_required(fresh=True)])

simple_upload_parser = ns.parser()
simple_upload_parser.add_argument("name", type=str, location="form", required=True)
simple_upload_parser.add_argument("fold_id", type=str, location="form", required=True)
simple_upload_parser.add_argument("activity_file_bytes", type=FileStorage, location="files", required=False)
simple_upload_parser.add_argument("activity_file_path", type=str, location="form", required=False)
simple_upload_parser.add_argument("activity_file_from_evolution_id", type=int, location="form", required=False)

@ns.route('/evolve/create_evolve_directory')
class UploadActivityFileResource(Resource):
    @verify_has_edit_access
    @ns.expect(simple_upload_parser)
    def post(self) -> None:
        args = simple_upload_parser.parse_args()

        # Get form data
        name: str = args["name"]
        fold_id: int = int(args["fold_id"])
        evolve_directory: Path = Path("evolve") / name

        fsm = FoldStorageManager()
        fsm.setup()
        assert fsm.storage_manager is not None

        activity_file: bytes
        if args["activity_file_bytes"]:
            activity_file_input: FileStorage = args["activity_file_bytes"]
            activity_file_input.seek(0)
            activity_file = activity_file_input.read()
        elif args["activity_file_path"]:
            activity_file_path = Path(args["activity_file_path"])
            activity_file_blob = fsm.storage_manager.get_blob(fold_id, str(activity_file_path))
            activity_file = activity_file_blob.open('rb').read()
        elif args["activity_file_from_evolution_id"]:
            evolution_for_activity_file = Evolution.query.get(args["activity_file_from_evolution_id"])
            if not evolution_for_activity_file:
                raise BadRequest(f"Evolution not found {args['activity_file_from_evolution_id']}")

            activity_file_path = Path('evolve') / evolution_for_activity_file.name / 'activity.xlsx'
            activity_file_blob = fsm.storage_manager.get_blob(fold_id, str(activity_file_path))
            activity_file = activity_file_blob.open('rb').read()
        else:
            raise BadRequest("activity_file_bytes or activity_file_path is required")

        fsm.storage_manager.write_file(
            fold_id=fold_id,
            file_path=str(
                evolve_directory / "activity.xlsx"
            ),  # or whatever path/extension you want
            contents=activity_file,
            binary=True,
        )

        fsm.storage_manager.delete_folder(fold_id, str(evolve_directory), allow_list_suffixes=["activity.xlsx"])

        return

@ns.route("/evolve")
class EvolveResource(Resource):

    # @ns.consumes('multipart/form-data')
    @verify_has_edit_access
    @ns.expect(evolution_fields)
    @ns.marshal_with(evolution_fields)
    def post(self) -> Evolution:
        """Create a new evolution job with activity data file.

        Returns:
            Newly created Evolution record

        Raises:
            BadRequest: If required fields are missing or if fold is not found
        """
        req = request.get_json()

        print(f'request.data: {request.data}', flush=True)
        print(f'request.is_json: {request.is_json}', flush=True)

        # Get form data
        name: str = req["name"]
        fold_id: int = int(req["fold_id"])
        evolve_directory: Path = Path("evolve") / name

        mode: str = req["mode"]
        try:
            embedding_files: Optional[List[str]] = req['embedding_files'].split(',') if 'embedding_files' in req else None
        except Exception as e:
            raise BadRequest(f"Failed loading embedding_files {e}")

        try:
            naturalness_files: Optional[List[str]] = (
                req['naturalness_files'].split(',') if 'naturalness_files' in req else None
            )
        except Exception as e:
            raise BadRequest(f"Failed loading naturalness_files {e}")

        finetuning_model_checkpoint: Optional[str] = req.get("finetuning_model_checkpoint", None)
        few_shot_params: Optional[str] = req.get("few_shot_params", None)
        num_mutants: int = req["num_mutants"]

        if mode == "randomforest" or mode == "mlp":
            if not embedding_files:
                raise BadRequest("embedding_files are required for randomforest mode")
        elif mode == "finetuning":
            if not finetuning_model_checkpoint:
                raise BadRequest("finetuning_model_checkpoint is required for finetuning mode")
        elif is_valid_few_shot_model_name(mode):
            if not few_shot_params:
                raise BadRequest("few_shot_params are required for few shot models")
            if not embedding_files or not naturalness_files:
                raise BadRequest("embedding_files and naturalness_files are required for few shot models")
        else:
            raise BadRequest(f"Invalid mode: {mode}")

        # 0. Check if an evolve job with this name already exists.
        fold = Fold.query.get(fold_id)
        if not fold:
            raise BadRequest(f"Fold not found {fold_id}")

        # Make sure the folder and existing evolve have been cleared.
        existing_evolve = Evolution.query.filter(
            Evolution.name == name, Evolution.fold_id == fold_id
        ).first()
        if existing_evolve:
            # Delete existing evolve job.
            logging.info(f"Deleting existing evolution job {existing_evolve.id} for {name}")
            existing_evolve.delete()

        fsm = FoldStorageManager()
        fsm.setup()
        assert fsm.storage_manager is not None
        found_activity_file = False
        for file_dict in fsm.storage_manager.list_files(fold_id, str(evolve_directory)):
            if file_dict['key'].endswith("activity.xlsx"):
                found_activity_file = True
                continue
            raise BadRequest(f"Evolve directory {evolve_directory} has superfluous files maybe from an old run, found {file_dict['key']}")
        if not found_activity_file:
            raise BadRequest(f"Evolve directory {evolve_directory} is empty, no activity.xlsx file found")

        # 2. Create an invokation record for the evolve job.
        new_invokation_id: int = get_job_type_replacement(fold, f"evolve_{name}")

        # 3. Create a new Evolution record.
        evolve_record: Evolution = Evolution.create(
            name=name,
            fold_id=fold_id,
            mode=mode,
            embedding_files=",".join(embedding_files) if embedding_files else None,
            naturalness_files=",".join(naturalness_files) if naturalness_files else None,
            finetuning_model_checkpoint=finetuning_model_checkpoint,
            invokation_id=new_invokation_id,
            few_shot_params=few_shot_params,
            num_mutants=num_mutants
        )

        # 4. Start the job based on mode
        enqueued_job: Job

        if mode == "finetuning":
            enqueued_job = get_queue("esm").enqueue(
                esm_jobs.finetune_esm_model,
                evolve_record.id,
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
                on_success=Callback(send_success_email, timeout='10s'),
                on_failure=Callback(send_failure_email, timeout='10s'),
            )
            add_meta_to_job(enqueued_job, fold, "evolve", evolve_record.id)

            logging.info(
                f"Queued finetuning job {enqueued_job.id} for evolution {evolve_record.id}"
            )
        else:
            enqueued_job = get_queue("cpu").enqueue(
                evolve_jobs.run_evolvepro,
                evolve_record.id,
                job_timeout="6h",
                on_success=Callback(send_success_email, timeout='10s'),
                on_failure=Callback(send_failure_email, timeout='10s'),
            )
            add_meta_to_job(enqueued_job, fold, "evolve", evolve_record.id)

            logging.info(f"Queued {mode} job {enqueued_job.id} for evolution {evolve_record.id}")

        return evolve_record


@ns.route('/evolve/<int:evolution_id>')
class SingleEvolveResource(Resource):
    @ns.marshal_with(evolution_fields)
    def get(self, evolution_id: int) -> Evolution:
        """Get evolution record by ID.

        Args:
            evolution_id: ID of the evolution to retrieve

        Returns:
            Evolution record
        """
        evolution = Evolution.query.get(evolution_id)
        if not evolution:
            raise BadRequest(f"Evolution not found {evolution_id}")
        return evolution

    @verify_has_edit_access
    def delete(self, evolution_id: int) -> None:
        """Delete an evolution record by ID.

        Args:
            evolution_id: ID of the evolution to delete
        """
        evolution = Evolution.query.get(evolution_id)
        if not evolution:
            raise BadRequest(f"Evolution not found {evolution_id}")

        manager = FoldStorageManager()
        manager.setup()

        assert manager.storage_manager is not None

        manager.storage_manager.delete_folder(evolution.fold_id, f'evolve/{evolution.name}')

        if evolution.invokation_id:
            invokation = Invokation.query.get(evolution.invokation_id)
            if invokation:
                invokation.delete()

        evolution.delete()

        return None
