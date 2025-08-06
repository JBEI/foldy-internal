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
from app.models import CampaignRound, FewShot, Fold, Invokation
from app.util import get_job_type_replacement
from app.views.other_views import few_shot_fields
from folde.few_shot_models import is_valid_few_shot_model_name

ns = Namespace("few_shot_views", decorators=[jwt_required(fresh=True)])

simple_upload_parser = ns.parser()
simple_upload_parser.add_argument("name", type=str, location="form", required=True)
simple_upload_parser.add_argument("fold_id", type=str, location="form", required=True)
simple_upload_parser.add_argument(
    "activity_file_bytes", type=FileStorage, location="files", required=False
)
simple_upload_parser.add_argument("activity_file_path", type=str, location="form", required=False)
simple_upload_parser.add_argument(
    "activity_file_from_few_shot_id", type=int, location="form", required=False
)
simple_upload_parser.add_argument(
    "activity_file_from_campaign_round_id", type=int, location="form", required=False
)


@ns.route("/few_shots/create_few_shot_directory")
class UploadActivityFileResource(Resource):
    @verify_has_edit_access
    @ns.expect(simple_upload_parser)
    def post(self) -> None:
        args = simple_upload_parser.parse_args()

        # Get form data
        name: str = args["name"]
        fold_id: int = int(args["fold_id"])
        few_shot_directory: Path = Path("few_shots") / name

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
            activity_file = activity_file_blob.open("rb").read()  # type: ignore[reportAssignmentType] # blob.read() return type ambiguity
        elif args["activity_file_from_few_shot_id"]:
            few_shot_for_activity_file = FewShot.query.get(args["activity_file_from_few_shot_id"])
            if not few_shot_for_activity_file:
                raise BadRequest(f"FewShot not found {args['activity_file_from_few_shot_id']}")

            try:
                assert (
                    few_shot_for_activity_file.input_activity_fpath is not None
                ), f"FewShot {args['activity_file_from_few_shot_id']} has no input_activity_fpath"
                activity_file_blob = fsm.storage_manager.get_blob(
                    fold_id, few_shot_for_activity_file.input_activity_fpath
                )
                activity_file = activity_file_blob.open("rb").read()  # type: ignore[reportAssignmentType] # blob.read() return type ambiguity
            except Exception as e:
                raise BadRequest(
                    f"Failed to get activity file from few shot {args['activity_file_from_few_shot_id']}: {e}"
                )
        elif args["activity_file_from_campaign_round_id"]:
            campaign_round = CampaignRound.query.get(args["activity_file_from_campaign_round_id"])
            if not campaign_round:
                raise BadRequest(
                    f"CampaignRound not found {args['activity_file_from_campaign_round_id']}"
                )

            if not campaign_round.result_activity_fpath:
                raise BadRequest(
                    f"CampaignRound {args['activity_file_from_campaign_round_id']} has no activity file"
                )

            try:
                activity_file_blob = fsm.storage_manager.get_blob(
                    fold_id, campaign_round.result_activity_fpath
                )
                activity_file = activity_file_blob.open("rb").read()  # type: ignore[reportAssignmentType] # blob.read() return type ambiguity
            except Exception as e:
                raise BadRequest(
                    f"Failed to get activity file from campaign round {args['activity_file_from_campaign_round_id']}: {e}"
                )
        else:
            raise BadRequest(
                "activity_file_bytes, activity_file_path, activity_file_from_few_shot_id, or activity_file_from_campaign_round_id is required"
            )

        fsm.storage_manager.write_file(
            fold_id=fold_id,
            file_path=str(
                few_shot_directory / "activity.xlsx"
            ),  # or whatever path/extension you want
            contents=activity_file,
            binary=True,
        )

        fsm.storage_manager.delete_folder(
            fold_id, str(few_shot_directory), allow_list_suffixes=["activity.xlsx"]
        )
        return


@ns.route("/few_shot")
class FewShotResource(Resource):

    # @ns.consumes('multipart/form-data')
    @verify_has_edit_access
    @ns.expect(few_shot_fields)
    @ns.marshal_with(few_shot_fields)
    def post(self) -> FewShot:
        """Create a new slate build job with activity data file.

        Returns:
            Newly created FewShot record

        Raises:
            BadRequest: If required fields are missing or if fold is not found
        """
        req = request.get_json()

        print(f"request.data: {request.data}", flush=True)
        print(f"request.is_json: {request.is_json}", flush=True)

        # Get form data
        name: str = req["name"]
        fold_id: int = int(req["fold_id"])
        few_shot_directory: Path = Path("few_shots") / name

        mode: str = req["mode"]
        try:
            embedding_files: Optional[List[str]] = (
                req["embedding_files"].split(",") if "embedding_files" in req else None
            )
        except Exception as e:
            raise BadRequest(f"Failed loading embedding_files {e}")

        try:
            naturalness_files: Optional[List[str]] = (
                req["naturalness_files"].split(",") if "naturalness_files" in req else None
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
                raise BadRequest(
                    "embedding_files and naturalness_files are required for few shot models"
                )
        else:
            raise BadRequest(f"Invalid mode: {mode}")

        # 0. Check if a few shot job with this name already exists.
        fold = Fold.query.get(fold_id)
        if not fold:
            raise BadRequest(f"Fold not found {fold_id}")

        # Make sure the folder and existing few shot have been cleared.
        existing_few_shot = FewShot.query.filter(
            FewShot.name == name, FewShot.fold_id == fold_id
        ).first()
        if existing_few_shot:
            # Delete existing few shot job.
            logging.info(f"Deleting existing few shot job {existing_few_shot.id} for {name}")
            existing_few_shot.delete()

        fsm = FoldStorageManager()
        fsm.setup()
        assert fsm.storage_manager is not None
        input_activity_fpath = None
        for file_dict in fsm.storage_manager.list_files(fold_id, str(few_shot_directory)):
            if file_dict["key"].endswith("activity.xlsx"):
                input_activity_fpath = str(few_shot_directory / file_dict["key"].lstrip("/"))
                continue
            raise BadRequest(
                f"Slate build directory {few_shot_directory} has superfluous files maybe from an old run, found {file_dict['key']}"
            )
        if not input_activity_fpath:
            raise BadRequest(
                f"Slate build directory {few_shot_directory} is empty, no activity.xlsx file found"
            )

        # 2. Create an invokation record for the slate build job.
        new_invokation_id: int = get_job_type_replacement(fold, f"few_shot_{name}")

        # 3. Create a new FewShot record.
        few_shot_record: FewShot = FewShot.create(
            name=name,
            fold_id=fold_id,
            mode=mode,
            embedding_files=",".join(embedding_files) if embedding_files else None,
            naturalness_files=",".join(naturalness_files) if naturalness_files else None,
            finetuning_model_checkpoint=finetuning_model_checkpoint,
            invokation_id=new_invokation_id,
            few_shot_params=few_shot_params,
            num_mutants=num_mutants,
            input_activity_fpath=input_activity_fpath,
        )

        # 4. Start the job based on mode
        enqueued_job: Job

        if mode == "finetuning":
            enqueued_job = get_queue("esm").enqueue(
                esm_jobs.finetune_esm_model,
                few_shot_record.id,
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
                on_success=Callback(send_success_email, timeout="10s"),
                on_failure=Callback(send_failure_email, timeout="10s"),
            )
            add_meta_to_job(enqueued_job, fold, "few_shot", few_shot_record.id)

            logging.info(
                f"Queued finetuning job {enqueued_job.id} for slate build {few_shot_record.id}"
            )
        else:
            enqueued_job = get_queue("cpu").enqueue(
                evolve_jobs.run_evolvepro,
                few_shot_record.id,
                job_timeout="6h",
                on_success=Callback(send_success_email, timeout="10s"),
                on_failure=Callback(send_failure_email, timeout="10s"),
            )
            add_meta_to_job(enqueued_job, fold, "few_shot", few_shot_record.id)

            logging.info(
                f"Queued {mode} job {enqueued_job.id} for slate build {few_shot_record.id}"
            )

        return few_shot_record


@ns.route("/few_shots/<int:few_shot_id>")
class SingleFewShotResource(Resource):
    @ns.marshal_with(few_shot_fields)
    def get(self, few_shot_id: int) -> FewShot:
        """Get slate build record by ID.

        Args:
            few_shot_id: ID of the slate build to retrieve

        Returns:
            FewShot record
        """
        few_shot = FewShot.query.get(few_shot_id)
        if not few_shot:
            raise BadRequest(f"FewShot not found {few_shot_id}")
        return few_shot

    @verify_has_edit_access
    def delete(self, few_shot_id: int) -> None:
        """Delete a slate build record by ID.

        Args:
            few_shot_id: ID of the slate build to delete
        """
        few_shot = FewShot.query.get(few_shot_id)
        if not few_shot:
            raise BadRequest(f"FewShot not found {few_shot_id}")

        manager = FoldStorageManager()
        manager.setup()

        assert manager.storage_manager is not None

        manager.storage_manager.delete_folder(few_shot.fold_id, f"few_shots/{few_shot.name}")

        if few_shot.invokation_id:
            invokation = Invokation.query.get(few_shot.invokation_id)
            if invokation:
                invokation.delete()

        few_shot.delete()

        return None
