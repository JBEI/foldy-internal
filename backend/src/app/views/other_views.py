import io
import logging
import re
import time
from datetime import datetime
from typing import (
    Any,
    BinaryIO,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
from flask import (
    Response,
    current_app,
    make_response,
    request,
    send_file,
    stream_with_context,
)
from flask_jwt_extended import jwt_required
from flask_jwt_extended.utils import get_jwt, get_jwt_identity
from flask_restx import Namespace, Resource, fields, reqparse
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.authorization import user_jwt_grants_edit_access, verify_has_edit_access
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.rq_helpers import get_queue
from app.jobs import esm_jobs, other_jobs
from app.models import Dock, Fold, Invokation, User
from app.util import get_job_type_replacement, make_new_folds, start_stage

ns = Namespace("other_views", decorators=[jwt_required(fresh=True)])


@ns.route("/test")
class TestResource(Resource):
    def get(self):
        return "Healthy"


simple_invokation_fields = ns.model(
    "SimpleInvokationFields",
    {
        "id": fields.Integer(required=True),
        "type": fields.String(required=False),
        "state": fields.String(required=False),
        "command": fields.String(required=False),
    },
)

full_invokation_fields = ns.clone(
    "InvokationFields",
    simple_invokation_fields,
    {
        "job_id": fields.String(required=False),
        "log": fields.String(required=False),
        "timedelta_sec": fields.Float(
            readonly=True,
            required=False,
            attribute=lambda r: (r.timedelta.total_seconds() if r and r.timedelta else None),
        ),
        "starttime": fields.DateTime(
            format="iso8601Z", dt_format="iso8601", readonly=True, required=False
        ),
    },
)


dock_fields = ns.model(
    "DockFields",
    {
        "id": fields.Integer(readonly=True),
        "fold_id": fields.Integer(required=True),
        "ligand_name": fields.String(required=True),
        "ligand_smiles": fields.String(required=True),
        "tool": fields.String(required=False),
        "bounding_box_residue": fields.String(
            required=False,
            nullable=True,
            help="Residue to center bounding box on, like Y74.",
        ),
        "bounding_box_radius_angstrom": fields.Float(
            required=False, nullable=True, help="Radius of bounding box in angstroms."
        ),
        "invokation_id": fields.Integer(required=False),
        "pose_energy": fields.Float(required=False),
        "pose_confidences": fields.String(required=False),
    },
)

naturalness_fields = ns.model(
    "NaturalnessFields",
    {
        "id": fields.Integer(required=False),
        "name": fields.String(required=True),
        "fold_id": fields.Integer(required=False),
        "logit_model": fields.String(required=True),
        "use_structure": fields.Boolean(required=False),
        "get_depth_two_logits": fields.Boolean(required=False),
        "output_fpath": fields.String(required=False),
        "output_fpath_computed": fields.String(required=False),
        "invokation_id": fields.Integer(required=False),
        "date_created": fields.DateTime(required=False),
    },
)

embedding_fields = ns.model(
    "EmbeddingFields",
    {
        "name": fields.String(required=True),
        "fold_id": fields.Integer(required=True),
        "embedding_model": fields.String(required=True),
        "id": fields.Integer(required=False),
        "extra_seq_ids": fields.String(required=False),
        "dms_starting_seq_ids": fields.String(required=False),
        "homolog_fasta": fields.String(required=False),
        "extra_layers": fields.String(required=False),
        "domain_boundaries": fields.String(required=False),
        "output_fpath": fields.String(required=False),
        "output_fpath_computed": fields.String(required=False),
        "invokation_id": fields.Integer(required=False),
        "date_created": fields.DateTime(required=False),
    },
)

few_shot_fields = ns.model(
    "FewShotFields",
    {
        "id": fields.Integer(required=False),
        "name": fields.String(required=False),
        "fold_id": fields.Integer(required=False),
        "mode": fields.String(required=False),
        "embedding_files": fields.String(nullable=True),
        "naturalness_files": fields.String(nullable=True),
        "finetuning_model_checkpoint": fields.String(nullable=True),
        "invokation_id": fields.Integer(required=False),
        "few_shot_params": fields.String(nullable=True),
        "num_mutants": fields.Integer(required=False),
        "input_activity_fpath": fields.String(required=False, nullable=True),
        "output_fpath": fields.String(required=False, nullable=True),
        "output_fpath_computed": fields.String(required=False, nullable=True),
        "date_created": fields.DateTime(required=False, nullable=True),
    },
)

get_folds_fields = ns.model("GetFolds", {"filter": fields.String(required=False)})


def log_getattr(a, field, default, debuginfo):
    returnval = getattr(a, field, default)
    # logging.error(f"Got {returnval} for {field} ({debuginfo})")
    return returnval


fold_fields = ns.model(
    "Fold",
    {
        "id": fields.Integer(readonly=True, required=False),
        "name": fields.String(),
        "owner": fields.String(attribute="user.email", required=False),
        "tags": fields.List(fields.String()),
        "create_date": fields.DateTime(
            format="iso8601Z", dt_format="iso8601", readonly=True, required=False
        ),
        "public": fields.Boolean(required=False),
        "yaml_config": fields.String(required=False),
        "diffusion_samples": fields.Integer(required=False),
        "disable_relaxation": fields.Boolean(required=False),
        "jobs": fields.List(fields.Nested(simple_invokation_fields)),
        "docks": fields.List(
            fields.Nested(dock_fields),
            attribute=lambda x: (
                [] if log_getattr(x, "_skip_embedded_fields", False, "docks") else x.docks
            ),
        ),
        "naturalness_runs": fields.List(
            fields.Nested(naturalness_fields),
            attribute=lambda x: (
                []
                if log_getattr(x, "_skip_embedded_fields", False, "naturalness_runs")
                else x.naturalness_runs
            ),
        ),
        "embeddings": fields.List(
            fields.Nested(embedding_fields),
            attribute=lambda x: (
                [] if log_getattr(x, "_skip_embedded_fields", False, "embeddings") else x.embeddings
            ),
        ),
        "few_shots": fields.List(
            fields.Nested(few_shot_fields),
            attribute=lambda x: (
                [] if log_getattr(x, "_skip_embedded_fields", False, "few_shots") else x.few_shots
            ),
        ),
        # Old AF2 inputs.
        "sequence": fields.String(required=False),
        "af2_model_preset": fields.String(required=False),
    },
)

new_folds_fields = ns.model(
    "NewFolds",
    {
        "folds_data": fields.List(fields.Nested(fold_fields, skip_none=True)),
        "start_fold_job": fields.Boolean(required=False),
        "email_on_completion": fields.Boolean(required=False),
        "skip_duplicate_entries": fields.Boolean(required=False),
        "is_dry_run": fields.Boolean(required=False),
    },
)

# Pagination metadata model
pagination_fields = ns.model(
    "Pagination",
    {
        "page": fields.Integer(),
        "per_page": fields.Integer(),
        "total": fields.Integer(required=False),
        "pages": fields.Integer(required=False),
        "has_prev": fields.Boolean(required=False),
        "has_next": fields.Boolean(required=False),
    },
)

# Pagination response model
paginated_folds_fields = ns.model(
    "PaginatedFolds",
    {
        "data": fields.List(fields.Nested(fold_fields, skip_none=True)),
        "pagination": fields.Nested(pagination_fields),
    },
)


get_folds_parser = reqparse.RequestParser()
get_folds_parser.add_argument(
    "filter",
    type=str,
    help="Optional filter parameter.",
    required=False,
)
get_folds_parser.add_argument(
    "tag",
    type=str,
    help="Optionally filter to this tag.",
    required=False,
)
get_folds_parser.add_argument(
    "page",
    type=int,
    help="Page number to retrieve.",
    required=False,
)
get_folds_parser.add_argument(
    "per_page",
    type=int,
    help="Number of items per page.",
    required=False,
)


@ns.route("/fold")
class FoldsResource(Resource):
    # TODO(jbr): Figure out what is causing this call to fail and add validation.
    @ns.expect(new_folds_fields, validate=False)
    @verify_has_edit_access
    def post(self):
        """Returns True if queueing is successful."""
        fsm = FoldStorageManager()
        fsm.setup()

        folds_data = request.get_json()["folds_data"]
        start_fold_job = request.get_json()["start_fold_job"]
        email_on_completion = request.get_json().get("email_on_completion", False)
        skip_duplicate_entries = request.get_json().get("skip_duplicate_entries", False)
        is_dry_run = request.get_json().get("is_dry_run", False)

        return make_new_folds(
            fsm,
            get_jwt_identity(),
            folds_data,
            start_fold_job,
            email_on_completion,
            skip_duplicate_entries,
            is_dry_run,
        )


@ns.route("/paginated_fold")
class PaginatedFoldsResource(Resource):
    @ns.expect(get_folds_parser)
    @ns.marshal_with(paginated_folds_fields, skip_none=True)
    def get(self):
        start_time = time.time()
        args = get_folds_parser.parse_args()
        print(args, flush=True)

        filter = args.get("filter", None)
        tag = args.get("tag", None)
        page = args.get("page", None)
        per_page = args.get("per_page", None)

        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        manager = FoldStorageManager()
        manager.setup()

        folds = manager.get_folds_with_pagination(filter, tag, only_public, page, per_page)
        logging.error(
            f"Returning {len(folds['data'])} folds in {time.time() - start_time} seconds",
        )
        for fold in folds["data"]:
            fold._skip_embedded_fields = True
        return folds


# Tags response model
tag_info_fields = ns.model(
    "TagInfo",
    {
        "tag": fields.String(required=True, description="The tag name"),
        "fold_count": fields.Integer(required=True, description="Number of folds with this tag"),
        "contributors": fields.List(
            fields.String, description="Users who have folds with this tag"
        ),
        "recent_folds": fields.List(fields.String, description="Recent fold names with this tag"),
    },
)

tags_response_fields = ns.model(
    "TagsResponse", {"tags": fields.List(fields.Nested(tag_info_fields))}
)


@ns.route("/tags")
class TagsResource(Resource):
    @ns.marshal_with(tags_response_fields)
    def get(self):
        """Get all tags with their fold counts and contributors."""
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        # Build query based on access permissions
        query = db.session.query(Fold).join(User)
        if only_public:
            query = query.filter(Fold.public == True)

        # Get all folds with tags, ordered by creation date (newest first)
        folds_with_tags = (
            query.filter(Fold.tagstring != "")
            .filter(Fold.tagstring.isnot(None))
            .order_by(Fold.create_date.desc())
            .all()
        )

        # Process tags
        tag_info = {}
        for fold in folds_with_tags:
            if fold.tagstring:
                tags = [tag.strip() for tag in fold.tagstring.split(",") if tag.strip()]
                for tag in tags:
                    if tag not in tag_info:
                        tag_info[tag] = {
                            "tag": tag,
                            "fold_count": 0,
                            "contributors": set(),
                            "recent_folds": [],
                            "most_recent_fold_date": fold.create_date or datetime.min,
                        }

                    tag_info[tag]["fold_count"] += 1
                    # Only add non-None user names to contributors
                    if fold.user.name is not None:
                        tag_info[tag]["contributors"].add(fold.user.name)

                    # Update most recent fold date for this tag
                    if (
                        fold.create_date
                        and fold.create_date > tag_info[tag]["most_recent_fold_date"]
                    ):
                        tag_info[tag]["most_recent_fold_date"] = fold.create_date

                    # Keep track of recent folds (limit to 5)
                    if len(tag_info[tag]["recent_folds"]) < 5:
                        tag_info[tag]["recent_folds"].append(fold.name)

        # Convert sets to lists
        result = []
        for tag_data in tag_info.values():
            tag_data["contributors"] = sorted(list(tag_data["contributors"]))
            result.append(tag_data)

        # Sort by most recent fold date (descending), then by fold count
        try:
            result.sort(
                key=lambda x: (tag_info[x["tag"]]["most_recent_fold_date"], x["fold_count"]),
                reverse=True,
            )
        except Exception as e:
            print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF ")
            print(tag_info, flush=True)
            raise e

        return {"tags": result}


@ns.route("/fold/<int:fold_id>")
class FoldResource(Resource):
    @ns.marshal_with(fold_fields)
    def get(self, fold_id):
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        manager = FoldStorageManager()
        manager.setup()
        return manager.get_fold_with_state(fold_id, only_public)

    @ns.expect(fold_fields, validate=False)
    @verify_has_edit_access
    def post(self, fold_id):
        try:
            fields_to_update = request.get_json()
            if "tags" in fields_to_update:
                for tag in fields_to_update["tags"]:
                    if not re.match(r"^[a-zA-Z0-9_-]+$", tag):
                        raise BadRequest(
                            f"Invalid tag: {tag} contains a character which is not a letter, number, hyphen, or underscore."
                        )
                fields_to_update["tagstring"] = ",".join(fields_to_update["tags"])
                del fields_to_update["tags"]
            fold = Fold.get_by_id(fold_id)
            if fold:
                fold.update(**fields_to_update)
            else:
                raise BadRequest("Fold not found")
            return True
        except Exception as e:
            raise BadRequest(f"Update operation failed {e}")


@ns.route("/invokation/<int:invokation_id>")
class InvokationLogsResource(Resource):
    @ns.marshal_with(full_invokation_fields)
    def get(self, invokation_id):
        return Invokation.get_by_id(invokation_id)


def convert_array_to_json_string(table: np.ndarray) -> str:
    """Convert a numpy array of floats to json compatible table.

    Useful when array.tolist would take up too much space
    (which is the case for big contact maps and PAE).

    Args:
        table: 2D numpy array containing data to convert

    Returns:
        String with JSON array format representation of data
    """
    json_table = "["
    for ii, row in enumerate(table):
        if ii > 0:
            json_table += ",\n"
        row_str = "[" + ", ".join([f"{e:.4g}" for e in row]) + "]"
        json_table += row_str
    json_table += " ]"
    return json_table


pae_fields = ns.model(
    "PAE",
    {
        "pae": fields.List(fields.List(fields.Float(readonly=True))),
    },
)


@ns.route("/pae/<int:fold_id>/<int:model_number>")
class PaeResource(Resource):
    # We can't marshal, since we're not returning json.
    # @ns.marshal_with(pae_fields)
    def get(self, fold_id: int, model_number: int) -> Response:
        """Get predicted aligned error (PAE) matrix for a fold model.

        Args:
            fold_id: ID of the fold
            model_number: Model number to retrieve

        Returns:
            Response with PAE matrix in JSON format
        """
        try:
            manager = FoldStorageManager()
            manager.setup()

            print(
                f"Retrieving PAE data for fold_id={fold_id}, model_number={model_number}",
                flush=True,
            )

            try:
                pae = manager.get_model_pae(fold_id, model_number)

                # Check if the PAE data is valid
                if pae is None:
                    print(
                        f"PAE data is None for fold_id={fold_id}, model_number={model_number}",
                        flush=True,
                    )
                    return make_response({"error": "PAE data not found"}, 404)

                if hasattr(pae, "files"):
                    pae = pae["pae"]

                if not isinstance(pae, np.ndarray):
                    message = f"PAE data is not a numpy array: {type(pae)} {pae.files}"
                    print(message, flush=True)
                    return make_response({"error": message}, 500)

                if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
                    print(f"PAE data has invalid shape: {pae.shape}", flush=True)
                    return make_response({"error": f"Invalid PAE matrix shape: {pae.shape}"}, 500)

                print(
                    f"Successfully retrieved PAE data with shape {pae.shape}",
                    flush=True,
                )

                # Note that using "tolist" and then json.dumps (or jsonify) would take
                # up too much memory, so we convert to json manually.
                json_resp = '{ "pae": ' + convert_array_to_json_string(pae) + " }"
                resp = make_response(
                    json_resp,
                )
                resp.headers["Content-Type"] = "application/json"
                resp.status_code = 200
                return resp

            except BadRequest as e:
                print(f"BadRequest error retrieving PAE: {str(e)}", flush=True)
                return make_response({"error": str(e)}, 400)

        except Exception as e:
            print(f"Unexpected error retrieving PAE: {str(e)}", flush=True)
            import traceback

            traceback.print_exc()
            return make_response({"error": f"Failed to retrieve PAE data: {str(e)}"}, 500)


contact_prob_fields = ns.model(
    "ContactProb",
    {
        "contact_prob": fields.List(fields.List(fields.Float(readonly=True))),
    },
)


@ns.route("/contact_prob/<int:fold_id>/<int:model_number>")
class ContactProbResource(Resource):
    # We can't marshal, since we're not returning json.
    # @ns.marshal_with(contact_prob_fields)
    def get(self, fold_id, model_number):
        manager = FoldStorageManager()
        manager.setup()
        contact_prob = manager.get_contact_prob(fold_id, model_number)

        json_resp = '{ "contact_prob": ' + convert_array_to_json_string(contact_prob) + " }"

        # with np.printoptions(threshold=np.inf):
        #   json_resp = ('{ "contact_prob": ' + np.array2string(
        #     contact_prob,
        #     precision=3,
        #     separator=',',
        #     suppress_small=True,
        #   ) + ' }')
        resp = make_response(
            json_resp,
        )
        resp.headers["Content-Type"] = "application/json"
        resp.status_code = 200

        return resp


@ns.route("/pfam/<int:fold_id>")
class PfamResource(Resource):
    def get(self, fold_id):
        manager = FoldStorageManager()
        manager.setup()
        pfam_annotations = manager.get_pfam(fold_id)

        return pfam_annotations


@ns.route("/dock")
class DockCreateResource(Resource):
    # TODO(jbr): Figure out what is causing this call to fail and add validation.
    # @ns.expect(dock_fields)
    @verify_has_edit_access
    def post(self) -> bool:
        """Create a new docking run for a fold with specified ligand.

        Returns:
            True if docking job was queued successfully

        Raises:
            BadRequest: If ligand name is not alphanumeric or tool is not supported
        """
        req = request.get_json()

        ligand_name = req["ligand_name"]
        ligand_smiles = req["ligand_smiles"]
        tool = req["tool"]
        fold_id = req["fold_id"]

        if not ligand_name.isalnum():
            raise BadRequest(f"Ligand names must be alphanumeric, {ligand_name} is invalid.")

        ALLOWED_DOCKING_TOOLS = ["vina", "diffdock"]
        if not tool in ALLOWED_DOCKING_TOOLS:
            raise BadRequest(f"Invalid docking tool {tool}: must be one of {ALLOWED_DOCKING_TOOLS}")

        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise BadRequest("Fold not found")

        new_invokation_id = get_job_type_replacement(fold, f"dock_{ligand_name}")

        new_dock: Dock = Dock(
            ligand_name=ligand_name,
            ligand_smiles=ligand_smiles,
            tool=tool,
            receptor_fold_id=fold_id,
            invokation_id=new_invokation_id,
            bounding_box_residue=req.get("bounding_box_residue", None),
            bounding_box_radius_angstrom=req.get("bounding_box_radius_angstrom", None),
        )
        new_dock.save()

        cpu_q = get_queue("cpu")
        job = cpu_q.enqueue(
            other_jobs.run_dock,
            new_dock.id,
            new_invokation_id,
            job_timeout="2h",
            result_ttl=48 * 60 * 60,  # 2 days
        )

        logging.info(f"Queued docking job {job.id} for fold {fold_id} with ligand {ligand_name}")
        return True


@ns.route("/dock/<int:dock_id>")
class DockResource(Resource):
    def delete(self, dock_id: int) -> bool:
        """Delete a docking run by ID.

        Args:
            dock_id: ID of the dock to delete

        Returns:
            True if deletion was successful
        """
        dock = Dock.get_by_id(dock_id)

        if not dock:
            raise BadRequest(f"Dock with ID {dock_id} not found")

        logging.info(f"Deleting dock {dock_id} (ligand: {dock.ligand_name})")
        dock.delete()

        return True


queue_job_fields = ns.model(
    "QueueFold",
    {
        "fold_id": fields.String(),
        "stage": fields.String(),
        "email_on_completion": fields.Boolean(),
    },
)


@ns.route("/queuejob")
class QueueJobResource(Resource):
    @ns.expect()
    @verify_has_edit_access
    def post(self):
        start_stage(
            request.get_json()["fold_id"],
            request.get_json()["stage"],
            request.get_json()["email_on_completion"],
        )
