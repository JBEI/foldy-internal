"""API endpoints for bulk Boltz variant-ligand docking batches."""

from typing import Any, Dict

from flask import abort, request
from flask_jwt_extended import jwt_required
from flask_jwt_extended.utils import get_jwt, get_jwt_identity
from flask_restx import Namespace, Resource
from werkzeug.exceptions import BadRequest

from app.authorization import user_jwt_grants_edit_access, verify_has_edit_access
from app.helpers.boltz_bulk_dock import (
    create_batch,
    grade_batch,
    prepare_batch_plan,
    serialize_batch,
)
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import BoltzDockBatch, Fold

ns = Namespace("boltz_dock_views", decorators=[jwt_required(fresh=True)])


def _get_visible_batch(batch_id: int) -> BoltzDockBatch:
    batch = BoltzDockBatch.get_by_id(batch_id)
    if not batch:
        raise BadRequest(f"Boltz docking batch {batch_id} not found.")
    only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])
    if only_public and not batch.source_fold.public:
        abort(403, description="You do not have access to this resource.")
    return batch


@ns.route("/boltz_dock_batches/preview")
class BoltzDockBatchPreviewResource(Resource):
    @verify_has_edit_access
    def post(self) -> Dict[str, Any]:
        """Validate and expand a batch without creating folds or jobs."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise BadRequest("Request body must be a JSON object.")
        source_fold_id = payload.get("source_fold_id")
        if not isinstance(source_fold_id, int):
            raise BadRequest("source_fold_id must be an integer.")
        source_fold = Fold.get_by_id(source_fold_id)
        if not source_fold:
            raise BadRequest(f"Source fold {source_fold_id} not found.")
        plan = prepare_batch_plan(payload, source_fold)
        return {
            "name": plan["name"],
            "source_fold_id": source_fold_id,
            "variant_count": len(plan["variants"]),
            "state_count": len(plan["states"]),
            "ligand_count": len(plan["states"]),
            "job_count": plan["job_count"],
            "variant_ids": [variant["seq_id"] for variant in plan["variants"]],
            "state_names": [state["name"] for state in plan["states"]],
            "ligand_names": [state["name"] for state in plan["states"]],
            "config": plan["config"],
        }


@ns.route("/boltz_dock_batches")
class BoltzDockBatchesResource(Resource):
    def get(self) -> Dict[str, Any]:
        """List batches, optionally scoped to a source fold or campaign round."""
        query = BoltzDockBatch.query.join(Fold, BoltzDockBatch.source_fold_id == Fold.id)
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])
        if only_public:
            query = query.filter(Fold.public == True)

        source_fold_id = request.args.get("source_fold_id", type=int)
        campaign_round_id = request.args.get("campaign_round_id", type=int)
        if source_fold_id is not None:
            query = query.filter(BoltzDockBatch.source_fold_id == source_fold_id)
        if campaign_round_id is not None:
            query = query.filter(BoltzDockBatch.campaign_round_id == campaign_round_id)
        batches = query.order_by(BoltzDockBatch.id.desc()).all()
        return {"batches": [serialize_batch(batch, include_entries=False) for batch in batches]}

    @verify_has_edit_access
    def post(self) -> Any:
        """Create and optionally start a new bulk Boltz docking batch."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise BadRequest("Request body must be a JSON object.")
        fsm = FoldStorageManager()
        fsm.setup()
        batch = create_batch(payload, get_jwt_identity(), fsm)
        return serialize_batch(batch), 201


@ns.route("/boltz_dock_batches/<int:batch_id>")
class BoltzDockBatchResource(Resource):
    def get(self, batch_id: int) -> Dict[str, Any]:
        """Get one batch with all variant-ligand entries and current states."""
        return serialize_batch(_get_visible_batch(batch_id))


@ns.route("/boltz_dock_batches/<int:batch_id>/grade")
class BoltzDockBatchGradeResource(Resource):
    @verify_has_edit_access
    def post(self, batch_id: int) -> Dict[str, Any]:
        """Refresh structural metrics for every completed entry in a batch."""
        batch = _get_visible_batch(batch_id)
        fsm = FoldStorageManager()
        fsm.setup()
        updated = grade_batch(batch, fsm)
        response = serialize_batch(batch)
        response["graded_entries"] = updated
        return response
