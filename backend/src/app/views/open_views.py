import urllib
from re import DEBUG

from app.authorization import (
    email_should_get_edit_permission_by_default,
    email_should_get_upgraded_to_admin,
)
from app.extensions import db, rq
from app.models import Invokation, User
from authlib.integrations.flask_client import OAuth
from flask import current_app, jsonify, redirect, request, url_for
from flask_jwt_extended import (
    create_access_token,
    set_access_cookies,
    unset_jwt_cookies,
)
from flask_restx import Namespace, Resource, fields

ns = Namespace("open_views")


@ns.route("/check_for_dead_jobs")
class CheckForDeadJobsResource(Resource):
    def get(self):
        """Identify and handle dead jobs.

        Search for invokations with 'pending', 'running', or 'queued' status -
        any that are not found in flask-rq2 will be marked as failed.
        """
        # Get all invokations with 'pending', 'running', or 'queued' status
        invokations = Invokation.query.filter(
            Invokation.status.in_(["pending", "running", "queued"])
        ).all()
        # Check if each invokation is in flask-rq2
        for invokation in invokations:
            if invokation.id not in rq.get_queue().get_job_ids():
                invokation.status = "failed"
                db.session.commit()
        return True
