from flask_restx import Namespace, Resource

from app.extensions import db
from app.helpers.jobs_util import mark_invokations_failed_if_stale
from app.models import Invokation

ns = Namespace("open_views")


@ns.route("/check_for_dead_jobs")
class CheckForDeadJobsResource(Resource):
    def get(self) -> bool:
        """Identify and handle dead jobs.

        Search for invokations with 'pending', 'running', or 'queued' state.
        Any with stale heartbeats are marked as failed.
        """
        # Get all invokations with 'pending', 'running', or 'queued' state
        invokations = Invokation.query.filter(
            Invokation.state.in_(["pending", "running", "queued"])
        ).all()
        stale_count = mark_invokations_failed_if_stale(invokations)
        if stale_count:
            db.session.commit()
        return True
