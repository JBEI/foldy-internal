#!/usr/bin/env python
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import argparse
import logging
import os
import signal
import sys

import redis
from app.factory import create_app
from app.helpers.rq_helpers import get_redis_connection
from rq import SimpleWorker


class GracefulWorker(SimpleWorker):
    """
    Fast but *graceful* shutdown:
      • on SIGTERM/SIGINT → mark stop-requested
      • forward SIGTERM to the horse
      • let RQ’s own loop clean up & mark the job failed
    """

    def request_stop(self, signum, frame):
        self.log.warning("%s received – requesting warm shutdown", signal.Signals(signum).name)

        # Record when we asked so RQ's debounce still works
        from datetime import datetime

        self._shutdown_requested_date = datetime.now()

        # Tell RQ main loop to break after current job
        self._stop_requested = True

        # Do *not* wait() or raise SystemExit here.
        # monitor_work_horse() will reap exactly once and
        # handle_job_failure() will fire if the job dies.


# Parse command line arguments
parser = argparse.ArgumentParser(description="Run RQ worker")
parser.add_argument("queues", nargs="+", help="Queues to listen on")
parser.add_argument("--burst", action="store_true", help="Run in burst mode")
parser.add_argument(
    "--max-jobs", type=int, help="Maximum number of jobs to process before quitting"
)
args = parser.parse_args()


def main():
    # Initialize Flask app
    app = create_app("rq_worker_settings")

    with app.app_context():
        # Get Redis connection from Flask app config
        redis_conn = get_redis_connection()

        # Create and run worker with the specific queues
        worker = GracefulWorker(
            args.queues,
            connection=redis_conn,
            # You can add other worker config here:
            # job_timeout=app.config.get('RQ_DEFAULT_TIMEOUT')
        )

        print(f"Worker listening on queues: {', '.join(args.queues)}")
        worker.work(burst=args.burst, max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
