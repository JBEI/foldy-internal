import os

from app.extensions import db
from app.factory import create_app
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = create_app("rq_worker_settings")
