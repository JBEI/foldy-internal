# We use the same CUDA image in all our Dockerfiles.
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

COPY backend/requirements.txt /backend/
COPY worker/scripts/setup.sh /scripts/setup.sh
RUN /scripts/setup.sh

ENV APP_HOME /src
WORKDIR $APP_HOME

RUN /opt/conda/envs/worker/bin/pip install gunicorn
# Copy local code to the container image.
COPY backend/src/ ./

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD flask rq worker --burst all
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app_dispatch
CMD exec /opt/conda/envs/worker/bin/gunicorn --bind :$PORT --workers 1 --threads 3 --timeout 0 --log-level=debug --keep-alive 30 main:app_dispatch
