FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5
WORKDIR /sandbox
COPY runner_v2.py /sandbox/runner.py
COPY invoke.py /sandbox/invoke.py
RUN chmod 555 /sandbox /sandbox/*.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
ENTRYPOINT ["python", "-I", "/sandbox/runner.py"]
