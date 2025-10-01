# reverse_proxy/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app/

# speed/security basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# install deps (pyproject/requirements — adjust if the repo uses extras)
RUN pip install --upgrade pip && \
    pip install -e .[dev]

# the server in the repo is named server.py; change if different
EXPOSE 8080
CMD ["python", "server.py"]
