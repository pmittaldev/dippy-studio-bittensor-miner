# reverse_proxy/Dockerfile
FROM python:3.10-slim

WORKDIR /
COPY . /reverse_proxy

# speed/security basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install -r reverse_proxy/requirements.txt

# the server in the repo is named server.py; change if different
EXPOSE 8080
CMD ["python", "-m", "reverse_proxy.server"]
