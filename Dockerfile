FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with uv
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Copy application code
COPY training_server.py .
COPY config_generator.py .
COPY run.py .
COPY info.py .
COPY toolkit/ ./toolkit/
COPY extensions_built_in/ ./extensions_built_in/
COPY jobs/ ./jobs/

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8091

# Run the miner axon
CMD ["python", "training_server.py"]