# CUDA 12.4 runtime to match host Torch (cu124)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    # Correct arch syntax for Hopper:
    TORCH_CUDA_ARCH_LIST="90;90+PTX" \
    # Do NOT bake device selection into the image
    TRT_VERSION=10.5.0 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_CACHE_DISABLE=0 \
    TRT_SUPPRESS_CUDA_WARNINGS=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1


    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TRT_BUILD_SAFE=0 \
    USE_FLASH_ATTENTION=1  \
    XFORMERS_DISABLED=0 \
    MODEL_PATH=black-forest-labs/FLUX.1-dev \
    TRT_CACHE_DIR=/trt-cache \




RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git wget vim build-essential ninja-build \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv

WORKDIR /workspace

# EXACT host Torch matrix
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Make sure torchao is present (needed by your training path)
RUN python3.10 -m pip install --no-cache-dir torchao==0.9.0

# Verify CUDA works (optional but useful for debugging)
RUN python3.10 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
PY

# EXACT TRT
RUN python3.10 -m pip install --no-cache-dir tensorrt==10.5.0 polygraphy>=0.50

# Filter out ONLY torch, torchvision, TensorRT family, and flash-attn
COPY requirements.txt .
RUN awk 'BEGIN{IGNORECASE=1} \
  !/^(torch($|[[:space:]=<>]))/ && \
  !/^(torchvision($|[[:space:]=<>]))/ && \
  !/^(tensorrt($|[[:space:]=<>])|tensorrt-cu12-(bindings|libs)($|[[:space:]=<>]))/ && \
  !/^(flash[-_]attn($|[[:space:]=<>]))/ \
  {print}' requirements.txt > /tmp/req.filtered

# Install the rest
RUN python3.10 -m pip install --no-cache-dir -r /tmp/req.filtered

# App
COPY training_server.py .
COPY config_generator.py .
COPY run.py .
COPY info.py .
COPY toolkit/ ./toolkit/
COPY extensions_built_in/ ./extensions_built_in/
COPY jobs/ ./jobs/
COPY trt.py .
COPY lora_generate_image.py .
COPY miner_server.py .
COPY scripts/ ./scripts/

RUN chmod +x scripts/*.sh
RUN mkdir -p /trt-cache /app/output /app/datasets /app/config /app/models

EXPOSE 8091
RUN chmod +x /workspace/scripts/trt_build_in_container.sh \
 && /workspace/scripts/trt_build_in_container.sh


CMD ["python3", "miner_server.py"]