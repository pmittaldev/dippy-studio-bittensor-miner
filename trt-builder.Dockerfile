FROM piyushcse29/dippy-miner:latest

COPY scripts/trt_build_in_container.sh /workspace/scripts/trt_build_in_container.sh
RUN chmod +x /workspace/scripts/trt_build_in_container.sh

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_TOKEN="" \
    TRT_BUILD_SAFE=0 \
    USE_FLASH_ATTENTION=1 \
    XFORMERS_DISABLED=0 \
    MODEL_PATH=black-forest-labs/FLUX.1-dev \
    TRT_CACHE_DIR=/trt-cache

VOLUME ["/trt-cache", "/root/.cache/huggingface", "/workspace/scripts"]

# GPU flags are runtime concerns; you’ll still run with --gpus all
# and you can add --ipc=host if needed for some ops.
ENTRYPOINT ["/workspace/scripts/trt_build_in_container.sh"]

