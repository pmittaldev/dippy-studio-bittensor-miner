import os
import asyncio
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import subprocess
import requests
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from config_generator import generate_training_config, cleanup_job_files
from lora_generate_image import TRTInferenceServer, InferenceRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dippy Studio Bittensor Miner Server",
    description="Combined LoRA training and TensorRT inference server",
    version="1.0.0"
)

trt_server: Optional[TRTInferenceServer] = None
training_executor = ThreadPoolExecutor(max_workers=2)
inference_jobs: Dict[str, Dict[str, Any]] = {}
training_jobs: Dict[str, Dict[str, Any]] = {}

class TrainingRequest(BaseModel):
    job_type: str = "lora_training"
    params: Dict[str, Any]
    job_id: str
    validator_endpoint: Optional[str] = None

class InferenceRequestModel(BaseModel):
    prompt: str
    lora_path: Optional[str] = None  # Path to LoRA weights (.safetensors)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    num_inference_steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    seed: Optional[int] = None
    job_id: Optional[str] = None
    callback_url: Optional[str] = Field(
        default=None,
        description="HTTP endpoint that receives the generated image"
    )
    callback_secret: Optional[str] = Field(
        default=None,
        description="Secret forwarded via X-Callback-Secret header"
    )
    expiry: Optional[datetime] = Field(
        default=None,
        description="ISO 8601 timestamp (UTC) after which callbacks are skipped"
    )

class ServerConfig(BaseModel):
    enable_training: bool = Field(default=True, description="Enable training endpoints")
    enable_inference: bool = Field(default=True, description="Enable TRT inference endpoints")
    trt_engine_path: Optional[str] = Field(default=None, description="Path to TRT engine")
    model_path: str = Field(default="black-forest-labs/FLUX.1-dev", description="Base model path")
    output_dir: str = Field(default="./output", description="Output directory")
    preload_engines: bool = Field(default=True, description="Preload TRT engines at startup")


@dataclass
class QueuedInferenceTask:
    job_id: str
    inference_request: InferenceRequest
    callback_url: Optional[str]
    callback_secret: Optional[str]
    expiry: Optional[datetime]
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CallbackStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


def normalize_expiry(expiry: Optional[datetime]) -> Optional[datetime]:
    if expiry is None:
        return None
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=timezone.utc)
    return expiry.astimezone(timezone.utc)


async def wait_for_image(output_path: Path, timeout: Optional[float], poll_interval: float = 0.5) -> Path:
    start_time = time.monotonic()
    while True:
        if output_path.exists():
            return output_path

        if timeout and timeout > 0 and (time.monotonic() - start_time) > timeout:
            raise TimeoutError(
                f"Timed out after {timeout} seconds waiting for image at {output_path}"
            )

        await asyncio.sleep(poll_interval)


async def dispatch_callback(
    task: QueuedInferenceTask,
    status: CallbackStatus,
    *,
    image_path: Optional[Path] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    attempted_at = datetime.now(timezone.utc)
    attempt_iso = attempted_at.isoformat()

    if not task.callback_url:
        return {
            "status": "skipped",
            "reason": "missing_callback_url",
            "attempted_at": attempt_iso,
            "payload_status": status.value
        }

    expiry = task.expiry
    if expiry and attempted_at > expiry:
        return {
            "status": "expired",
            "attempted_at": attempt_iso,
            "expiry": expiry.isoformat(),
            "payload_status": status.value
        }

    callback_timeout = getattr(app.state, "callback_timeout", 30)
    headers = {}
    if task.callback_secret:
        headers["X-Callback-Secret"] = task.callback_secret

    form = aiohttp.FormData()
    form.add_field("job_id", task.job_id)
    form.add_field("status", status.value)
    form.add_field("completed_at", attempt_iso)
    if error:
        form.add_field("error", error)

    image_bytes: Optional[bytes] = None
    if image_path is not None:
        try:
            image_bytes = await asyncio.to_thread(image_path.read_bytes)
        except FileNotFoundError:
            logger.warning(f"Image path not found for job {task.job_id}: {image_path}")
        except Exception as exc:
            logger.error(f"Failed to read generated image for job {task.job_id}: {exc}")
            return {
                "status": "failed",
                "error": f"Unable to read image: {exc}",
                "attempted_at": attempt_iso,
                "payload_status": status.value
            }

    if image_bytes is not None:
        form.add_field("image", image_bytes, filename=image_path.name, content_type="image/png")
        form.add_field("image_url", f"{app.state.SERVICE_URL}/images/{image_path.stem}.png")

    timeout_config = aiohttp.ClientTimeout(total=callback_timeout)

    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(task.callback_url, data=form, headers=headers) as response:
                response_text = await response.text()
                return {
                    "status": "delivered" if response.status < 400 else "error",
                    "status_code": response.status,
                    "response_preview": response_text[:500],
                    "attempted_at": attempt_iso,
                    "payload_status": status.value
                }
    except Exception as exc:
        logger.error(f"Callback delivery failed for job {task.job_id}: {exc}")
        return {
            "status": "failed",
            "error": str(exc),
            "attempted_at": attempt_iso,
            "payload_status": status.value
        }

def load_trt_server():
    """Load TRT server (called at startup or lazily)"""
    global trt_server

    if trt_server is not None:
        return trt_server  # Already loaded

    config = app.state.config
    if config.enable_inference and config.trt_engine_path:
        try:
            logger.info("Initializing TRT inference server...")
            engine_path = Path(config.trt_engine_path)
            mapping_env = os.getenv("TRT_MAPPING_PATH")
            if mapping_env:
                mapping_path = Path(mapping_env)
            else:
                mapping_candidate = engine_path.with_name("mapping.json")
                mapping_path = mapping_candidate if mapping_candidate.exists() else Path("./trt/mapping.json")
            trt_server = TRTInferenceServer(
                base_model_path=config.model_path,
                engine_path=str(engine_path),
                mapping_path=str(mapping_path)
            )
            # Start the inference server in background
            trt_server.start()
            logger.info("TRT inference server loaded successfully")
            return trt_server
        except Exception as e:
            logger.error(f"Failed to load TRT inference server: {e}")
            raise
    return None

def unload_trt_server():
    """Unload TRT server to free GPU memory"""
    global trt_server

    if trt_server is not None:
        logger.info("Unloading TRT server to free GPU memory...")
        try:
            trt_server.stop()
            trt_server = None

            # Force GPU memory cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("TRT server unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading TRT server: {e}")

async def preload_trt_engines():
    """Preload TRT engines at startup"""
    config = app.state.config
    
    if not config.preload_engines or not config.enable_inference:
        logger.info("Engine preloading disabled, will load on first request")
        return
    
    if not config.trt_engine_path:
        logger.warning("No TRT engine path specified, skipping preload")
        return
        
    try:
        logger.info("Preloading TRT engines at startup...")
        start_time = time.time()
        
        # Load TRT server synchronously during startup
        server = load_trt_server()
        if server is None:
            logger.warning("TRT server could not be loaded")
            return
        
        logger.info(f"TRT engines preloaded in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to preload TRT engines: {e}")

async def initialize_servers():
    # Store config globally
    config = ServerConfig(
        enable_training=os.getenv("ENABLE_TRAINING", "true").lower() == "true",
        enable_inference=os.getenv("ENABLE_INFERENCE", "true").lower() == "true",
        trt_engine_path=os.getenv("TRT_ENGINE_PATH", "/app/trt/transformer.plan"),
        model_path=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        output_dir=os.getenv("OUTPUT_DIR", "./output"),
        preload_engines=os.getenv("PRELOAD_ENGINES", "true").lower() == "true"
    )

    # Create output directory if it doesn't exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Store config for later use
    app.state.config = config
    app.state.OUTPUT_DIR = config.output_dir
    # Get service URL from environment or construct from host/port
    service_url = os.getenv("SERVICE_URL")
    if not service_url:
        host = os.getenv("MINER_SERVER_HOST", "0.0.0.0")
        port = os.getenv("MINER_SERVER_PORT", "8091")
        # In production, this should be set to your actual domain
        service_url = f"http://localhost:{port}"
    app.state.SERVICE_URL = service_url

    # Mount static files for serving images
    app.mount("/images", StaticFiles(directory=config.output_dir), name="images")

    logger.info(f"Server initialized. Output directory: {config.output_dir}")
    logger.info(f"Service URL: {app.state.SERVICE_URL}")
    
    # Configure inference queue and workers
    app.state.inference_timeout = int(os.getenv("INFERENCE_TIMEOUT", "300"))
    app.state.callback_timeout = int(os.getenv("CALLBACK_TIMEOUT", "30"))
    queue_maxsize_env = int(os.getenv("INFERENCE_QUEUE_MAXSIZE", "0"))
    queue_maxsize = max(0, queue_maxsize_env)
    worker_count = max(1, int(os.getenv("INFERENCE_WORKERS", "1")))

    if config.enable_inference:
        app.state.inference_queue = asyncio.Queue(maxsize=queue_maxsize)
        app.state.inference_worker_tasks = [
            asyncio.create_task(inference_queue_worker(idx))
            for idx in range(worker_count)
        ]
        queue_desc = "unbounded" if queue_maxsize == 0 else queue_maxsize
        logger.info(
            f"Inference queue ready (workers={worker_count}, maxsize={queue_desc}, timeout={app.state.inference_timeout}s)"
        )
    else:
        app.state.inference_queue = None
        app.state.inference_worker_tasks = []

    # Start TRT engine preloading if enabled
    await preload_trt_engines()

    return config


async def inference_queue_worker(worker_id: int):
    queue: asyncio.Queue = app.state.inference_queue
    logger.info(f"Inference worker {worker_id} started")

    while True:
        task = await queue.get()

        if task is None:
            logger.info(f"Inference worker {worker_id} shutting down")
            queue.task_done()
            break

        job_id = task.job_id
        job_record = inference_jobs.get(job_id, {})
        job_record.update({
            "id": job_id,
            "status": "processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "worker_id": worker_id
        })

        inference_jobs[job_id] = job_record

        try:
            server = load_trt_server()
            if server is None:
                raise RuntimeError("TRT inference server is not available")

            server.submit(task.inference_request)
            output_path = Path(task.inference_request.output_path)

            try:
                await wait_for_image(output_path, getattr(app.state, "inference_timeout", 300))
            except TimeoutError as exc:
                callback_info = await dispatch_callback(task, CallbackStatus.FAILED, error=str(exc))
                job_record.update({
                    "status": "timeout",
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "callback": callback_info
                })
                continue

            completed_at = datetime.now(timezone.utc).isoformat()
            job_record.update({
                "status": "completed",
                "completed_at": completed_at,
                "output_path": str(output_path),
                "result_url": f"/inference/result/{job_id}",
                "image_url": f"{app.state.SERVICE_URL}/images/{job_id}.png"
            })

            callback_info = await dispatch_callback(task, CallbackStatus.COMPLETED, image_path=output_path)
            job_record["callback"] = callback_info

        except Exception as exc:
            logger.error(f"Inference worker {worker_id} failed for job {job_id}: {exc}", exc_info=True)
            callback_info = await dispatch_callback(task, CallbackStatus.FAILED, error=str(exc))
            job_record.update({
                "status": "failed",
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "callback": callback_info
            })
        finally:
            inference_jobs[job_id] = job_record
            queue.task_done()

@app.post("/train", status_code=201)
async def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training asynchronously and return immediately"""

    if request.job_type != "lora_training":
        raise HTTPException(status_code=400, detail="Only lora_training supported")

    try:
        job_id = request.job_id  # must be present
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        # Store job info
        training_jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "type": request.job_type,
            "created_at": datetime.now().isoformat(),
            "params": request.params
        }

        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id,
            request.params,
            request.validator_endpoint
        )

        # Return immediately with 201 OK (empty response body matches training_server.py)
        return

    except Exception as e:
        logger.error(f"Error submitting training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_training_job(job_id: str, params: Dict[str, Any], validator_endpoint: Optional[str]):
    """Background task to run training and post results to validator"""

    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()

        # Unload TRT server before training to free GPU memory
        unload_trt_server()

        # Generate config from params
        config_path = generate_training_config(job_id, params)

        # Run the actual training
        result = subprocess.run([
            "python3", "run.py", config_path, "--seed", str(params.get("seed", 42))
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Training failed for job {job_id}: {result.stderr}")
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = result.stderr
            training_jobs[job_id]["failed_at"] = datetime.now().isoformat()
            return

        # Upload to HuggingFace and get model URL
        huggingface_url = upload_to_huggingface(job_id)

        # Update job status
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["huggingface_url"] = huggingface_url

        # Post results to validator
        if validator_endpoint:
            post_results_to_validator(job_id, huggingface_url, validator_endpoint)

    except Exception as e:
        logger.error(f"Error in background training for job {job_id}: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["failed_at"] = datetime.now().isoformat()
    finally:
        cleanup_job_files(job_id)
        # logger.info(f"Keeping job files for debugging: job_{job_id}")

def upload_to_huggingface(job_id: str) -> str:
    """Upload trained model to HuggingFace and return model URL"""
    from huggingface_hub import HfApi, upload_folder

    model_dir = Path(f"output/job_{job_id}")
    if not model_dir.exists():
        model_dir = Path(app.state.OUTPUT_DIR) / f"job_{job_id}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model output not found for job {job_id} at {model_dir}")

    # Create unique model name
    model_name = f"lora_{job_id}"

    # Upload to HuggingFace (uses token's associated user)
    api = HfApi()
    repo_id = api.create_repo(repo_id=model_name, exist_ok=True).repo_id
    upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model"
    )

    # Return the full HuggingFace URL
    return f"https://huggingface.co/{repo_id}"

def post_results_to_validator(job_id: str, huggingface_url: str, validator_endpoint: str):
    """Post completion results to validator"""
    import requests

    payload = {
        "job_id": job_id,
        "hugging_face_url": huggingface_url
    }

    try:
        response = requests.post(
            validator_endpoint,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Successfully posted results for job {job_id}")

    except requests.RequestException as e:
        logger.error(f"Failed to post results to validator: {e}")

async def notify_validator(endpoint: str, job_id: str, status: str, output_path: str = None, error: str = None):
    import aiohttp

    payload = {
        "job_id": job_id,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    if output_path:
        payload["output_path"] = output_path
    if error:
        payload["error"] = error

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Validator notification failed: {response.status}")
    except Exception as e:
        logger.error(f"Failed to notify validator: {e}")

@app.post("/inference")
async def generate_image(request: InferenceRequestModel):
    config = app.state.config
    if not config.enable_inference:
        raise HTTPException(status_code=503, detail="Inference is currently disabled")

    inference_queue = getattr(app.state, "inference_queue", None)
    if inference_queue is None:
        raise HTTPException(status_code=503, detail="Inference queue is not ready")

    job_id = request.job_id or str(uuid.uuid4())

    existing_job = inference_jobs.get(job_id)
    if existing_job and existing_job.get("status") not in {"completed", "failed", "timeout"}:
        raise HTTPException(status_code=409, detail=f"Job {job_id} is already {existing_job.get('status')}")

    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"

    inference_req = InferenceRequest(
        prompt=request.prompt,
        output_path=str(output_path),
        lora_path=request.lora_path or "",
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed or 42
    )

    expiry = normalize_expiry(request.expiry)

    task = QueuedInferenceTask(
        job_id=job_id,
        inference_request=inference_req,
        callback_url=request.callback_url,
        callback_secret=request.callback_secret,
        expiry=expiry
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    inference_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "queued_at": now_iso,
        "request": {
            "prompt": request.prompt,
            "lora_path": request.lora_path,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "callback_url": request.callback_url,
            "callback_secret_provided": bool(request.callback_secret),
            "expiry": expiry.isoformat() if expiry else None
        }
    }

    try:
        await inference_queue.put(task)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"Failed to enqueue inference job {job_id}: {exc}")
        inference_jobs[job_id]["status"] = "failed"
        inference_jobs[job_id]["error"] = str(exc)
        inference_jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
        raise HTTPException(status_code=500, detail="Failed to queue inference job")

    logger.info(f"Queued inference job {job_id} for prompt '{request.prompt[:50]}...'")

    return {
        "accepted": True,
        "job_id": job_id,
        "status": "queued",
        "message": "Inference job queued",
        "status_url": f"/inference/status/{job_id}",
        "result_url": f"/inference/result/{job_id}"
    }

@app.get("/inference/status/{job_id}")
async def get_inference_status(job_id: str):
    if job_id not in inference_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = inference_jobs[job_id]

    # Check if output exists using global OUTPUT_DIR
    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"
    if output_path.exists() and job.get("status") in {"queued", "processing"}:
        job.update({
            "status": "completed",
            "output_path": str(output_path),
            "result_url": f"/inference/result/{job_id}",
            "image_url": f"{app.state.SERVICE_URL}/images/{job_id}.png",
            "completed_at": job.get("completed_at", datetime.now(timezone.utc).isoformat())
        })
        inference_jobs[job_id] = job

    return job

@app.get("/inference/result/{job_id}")
async def get_inference_result(job_id: str):
    if job_id not in inference_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found. Job may still be processing.")

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"{job_id}.png"
    )

@app.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return training_jobs[job_id]

@app.get("/jobs")
async def list_jobs(job_type: Optional[str] = None):
    if job_type == "training":
        return {"training_jobs": list(training_jobs.values())}
    elif job_type == "inference":
        return {"inference_jobs": list(inference_jobs.values())}
    else:
        return {
            "training_jobs": list(training_jobs.values()),
            "inference_jobs": list(inference_jobs.values())
        }

@app.get("/health")
async def health_check():
    config = app.state.config
    trt_status = "disabled"
    
    if config.enable_inference:
        if trt_server is not None:
            trt_status = "loaded"
        elif config.preload_engines:
            trt_status = "preloading"  # May still be loading in background
        else:
            trt_status = "lazy_load_enabled"
    
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "training": "enabled" if config.enable_training else "disabled",
            "inference": trt_status
        },
        "config": {
            "preload_engines": config.preload_engines,
            "engine_path": config.trt_engine_path
        }
    }

    if trt_server is not None:
        status["services"]["trt_engine"] = "loaded"

    return status


@app.get("/")
async def root():
    return {
        "name": "Dippy Studio Bittensor Miner Server",
        "version": "1.0.0",
        "endpoints": {
            "training": {
                "POST /train": "Submit training job",
                "GET /training/status/{job_id}": "Get training job status"
            },
            "inference": {
                "POST /inference": "Generate image with TRT",
                "GET /inference/status/{job_id}": "Get inference job status",
                "GET /inference/result/{job_id}": "Download generated image"
            },
            "general": {
                "GET /jobs": "List all jobs",
                "GET /health": "Health check",
                "GET /engine/status": "Detailed engine status"
            }
        }
    }

@app.on_event("startup")
async def startup_event():
    await initialize_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global trt_server
    inference_queue = getattr(app.state, "inference_queue", None)
    worker_tasks = getattr(app.state, "inference_worker_tasks", [])

    if inference_queue:
        # Wait for in-flight tasks to complete before shutting down workers
        await inference_queue.join()
        for _ in worker_tasks:
            await inference_queue.put(None)

        await asyncio.gather(*worker_tasks, return_exceptions=True)

    if trt_server:
        trt_server.stop()
    training_executor.shutdown(wait=True)

def main():
    port = int(os.getenv("MINER_SERVER_PORT", "8091"))
    host = os.getenv("MINER_SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting Dippy Studio Bittensor Miner Server on {host}:{port}")

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
