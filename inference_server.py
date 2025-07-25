import asyncio
import logging
import os
import tempfile
import traceback
import uuid
from typing import Optional

import aiohttp
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToImageInput(BaseModel):
    model: str
    prompt: str
    seed: int
    size: Optional[str] = None
    n: Optional[int] = None


class HttpClient:
    session: aiohttp.ClientSession = None

    def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()
            self.session = None

    def __call__(self) -> aiohttp.ClientSession:
        assert self.session is not None
        return self.session


class TextToImagePipeline:
    def __init__(self):
        self.pipeline = None
        self.device = None

    def start(self):
        if not torch.cuda.is_available():
            raise Exception("No CUDA device available")
        
        self.device = "cuda"
        logger.info("Loading diffusion pipeline on CUDA")
        
        # Get model paths from environment variables
        base_model_path = os.getenv("BASE_MODEL_PATH")
        lora_path = os.getenv("LORA_PATH")
        
        if not base_model_path:
            # Default to a common Flux model if no path specified
            base_model_path = "black-forest-labs/FLUX.1-schnell"
            logger.info(f"No BASE_MODEL_PATH specified, using default: {base_model_path}")
        
        try:
            # Load the base pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load LoRA weights if path is provided
            if lora_path and os.path.exists(lora_path):
                logger.info(f"Loading LoRA weights from: {lora_path}")
                self.pipeline.load_lora_weights(lora_path)
            elif lora_path:
                logger.warning(f"LoRA path specified but not found: {lora_path}")
            
            self.pipeline = self.pipeline.to(self.device)
            logger.info("Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def generate_image(self, prompt: str, seed: int = 42, **kwargs):
        if not self.pipeline:
            raise Exception("Pipeline not initialized")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Default parameters
        generation_params = {
            "prompt": prompt,
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "generator": generator,
        }
        
        # Update with any additional parameters
        generation_params.update(kwargs)
        
        try:
            result = self.pipeline(**generation_params)
            return result.images[0]
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise


# Initialize FastAPI app
app = FastAPI(title="Inference Server", version="1.0.0")

# Configuration
service_url = os.getenv("SERVICE_URL", "http://localhost:7001")
image_dir = os.path.join(tempfile.gettempdir(), "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Mount static files for serving images
app.mount("/images", StaticFiles(directory=image_dir), name="images")

# Initialize global objects
http_client = HttpClient()
shared_pipeline = TextToImagePipeline()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    http_client.start()
    shared_pipeline.start()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await http_client.stop()


def save_image(image):
    """Save generated image to disk and return URL"""
    filename = f"draw{str(uuid.uuid4()).split('-')[0]}.png"
    image_path = os.path.join(image_dir, filename)
    logger.info(f"Saving image to {image_path}")
    image.save(image_path)
    return f"{service_url}/images/{filename}"


@app.get("/")
@app.post("/")
@app.options("/")
async def base():
    """Health check endpoint"""
    return {"status": "ok", "service": "inference_server"}


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "pipeline_loaded": shared_pipeline.pipeline is not None,
        "device": shared_pipeline.device
    }


@app.post("/generate")
async def generate_image(image_input: TextToImageInput):
    """Generate an image from text prompt"""
    try:
        if not shared_pipeline.pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        logger.info(f"Generating image with prompt: {image_input.prompt[:100]}...")
        
        # Run image generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None, 
            shared_pipeline.generate_image,
            image_input.prompt,
            image_input.seed
        )
        
        # Save image and return URL
        image_url = save_image(image)
        logger.info(f"Image generated successfully: {image_url}")
        
        return {"data": [{"url": image_url}]}
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        logger.error(traceback.format_exc())
        
        if isinstance(e, HTTPException):
            raise e
        
        error_detail = str(e)
        if hasattr(e, "message"):
            error_detail = e.message
            
        raise HTTPException(
            status_code=500, 
            detail=f"Image generation failed: {error_detail}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 7001
    port = int(os.getenv("PORT", 7001))
    
    logger.info(f"Starting inference server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

