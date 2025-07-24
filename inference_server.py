from diffusers import DiffusionPipeline
import torch
base_flux_path = "/path/to/base/flux"

pipe = DiffusionPipeline.from_pretrained(
    base_flux_path,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

lora_path = "/path/to/lora/path"
pipe.load_lora_weights(lora_path)

# Generate image
prompt = "A cute anime girl wearing a hoodie, sitting on a sofa in the living room sipping coffee"
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    prompt,
    num_inference_steps=28,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    generator=generator,
    # lora_scale=1.0  # Adjust LoRA strength (0.5–1.3)
).images[0]

# Save the output
image.save("output_image.png")


import asyncio
import logging
import os
import random
import tempfile
import traceback
import uuid

import aiohttp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TextToImageInput(BaseModel):
    model: str
    prompt: str
    seed: int
    size: str | None = None
    n: int | None = None


class HttpClient:
    session: aiohttp.ClientSession = None

    def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        await self.session.close()
        self.session = None

    def __call__(self) -> aiohttp.ClientSession:
        assert self.session is not None
        return self.session


class TextToImagePipeline:
    pipeline = None
    device: str = None

    def start(self):
        if torch.cuda.is_available():
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = pipe
        else:
            raise Exception("No CUDA or MPS device available")


app = FastAPI()
service_url = os.getenv("SERVICE_URL", "http://localhost:8000")
image_dir = os.path.join(tempfile.gettempdir(), "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
app.mount("/images", StaticFiles(directory=image_dir), name="images")
http_client = HttpClient()
shared_pipeline = TextToImagePipeline()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, e.g., GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
def startup():
    http_client.start()
    shared_pipeline.start()


def save_image(image):
    filename = "draw" + str(uuid.uuid4()).split("-")[0] + ".png"
    image_path = os.path.join(image_dir, filename)
    # write image to disk at image_path
    logger.info(f"Saving image to {image_path}")
    image.save(image_path)
    return os.path.join(service_url, "images", filename)


@app.get("/")
@app.post("/")
@app.options("/")
async def base():
    return "ok"


@app.post("/generate")
async def generate_image(image_input: TextToImageInput):
    try:
        loop = asyncio.get_event_loop()
        scheduler = shared_pipeline.pipeline.scheduler.from_config(shared_pipeline.pipeline.scheduler.config)
        pipeline = pipe.from_pipe(shared_pipeline.pipeline, scheduler=scheduler)
        generator = torch.Generator(device=shared_pipeline.device)
        generator.manual_seed(image_input.seed)
        output = await loop.run_in_executor(None, lambda: pipeline(image_input.prompt, generator=generator))
        logger.info(f"output: {output}")
        image_url = save_image(output.images[0])
        return {"data": [{"url": image_url}]}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        elif hasattr(e, "message"):
            raise HTTPException(status_code=500, detail=e.message + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e) + traceback.format_exc())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7001)

