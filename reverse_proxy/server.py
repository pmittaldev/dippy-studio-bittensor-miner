"""Main reverse proxy server for Bittensor subnet miner."""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from .epistula import EpistulaVerifier


# Global instances
epistula_verifier: Optional[EpistulaVerifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global epistula_verifier
    
    # Startup
    logger.info("Starting Bittensor Miner Reverse Proxy")
    
    # Initialize Epistula verifier
    try:
        epistula_verifier = EpistulaVerifier(
            allowed_delta_ms=int(os.getenv("ALLOWED_DELTA_MS", "8000")),
            cache_duration=int(os.getenv("CACHE_DURATION", "3600"))
        )
        logger.info("Epistula verifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Epistula verifier: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Bittensor Miner Reverse Proxy")


# Create FastAPI app
app = FastAPI(
    title="Bittensor Miner Reverse Proxy",
    description="Reverse proxy server for Bittensor subnet miner with Epistula authentication",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_epistula_auth(request: Request) -> bool:
    """Dependency to verify Epistula authentication."""
    if not epistula_verifier:
        raise HTTPException(status_code=500, detail="Authentication service not available")
    
    # Extract authentication headers
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not auth_header.startswith("Epistula "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")

    header_signature = auth_header.split(" ", 1)[1].strip()
    signature_header = request.headers.get("Epistula-Signature")
    signature = signature_header or header_signature

    if signature_header and signature_header.strip() != header_signature:
        raise HTTPException(status_code=401, detail="Authorization signature mismatch")

    timestamp = request.headers.get("Epistula-Timestamp")
    uuid_str = request.headers.get("Epistula-UUID")
    signed_by = request.headers.get("Epistula-Signed-By")
    signed_for = request.headers.get("Epistula-Signed-For")

    if not all([signature, timestamp, uuid_str, signed_by, signed_for]):
        raise HTTPException(status_code=401, detail="Missing Epistula authentication headers")

    if not signature.startswith("0x"):
        signature = f"0x{signature}"

    try:
        body = await request.body()
    except Exception as exc:
        logger.error(f"Failed to read request body for authentication: {exc}")
        raise HTTPException(status_code=400, detail="Unable to read request body")

    request._body = body

    now_ms = int(time.time() * 1000)

    error = await epistula_verifier.verify_signature(
        signature=signature,
        body=body,
        timestamp=timestamp,
        uuid_str=uuid_str,
        signed_for=signed_for,
        signed_by=signed_by,
        now=now_ms,
        path=request.url.path,
    )

    if error:
        logger.warning(f"Epistula authentication failed: {error}")
        raise HTTPException(status_code=401, detail=error)

    logger.debug(
        f"Epistula authentication succeeded: signed_by={signed_by}, signed_for={signed_for}, uuid={uuid_str}"
    )

    return True


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "bittensor-miner-reverse-proxy"}


@app.get("/status")
async def status():
    """Status endpoint with service information."""
    return {
        "status": "running",
        "service": "bittensor-miner-reverse-proxy",
        "version": "0.1.0",
        "epistula_ready": epistula_verifier is not None
    }


@app.get("/capacity")
async def capacity(authenticated: bool = Depends(verify_epistula_auth)):
    """Return supported capacity information."""
    return {
        "inference": ["base"],
        "training": ["H100pcie"],
    }


@app.post("/train")
async def proxy_training_request(
    request: Request,
    authenticated: bool = Depends(verify_epistula_auth)
):
    """Proxy training requests to the training server."""
    training_server_url = os.getenv("TRAINING_SERVER_URL", "http://localhost:8091")
    # Forward the request to the training server
    async with httpx.AsyncClient() as client:
        try:
            # Get the raw request body
            body = await request.body()
            
            # Forward the request with original headers and body
            response = await client.post(
                f"{training_server_url}{request.url.path}",
                headers=request.headers,
                content=body,
                timeout=300.0  # 5 minute timeout
            )
            
            # Return the proxied response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except httpx.RequestError as e:
            logger.error(f"Error forwarding request to training server: {e}")
            raise HTTPException(status_code=502, detail="Error connecting to training server")
    logger.info("Proxying training request to training server")
    
    return {"message": "Training request received", "status": "forwarded"}


@app.post("/inference")
async def proxy_inference_request(
    request: Request,
    authenticated: bool = Depends(verify_epistula_auth)
):
    """Proxy inference requests to the inference server."""
    inference_server_url = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8091")
    
    # Forward the request to the inference server
    async with httpx.AsyncClient() as client:
        try:
            # Get the raw request body
            body = await request.body()
            
            # Forward the request with original headers and body
            response = await client.post(
                f"{inference_server_url}{request.url.path}",
                headers=request.headers,
                content=body,
                timeout=300.0  # 5 minute timeout
            )
            
            # Return the proxied response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except httpx.RequestError as e:
            logger.error(f"Error forwarding request to inference server: {e}")
            raise HTTPException(status_code=502, detail="Error connecting to inference server")
    
    logger.info("Proxying inference request to inference server")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def main():
    """Main entry point for the server."""
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "reverse_proxy.server:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        access_log=True
    )


if __name__ == "__main__":
    main()
