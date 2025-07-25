"""Configuration management for the reverse proxy server."""

import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False
    log_level: str = "INFO"


@dataclass
class AuthConfig:
    """Authentication configuration settings."""
    miner_hotkey: str
    allowed_delta_ms: int = 8000
    cache_duration: int = 3600
    chain_endpoint: str = "wss://entrypoint-finney.opentensor.ai:443"


@dataclass
class ServiceConfig:
    """Internal service configuration."""
    training_server_url: str = "http://localhost:7000"
    inference_server_url: str = "http://localhost:7001"


@dataclass
class Config:
    """Complete application configuration."""
    server: ServerConfig
    auth: AuthConfig
    services: ServiceConfig


def load_config() -> Config:
    """Load configuration from environment variables."""
    # Required environment variables
    miner_hotkey = os.getenv("MINER_HOTKEY")
    if not miner_hotkey:
        raise ValueError("MINER_HOTKEY environment variable must be set")
    
    # Server configuration
    server_config = ServerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    
    # Authentication configuration
    auth_config = AuthConfig(
        miner_hotkey=miner_hotkey,
        allowed_delta_ms=int(os.getenv("ALLOWED_DELTA_MS", "8000")),
        cache_duration=int(os.getenv("CACHE_DURATION", "3600")),
        chain_endpoint=os.getenv("CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443"),
    )
    
    # Service configuration
    service_config = ServiceConfig(
        training_server_url=os.getenv("TRAINING_SERVER_URL", "http://localhost:7000"),
        inference_server_url=os.getenv("INFERENCE_SERVER_URL", "http://localhost:7001"),
    )
    
    return Config(
        server=server_config,
        auth=auth_config,
        services=service_config,
    )


# Global configuration instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = load_config()
    return config 