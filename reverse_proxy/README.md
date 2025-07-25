# Bittensor Miner Reverse Proxy

A FastAPI-based reverse proxy server for Bittensor subnet miners with Epistula authentication.

## Overview

This reverse proxy serves as the public-facing entrypoint for the Bittensor miner, providing:
- Epistula protocol authentication 
- Request routing to internal training and inference servers
- Security layer between external validators and internal services

## Setup

### Prerequisites

- Python 3.8 or higher
- A valid Bittensor hotkey for authentication

### Installation

#### Using uv (Recommended)

```bash
# Install the package in development mode with all dependencies
uv pip install -e .[dev]
```

#### Using pip

```bash
# Install the package in development mode
pip install -e .[dev]

# Or for production (without dev dependencies)
pip install -e .
```

### Environment Configuration

Create a `.env` file in the reverse_proxy directory:

```bash
# Copy the example file and customize it
cp env.example .env
# Then edit .env with your specific values
```

The `env.example` file contains all available configuration options with detailed comments. Key settings include:

**Required:**
- `MINER_HOTKEY` - Your Bittensor miner hotkey for Epistula authentication

**Important Optional:**
- `HOST` / `PORT` - Server binding configuration  
- `TRAINING_SERVER_URL` / `INFERENCE_SERVER_URL` - Internal service endpoints
- `CHAIN_ENDPOINT` - Bittensor chain endpoint (mainnet/testnet)
- `LOG_LEVEL` - Logging verbosity

## Usage

### Running the Server

#### Using the CLI command (after installation)

```bash
reverse-proxy
```

#### Using Python module

```bash
python -m reverse_proxy.server
```

#### Direct execution

```bash
python reverse_proxy/server.py
```

### API Endpoints

- `GET /` - Health check endpoint
- `GET /status` - Service status with component readiness
- `POST /train` - Proxy training requests (requires authentication)
- `POST /inference` - Proxy inference requests (requires authentication)

### Authentication

All `/train` and `/inference` endpoints require Epistula authentication. Include the appropriate authorization headers as per the Epistula protocol specification.

## Development

### Code Quality Tools

The project includes comprehensive tooling configuration:

```bash
# Format code
black reverse_proxy/

# Sort imports
isort reverse_proxy/

# Lint code
flake8 reverse_proxy/

# Type checking
mypy reverse_proxy/

# Run tests
pytest
```

### Project Structure

```
reverse_proxy/
├── __init__.py              # Package initialization
├── __main__.py              # Module execution entry point
├── server.py                # Main FastAPI server
├── config.py                # Configuration management
├── env.example              # Environment variables example
├── epistula/                # Authentication module
│   ├── __init__.py
│   └── lib.py              # Epistula verifier implementation
├── pyproject.toml          # Project configuration
├── requirements.txt        # Legacy requirements (see pyproject.toml)
└── README.md               # This file
```

### Dependencies

Core dependencies:
- `bittensor` - Bittensor subnet integration
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `substrate-interface` - Substrate blockchain interface
- `loguru` - Structured logging
- `base58` - Address encoding
- `websockets` - WebSocket support
- `python-dotenv` - Environment variable loading

Development dependencies include testing, linting, and formatting tools.

## Configuration

All configuration is handled through environment variables. See the Environment Configuration section above for details.

### Security Considerations

- The reverse proxy should be the only public-facing component
- Internal training and inference servers should only be accessible from the reverse proxy
- Configure CORS settings appropriately for your deployment environment
- Ensure proper firewall rules are in place

## Monitoring

The server provides structured logging with configurable levels. Monitor the following:

- Authentication success/failure rates
- Request proxying performance
- Internal service connectivity
- Error rates and patterns

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify `MINER_HOTKEY` is correctly set
   - Check chain endpoint connectivity
   - Validate Epistula protocol implementation

2. **Service Connectivity**
   - Verify internal service URLs are correct
   - Check that training/inference servers are running
   - Validate network connectivity between services

3. **Performance Issues**
   - Monitor authentication cache hit rates
   - Check internal service response times
   - Review logging levels and output

### Logs

Logs are structured and include:
- Timestamp
- Log level
- Module/function information
- Detailed message content

Configure log level via the `LOG_LEVEL` environment variable.

## License

MIT License - see the main project repository for details. 