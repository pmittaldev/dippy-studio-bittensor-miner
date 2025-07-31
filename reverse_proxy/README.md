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

### Miner Registry System

The Miner Registry system allows miners to register their network endpoints (IP address and port) on the Bittensor blockchain, enabling validators to discover and connect to miners.

> **Important**: This system is currently **only for use on testnet with subnet/netuid 231**. Do not use on mainnet or other subnets.

#### Overview

The miner registry replaces the previous model submission system with a simplified approach that stores only essential networking information:
- **Address**: IP address or hostname (max 45 characters)
- **Port**: Service port number (0-65535, max 5 characters)

The total commit string is limited to 128 bytes to comply with blockchain constraints.

#### Prerequisites

- Bittensor wallet set up with coldkey and hotkey
- Access to Bittensor testnet (netuid 231)
- Python environment with bittensor installed

#### Commands

The `register.py` script provides three main commands:

##### 1. Register Command

Register your miner's network endpoint on the blockchain.

```bash
python register.py register --network test --netuid 231 --wallet.name <coldkey> --wallet.hotkey <hotkey> --address <ip_address> --port <port> --online
```

**Parameters:**
- `--network test`: Use testnet (required)
- `--netuid 231`: Subnet ID 231 (required)
- `--wallet.name`: Your coldkey name
- `--wallet.hotkey`: Your hotkey name
- `--address`: Your miner's IP address or hostname (e.g., "13.89.38.129")
- `--port`: Your miner's service port (e.g., "9999")
- `--online`: Actually commit to blockchain (omit for dry run)

**Example:**
```bash
python register.py register --network test --netuid 231 --wallet.name coldkey --wallet.hotkey hotkey231 --address "13.89.38.129" --port "9999" --online
```

##### 2. Check Command

View all registered miners on the subnet.

```bash
python register.py check --network test --netuid 231 [--verbose]
```

**Parameters:**
- `--network test`: Use testnet (required)
- `--netuid 231`: Subnet ID 231 (required)
- `--verbose`: Show detailed error information for invalid entries (optional)

**Example (basic):**
```bash
python register.py check --network test --netuid 231
```

**Example (verbose):**
```bash
python register.py check --network test --netuid 231 --verbose
```

##### 3. Read Command

Read all registrations (legacy format support).

```bash
python register.py read --network test --netuid 231
```

#### Output Format

##### Successful Registration
When successfully registered, you'll see:
- Coldkey and hotkey addresses
- Network confirmation (test)
- Compressed string that was committed
- Success message with transaction details

##### Check Command Output
The check command displays:
- Total number of commitments
- Valid commitments section showing:
  - Hotkey
  - Block number
  - Decoded address and port
- Invalid commitments section showing:
  - Hotkey
  - Raw string
  - Error details (only with --verbose)

#### Troubleshooting

##### Common Errors

1. **"Address exceeds maximum length"**
   - Ensure your address is under 45 characters

2. **"Port must be between 0 and 65535"**
   - Use a valid port number

3. **"Commit string length exceeds maximum"**
   - This shouldn't happen with the new format, but check your inputs

4. **No commitments found**
   - Verify you're using the correct network and netuid
   - Ensure your wallet has registered miners

##### Debug Mode

For detailed logging, add `--logging.debug`:

```bash
python register.py register --network test --netuid 231 --wallet.name coldkey --wallet.hotkey hotkey231 --address "13.89.38.129" --port "9999" --logging.debug --online
```

#### Important Notes

1. **Testnet Only**: This system is currently configured for testnet subnet 231 only
2. **No Hotkey Storage**: The hotkey is not stored in the registry data
3. **Simple Format**: Only address:port is stored (e.g., "13.89.38.129:9999")
4. **Size Limit**: Total string must be under 128 characters
5. **Validation**: Port must be a valid number between 0-65535

#### Migration from Old System

If you see entries with the old format (containing namespace, model names, etc.), these will appear in the "Invalid Commitments" section when using the check command. Only new registrations using the address:port format will be valid.

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