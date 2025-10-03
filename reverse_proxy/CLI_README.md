# Reverse Proxy CLI Usage

The CLI in `reverse_proxy/cli.py` manages miner registry metadata on the Bittensor network. All commands can be invoked either with `python reverse_proxy/cli.py …` or `python -m reverse_proxy.cli …`.

## Available Commands

- `register`: Publish (or republish) your miner's address/port combination.
- `read`: List every registration stored for a subnet.
- `check`: Inspect the raw commit data on-chain and surface parse errors.

Running the module without a subcommand defaults to `register`.

## Prerequisites

- Python environment with the reverse proxy dependencies installed (for example `uv pip install -r reverse_proxy/requirements.txt`).
- A configured Bittensor wallet (`--wallet.name`, `--wallet.hotkey`) accessible on disk.
- Optional environment variables in `.env` if you rely on them; the CLI calls `dotenv.load_dotenv()` through the Bittensor helpers.

## Command Reference

### Register a miner

```bash
python -m reverse_proxy.cli register \
  --network finney \
  --netuid 11 \
  --wallet.name my_coldkey \
  --wallet.hotkey my_hotkey \
  --address ideally_some.hostname \
  --port 9999 \
  --online
```

- Drop `--online` to run offline and validate inputs without committing to the chain.
- Override `--network` with `local` when testing against a development subtensor.
- Note that when you setup your network ingress, https is a prerequisite
### Read all registrations for a subnet

```bash
python -m reverse_proxy.cli read --network finney --netuid 11 --wallet.name my_coldkey
```

Wallet flags are optional here but can help the Bittensor SDK locate your keys/config.

### Check commit data health

```bash
python -m reverse_proxy.cli check \
  --network finney \
  --netuid 11 \
  --wallet.name my_coldkey \
  --verbose
```

Use `--verbose` to print parsing errors for malformed registrations.

## Shared Flags

- `--netuid`: Subnet UID to target (defaults to 11 if omitted).
- `--network`: Bittensor network (`finney` mainnet or `local`).
- Bittensor adds standard logging flags such as `--logging.debug`, `--logging.trace`, and wallet path overrides; run `python -m reverse_proxy.cli --help` to view the full list.
