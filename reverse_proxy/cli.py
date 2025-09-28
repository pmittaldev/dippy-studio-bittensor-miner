import argparse
import hashlib
import sys
from typing import Any, ClassVar, Dict, Optional, Type, cast, Union
from pydantic import BaseModel, Field, PositiveInt, field_validator
import bittensor as bt
from bittensor.core.chain_data import decode_account_id
from enum import Enum
import requests

DEFAULT_NETUID = 11

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# Maximum length for IP address (IPv4: 15 chars, IPv6: 45 chars)
MAX_IP_LENGTH = 45
# Maximum length for port (5 chars for 0-65535)
MAX_PORT_LENGTH = 5
# Separator length
SEPARATOR_LENGTH = 1
CHECK_ENDPOINT_TIMEOUT = 3.0


class CommitDataStatus(Enum):
    """Status of commit data parsing"""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    UNKNOWN_FORMAT = "unknown_format"


class MinerRegistry(BaseModel):
    """Stores miner network information for validators to connect"""

    address: str = Field(
        description="IP address or hostname of the miner",
        max_length=MAX_IP_LENGTH
    )
    port: str = Field(
        description="Port number for the miner service",
        max_length=MAX_PORT_LENGTH
    )

    @field_validator('address')
    def validate_address_length(cls, v):
        if len(v) > MAX_IP_LENGTH:
            raise ValueError(f"Address must be at most {MAX_IP_LENGTH} characters")
        return v

    @field_validator('port')
    def validate_port(cls, v):
        if len(v) > MAX_PORT_LENGTH:
            raise ValueError(f"Port must be at most {MAX_PORT_LENGTH} characters")
        try:
            port_int = int(v)
            if not 0 <= port_int <= 65535:
                raise ValueError("Port must be between 0 and 65535")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Port must be a valid number")
            raise
        return v

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.address}:{self.port}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["MinerRegistry"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":", 1)  # Split on first colon only in case IPv6
        if len(tokens) != 2:
            raise ValueError(f"Invalid compressed string format: {cs}")
        return cls(
            address=tokens[0],
            port=tokens[1]
        )
    
    def get_total_length(self) -> int:
        """Calculate total length of compressed string"""
        return len(self.address) + SEPARATOR_LENGTH + len(self.port)


class CommitData(BaseModel):
    """Base class for commit data"""
    hotkey: str
    block: int
    raw_string: str
    string_length: int
    status: CommitDataStatus


class ValidCommitData(CommitData):
    """Represents successfully parsed commit data"""
    registry: MinerRegistry
    status: CommitDataStatus = CommitDataStatus.VALID


class InvalidCommitData(CommitData):
    """Represents commit data that failed to parse"""
    error_message: str
    error_type: str
    status: CommitDataStatus = CommitDataStatus.INVALID_FORMAT


def add_common_args(parser):
    """Add common arguments that all subcommands use"""
    parser.add_argument(
        "--netuid",
        type=int,
        default=DEFAULT_NETUID,
        help=f"The subnet UID. Defaults to {DEFAULT_NETUID}",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="finney",
        choices=["finney", "local"],
        help="Bittensor network selection (finney for mainnet, local for development)",
    )
    
    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)


def get_config():
    """Create argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        description="Miner Registry CLI for Bittensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register on mainnet (subnet 11)
  python register.py register --network finney --netuid 11 --wallet.name coldkey --wallet.hotkey hotkey1 --address '13.89.38.129' --port '9999' --online

  # Read all registrations
  python register.py read --network finney --netuid 11

  # Check your commit data
  python register.py check --network finney --netuid 11 --wallet.name coldkey --wallet.hotkey hotkey1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new miner')
    register_parser.add_argument(
        "--address",
        required=True,
        type=str,
        help=f"IP address or hostname of the miner (max {MAX_IP_LENGTH} chars)",
    )
    register_parser.add_argument(
        "--port",
        required=True,
        type=str,
        help=f"Port number for the miner service (0-65535, max {MAX_PORT_LENGTH} chars)",
    )
    register_parser.add_argument(
        "--online",
        action="store_true",
        help="Make the commit call to the bittensor network",
    )
    add_common_args(register_parser)
    
    # Read command
    read_parser = subparsers.add_parser('read', help='Read all registrations')
    add_common_args(read_parser)
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Dump all commitments for a netuid')
    check_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed error information for invalid commitments"
    )
    add_common_args(check_parser)
    
    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)
    
    # If no command specified, default to register
    if config.command is None:
        config.command = 'register'
        # Re-parse with register as default to get all register args
        import sys
        sys.argv.insert(1, 'register')
        config = bt.config(parser)
    
    return config


def register(config):
    bt.logging(config=config)
    
    # Debug: Print full config
    bt.logging.debug(f"Full config: {config}")
    bt.logging.info(f"Config fields: address={config.address}, port={config.port}")
    bt.logging.info(f"Network: {config.network if hasattr(config, 'network') else 'default'}")
    bt.logging.info(f"Netuid: {config.netuid}")
    bt.logging.info(f"Online mode: {config.online}")

    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet loaded - Coldkey: {wallet.coldkey.ss58_address}")
    
    # Override subtensor network if specified
    if hasattr(config, 'network') and config.network:
        bt.logging.info(f"Creating subtensor with network: {config.network}")
        subtensor = bt.subtensor(network=config.network)
        bt.logging.info(f"Subtensor created - Network: {subtensor.network}, Chain endpoint: {subtensor.chain_endpoint}")
    else:
        bt.logging.info("Creating subtensor with default config")
        subtensor = bt.subtensor(config=config)
        bt.logging.info(f"Subtensor created - Network: {subtensor.network}")

    hotkey = wallet.hotkey.ss58_address
    bt.logging.info(f"Hotkey address: {hotkey}")
    
    # Validate inputs before creating registry
    bt.logging.info("Validating inputs...")
    if len(config.address) > MAX_IP_LENGTH:
        bt.logging.error(f"Address '{config.address}' exceeds maximum length of {MAX_IP_LENGTH}")
        return
        
    if len(config.port) > MAX_PORT_LENGTH:
        bt.logging.error(f"Port '{config.port}' exceeds maximum length of {MAX_PORT_LENGTH}")
        return
    
    # Create registry object
    try:
        miner_registry = MinerRegistry(
            address=config.address,
            port=config.port
        )
    except ValueError as e:
        bt.logging.error(f"Invalid input: {e}")
        return
    
    bt.logging.info(f"MinerRegistry created: {miner_registry}")
    bt.logging.info(f"MinerRegistry dict: {miner_registry.model_dump()}")
    
    registry_commit_str = miner_registry.to_compressed_str()
    bt.logging.info(f"Compressed string: {registry_commit_str}")
    bt.logging.info(f"Compressed string length: {len(registry_commit_str)}")
    bt.logging.info(f"Total allowed length: {MAX_METADATA_BYTES}")
    
    # Check if commit string exceeds limit
    if len(registry_commit_str) > MAX_METADATA_BYTES:
        bt.logging.error(f"Commit string length {len(registry_commit_str)} exceeds maximum {MAX_METADATA_BYTES}")
        bt.logging.error(f"Address length: {len(config.address)}, Port length: {len(config.port)}")
        return
        
    bt.logging.info(f"Compressed string bytes: {registry_commit_str.encode()}")

    bt.logging.info(f"=== Registration Summary ===")
    bt.logging.info(f"Coldkey: {wallet.coldkey.ss58_address}")
    bt.logging.info(f"Hotkey: {hotkey}")
    bt.logging.info(f"Network: {subtensor.network}")
    bt.logging.info(f"Netuid: {config.netuid}")
    bt.logging.info(f"String to commit: {registry_commit_str}")
    bt.logging.info(f"===========================")

    netuid = config.netuid
    if config.online:
        bt.logging.info(f"Online mode - Attempting to commit to chain...")
        bt.logging.info(f"Calling subtensor.commit with:")
        bt.logging.info(f"  - wallet: {wallet}")
        bt.logging.info(f"  - netuid: {netuid}")
        bt.logging.info(f"  - data: {registry_commit_str}")
        
        try:
            bt.logging.info("Making commit call...")
            result = subtensor.commit(wallet, netuid, registry_commit_str)
            bt.logging.info(f"Commit result: {result}")
            bt.logging.info(f"Successfully committed {registry_commit_str} under {hotkey} on netuid {netuid}")
        except Exception as e:
            bt.logging.error(f"Error during commit: {type(e).__name__}: {e}")
            bt.logging.error(f"Full exception details: {repr(e)}")
            import traceback
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            print(f"Error: {e}")
    else:
        bt.logging.info("Offline mode - Skipping chain commit")


def extract_raw_data(data):
    """Extract raw data from chain response - same as in validator.py"""
    try:
        bt.logging.debug(f"extract_raw_data - Input data type: {type(data)}")
        bt.logging.debug(f"extract_raw_data - Input data: {data}")
        
        # Navigate to the fields tuple
        info = data.get('info', {})
        bt.logging.debug(f"extract_raw_data - Info: {info}")
        
        fields = info.get('fields', ())
        bt.logging.debug(f"extract_raw_data - Fields: {fields}")
        bt.logging.debug(f"extract_raw_data - Fields type: {type(fields)}")
        
        # The first element should be a tuple containing a dictionary
        if fields and isinstance(fields[0], tuple) and isinstance(fields[0][0], dict):
            # Find the 'Raw' key in the dictionary
            raw_dict = fields[0][0]
            bt.logging.debug(f"extract_raw_data - Raw dict: {raw_dict}")
            bt.logging.debug(f"extract_raw_data - Raw dict keys: {raw_dict.keys()}")
            
            raw_key = next((k for k in raw_dict.keys() if k.startswith('Raw')), None)
            bt.logging.debug(f"extract_raw_data - Found raw key: {raw_key}")
            
            if raw_key and raw_dict[raw_key]:
                # Extract the inner tuple of integers
                raw_value = raw_dict[raw_key]
                bt.logging.debug(f"extract_raw_data - Raw value: {raw_value}")
                bt.logging.debug(f"extract_raw_data - Raw value type: {type(raw_value)}")
                
                if isinstance(raw_value, (list, tuple)) and len(raw_value) > 0:
                    numbers = raw_value[0]
                    bt.logging.debug(f"extract_raw_data - Numbers: {numbers}")
                    bt.logging.debug(f"extract_raw_data - Numbers type: {type(numbers)}")
                    
                    # Convert to string
                    result = ''.join(chr(x) for x in numbers)
                    bt.logging.debug(f"extract_raw_data - Result string: {result}")
                    return result
                else:
                    bt.logging.warning(f"extract_raw_data - Raw value is not a list/tuple or is empty")
                
    except (IndexError, AttributeError) as e:
        bt.logging.error(f"extract_raw_data - Error: {type(e).__name__}: {e}")
    except Exception as e:
        bt.logging.error(f"extract_raw_data - Unexpected error: {type(e).__name__}: {e}")
    
    return None


def check_hotkey_endpoint(address: str, port: str, hotkey: str, timeout: float = CHECK_ENDPOINT_TIMEOUT) -> bool:
    """Return True if the miner's /check endpoint responds with HTTP 200."""
    host = address
    if ":" in address and not address.startswith("["):
        host = f"[{address}]"

    url = f"https://{host}:{port}/check/{hotkey}"

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True
        bt.logging.debug(f"/check endpoint returned {response.status_code} for hotkey {hotkey} ({url})")
    except requests.RequestException as exc:
        bt.logging.debug(f"Failed to reach /check endpoint for hotkey {hotkey} at {url}: {exc}")

    return False


def get_all_registrations(config):
    """
    Retrieve all miner registrations from the chain.
    
    Args:
        config: The configuration object with network and netuid settings
        
    Returns:
        Dict mapping hotkeys to their registry data including block and decoded registry info
    """
    bt.logging(config=config)
    
    netuid = config.netuid
    network = config.network if hasattr(config, 'network') else None
        
    # Create subtensor with appropriate network
    if network:
        subtensor = bt.subtensor(network=network)
        bt.logging.info(f"Using network: {network}")
    else:
        subtensor = bt.subtensor(config=config)
    
    bt.logging.info(f"Fetching all registrations from netuid {netuid}")
    
    try:
        # Query the chain for all commitments
        raw_commitments = subtensor.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[netuid]
        )
        
        registrations = {}
        
        for key, value in raw_commitments:
            try:
                hotkey = decode_account_id(key[0])
                body = cast(dict, value.value)
                chain_str = extract_raw_data(body)
                
                if chain_str:
                    # Parse the MinerRegistry from the chain string
                    try:
                        miner_registry = MinerRegistry.from_compressed_str(chain_str)
                        is_active = check_hotkey_endpoint(
                            miner_registry.address,
                            miner_registry.port,
                            str(hotkey)
                        )
                        registrations[str(hotkey)] = {
                            "block": body["block"],
                            "chain_str": chain_str,
                            "registry": miner_registry,
                            "decoded_fields": {
                                "address": miner_registry.address,
                                "port": miner_registry.port,
                            },
                            "check_valid": is_active,
                        }
                        bt.logging.info(f"Decoded registration for hotkey {hotkey}: {miner_registry}")
                    except Exception as e:
                        bt.logging.error(f"Failed to parse MinerRegistry for hotkey {hotkey}: {e}")
                        registrations[str(hotkey)] = {
                            "block": body["block"],
                            "chain_str": chain_str,
                            "registry": None,
                            "error": str(e),
                            "check_valid": False,
                        }
                        
            except Exception as e:
                bt.logging.error(f"Failed to decode commitment: {e}")
                continue
                
        bt.logging.info(f"Found {len(registrations)} total registrations")
        return registrations
        
    except Exception as e:
        bt.logging.error(f"Failed to fetch registrations: {e}")
        return {}


def print_all_registrations(config):
    """
    Print all registrations in a formatted table.
    
    Args:
        config: The configuration object with network and netuid settings
    """
    registrations = get_all_registrations(config)
    
    if not registrations:
        bt.logging.warning("No registrations found")
        return
        
    print("\n" + "="*100)
    print(f"MINER REGISTRATIONS (Total: {len(registrations)})")
    print("="*100)
    
    for hotkey, data in registrations.items():
        print(f"\nHotkey: {hotkey}")
        print(f"Block: {data['block']}")
        print(f"Raw Chain String: {data['chain_str']}")
        
        if data.get('registry'):
            print("Decoded Fields:")
            for field_name, field_value in data['decoded_fields'].items():
                print(f"  - {field_name}: {field_value}")
        if 'check_valid' in data:
            status_label = "valid" if data['check_valid'] else "invalid"
            print(f"Check Endpoint: {status_label}")
        if data.get('error'):
            print(f"Error: {data['error']}")
            
        print("-"*80)


def parse_commit_data(hotkey: str, block: int, chain_str: str) -> Union[ValidCommitData, InvalidCommitData]:
    """Parse commit data and return either ValidCommitData or InvalidCommitData"""
    try:
        miner_registry = MinerRegistry.from_compressed_str(chain_str)
        return ValidCommitData(
            hotkey=hotkey,
            block=block,
            raw_string=chain_str,
            string_length=len(chain_str),
            registry=miner_registry
        )
    except ValueError as e:
        return InvalidCommitData(
            hotkey=hotkey,
            block=block,
            raw_string=chain_str,
            string_length=len(chain_str),
            error_message=str(e),
            error_type="ValueError"
        )
    except Exception as e:
        return InvalidCommitData(
            hotkey=hotkey,
            block=block,
            raw_string=chain_str,
            string_length=len(chain_str),
            error_message=str(e),
            error_type=type(e).__name__
        )


def check_commit(config):
    """Dump all commitments for a specific netuid"""
    bt.logging(config=config)
    
    # Create subtensor with appropriate network
    if hasattr(config, 'network') and config.network:
        subtensor = bt.subtensor(network=config.network)
        bt.logging.info(f"Using network: {config.network}")
    else:
        subtensor = bt.subtensor(config=config)
    
    netuid = config.netuid
    
    bt.logging.info(f"Fetching all commitments for netuid: {netuid} on network: {subtensor.network}")
    
    try:
        # Query all commitments for this netuid
        raw_commitments = subtensor.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[netuid]
        )
        
        # Convert to list to check if empty and get count
        commitments_list = list(raw_commitments)
        
        if not commitments_list:
            bt.logging.warning(f"No commitments found for netuid {netuid}")
            return
            
        print("\n" + "="*80)
        print(f"ALL COMMITMENTS FOR NETUID: {netuid}")
        print(f"Network: {subtensor.network}")
        print(f"Total commitments: {len(commitments_list)}")
        print("="*80)
        
        # Parse all commitments
        parsed_commitments = []
        
        for key, value in commitments_list:
            try:
                hotkey = decode_account_id(key[0])
                body = cast(dict, value.value)
                block = body.get('block', 0)
                chain_str = extract_raw_data(body)
                
                if chain_str:
                    commit_data = parse_commit_data(hotkey, block, chain_str)
                    parsed_commitments.append(commit_data)
                else:
                    # No valid chain string
                    parsed_commitments.append(InvalidCommitData(
                        hotkey=hotkey,
                        block=block,
                        raw_string="",
                        string_length=0,
                        error_message="No valid chain string found in commitment",
                        error_type="MissingData"
                    ))
                
            except Exception as e:
                bt.logging.error(f"Error processing commitment: {e}")
                continue
        
        # Display parsed commitments
        valid_count = sum(1 for c in parsed_commitments if c.status == CommitDataStatus.VALID)
        invalid_count = len(parsed_commitments) - valid_count
        
        print(f"\nSummary: {valid_count} parsed commits, {invalid_count} invalid commits")
        print("-"*80)
        
        # Show valid commitments first
        print(f"\n{'='*20} PARSED COMMITMENTS {'='*20}")
        for commit in parsed_commitments:
            if commit.status == CommitDataStatus.VALID:
                endpoint_valid = check_hotkey_endpoint(
                    commit.registry.address,
                    commit.registry.port,
                    str(commit.hotkey)
                )
                endpoint_status = "valid" if endpoint_valid else "invalid"
                print(f"\nHotkey: {commit.hotkey}")
                print(f"Block: {commit.block}")
                print(f"Address: {commit.registry.address}")
                print(f"Port: {commit.registry.port}")
                print(f"Check Endpoint: {endpoint_status}")
                print(f"String Length: {commit.string_length}")
                print("-"*40)
        
        # Show invalid commitments
        if invalid_count > 0:
            print(f"\n{'='*20} INVALID COMMITMENTS {'='*20}")
            
            if hasattr(config, 'verbose') and config.verbose:
                # Verbose mode - show all details
                for commit in parsed_commitments:
                    if commit.status != CommitDataStatus.VALID:
                        print(f"\nHotkey: {commit.hotkey}")
                        print(f"Block: {commit.block}")
                        print(f"Raw String: {commit.raw_string}")
                        print(f"String Length: {commit.string_length}")
                        print(f"Error Type: {commit.error_type}")
                        print(f"Error: {commit.error_message}")
                        print("-"*40)
            else:
                # Non-verbose mode - just show hotkey and raw string
                for commit in parsed_commitments:
                    if commit.status != CommitDataStatus.VALID:
                        print(f"\nHotkey: {commit.hotkey}")
                        print(f"Raw String: {commit.raw_string}")
                        print(f"String Length: {commit.string_length}")
                        print("-"*40)
                
                print(f"\nTip: Use --verbose to see detailed error information")
                
        print("="*80)
            
    except Exception as e:
        bt.logging.error(f"Error fetching commitments: {e}")
        import traceback
        bt.logging.error(f"Traceback: {traceback.format_exc()}")


def main():
    """Main entry point with CLI commands"""
    config = get_config()
    
    # Execute the appropriate command
    if config.command == 'register':
        register(config)
    elif config.command == 'read':
        print_all_registrations(config)
    elif config.command == 'check':
        check_commit(config)
    else:
        # This shouldn't happen with argparse, but just in case
        print(f"Unknown command: {config.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
