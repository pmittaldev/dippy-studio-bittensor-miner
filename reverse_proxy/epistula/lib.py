"""Epistula authentication verifier."""

import base58
from hashlib import blake2b, sha256
from typing import Optional

from loguru import logger
from substrateinterface import Keypair

class EpistulaVerifier:
    """Handles verification of Epistula protocol authentication."""

    def __init__(
        self,
        miner_hotkey: str,
        allowed_delta_ms: int = 8000,
    ):
        if not miner_hotkey:
            raise ValueError("miner_hotkey must be provided for EpistulaVerifier")
        self.ALLOWED_DELTA_MS = allowed_delta_ms
        self.MINER_HOTKEY = miner_hotkey
        logger.info(
            f"Initialized EpistulaVerifier with miner hotkey: {self.MINER_HOTKEY}"
        )

    def ss58_decode(self, address: str, valid_ss58_format: Optional[int] = None) -> str:
        """
        Decodes given SS58 encoded address to an account ID
        Parameters
        ----------
        address: e.g. EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk
        valid_ss58_format

        Returns
        -------
        Decoded string AccountId
        """

        # Check if address is already decoded
        if address.startswith("0x"):
            return address

        if address == "":
            raise ValueError("Empty address provided")

        checksum_prefix = b"SS58PRE"

        address_decoded = base58.b58decode(address)

        if address_decoded[0] & 0b0100_0000:
            ss58_format_length = 2
            ss58_format = (
                ((address_decoded[0] & 0b0011_1111) << 2)
                | (address_decoded[1] >> 6)
                | ((address_decoded[1] & 0b0011_1111) << 8)
            )
        else:
            ss58_format_length = 1
            ss58_format = address_decoded[0]

        if ss58_format in [46, 47]:
            raise ValueError(f"{ss58_format} is a reserved SS58 format")

        if valid_ss58_format is not None and ss58_format != valid_ss58_format:
            raise ValueError("Invalid SS58 format")

        # Determine checksum length according to length of address string
        if len(address_decoded) in [3, 4, 6, 10]:
            checksum_length = 1
        elif len(address_decoded) in [
            5,
            7,
            11,
            34 + ss58_format_length,
            35 + ss58_format_length,
        ]:
            checksum_length = 2
        elif len(address_decoded) in [8, 12]:
            checksum_length = 3
        elif len(address_decoded) in [9, 13]:
            checksum_length = 4
        elif len(address_decoded) in [14]:
            checksum_length = 5
        elif len(address_decoded) in [15]:
            checksum_length = 6
        elif len(address_decoded) in [16]:
            checksum_length = 7
        elif len(address_decoded) in [17]:
            checksum_length = 8
        else:
            raise ValueError("Invalid address length")

        checksum = blake2b(
            checksum_prefix + address_decoded[0:-checksum_length]
        ).digest()

        if checksum[0:checksum_length] != address_decoded[-checksum_length:]:
            raise ValueError("Invalid checksum")

        return address_decoded[
            ss58_format_length : len(address_decoded) - checksum_length
        ].hex()

    def convert_ss58_to_hex(self, ss58_address: str) -> str:
        """Convert SS58 address to hex format."""
        address_bytes = self.ss58_decode(ss58_address)
        if isinstance(address_bytes, str):
            address_bytes = bytes.fromhex(address_bytes)
        return address_bytes.hex()

    async def verify_signature(
        self,
        signature: str,
        body: bytes,
        timestamp: str,
        uuid_str: str,
        signed_for: Optional[str],
        signed_by: str,
        now: int,
        path: str = "",
    ) -> Optional[str]:
        # Add miner hotkey verification
        if signed_for and signed_for != self.MINER_HOTKEY:
            logger.error(
                f"Request signed for {signed_for} but expected {self.MINER_HOTKEY}"
            )
            return "Invalid signed_for address"

        # Add debug logging
        logger.debug(f"Verifying signature with params:")
        logger.debug(f"signature: {signature}")
        logger.debug(f"body hash: {sha256(body).hexdigest()}")
        logger.debug(f"timestamp: {timestamp}")
        logger.debug(f"uuid: {uuid_str}")
        logger.debug(f"signed_for: {signed_for}")
        logger.debug(f"signed_by: {signed_by}")
        logger.debug(f"current time: {now}")

        # Validate input types
        if not isinstance(signature, str):
            return "Invalid Signature"
        try:
            timestamp = int(timestamp)
        except (ValueError, TypeError):
            return "Invalid Timestamp"
        if not isinstance(signed_by, str):
            return "Invalid Sender key"
        if not isinstance(uuid_str, str):
            return "Invalid uuid"
        if not isinstance(body, bytes):
            return "Body is not of type bytes"

        if timestamp + self.ALLOWED_DELTA_MS < now:
            return "Request is too stale"

        try:
            keypair = Keypair(ss58_address=signed_by)
        except Exception as e:
            logger.error(f"Invalid Keypair for signed_by '{signed_by}': {e}")
            return "Invalid Keypair"

        # Verify signature
        message = (
            f"{sha256(body).hexdigest()}.{uuid_str}.{timestamp}.{signed_for or ''}"
        )
        logger.debug(f"Constructed message for verification: {message}")

        try:
            signature_bytes = bytes.fromhex(signature[2:])  # Remove '0x' prefix
            logger.debug(f"Parsed signature bytes (hex): {signature_bytes.hex()}")
        except ValueError as e:
            logger.error(f"Failed to parse signature: {e}")
            return "Invalid Signature Format"

        verified = keypair.verify(message, signature_bytes)
        logger.debug(f"Signature verification result: {verified}")

        if not verified:
            return "Signature Mismatch"

        return None
