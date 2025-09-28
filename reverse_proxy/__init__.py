"""Bittensor Miner Reverse Proxy Package.

This package provides a reverse proxy server for Bittensor subnet miners
with Epistula authentication capabilities.
"""

__version__ = "0.1.0"
__author__ = "Bittensor Miner Team"

from reverse_proxy.epistula import EpistulaVerifier

__all__ = ["EpistulaVerifier"] 
