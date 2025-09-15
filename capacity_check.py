#!/usr/bin/env python3
"""Minimal client for fetching miner capacity through the reverse proxy."""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid

from typing import Dict

import requests
import bittensor as bt


DEFAULT_WALLET_NAME = os.getenv("BT_WALLET_NAME", "system1")
DEFAULT_WALLET_HOTKEY = os.getenv("BT_WALLET_HOTKEY", "minerkey1")
DEFAULT_URL = os.getenv("SUBNET_PROXY_URL", "http://localhost:8008/capacity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch miner capacity via Epistula-authenticated request")
    parser.add_argument("--wallet-name", default=DEFAULT_WALLET_NAME, help="Name of the coldkey folder (wallet)")
    parser.add_argument("--wallet-hotkey", default=DEFAULT_WALLET_HOTKEY, help="Hotkey name inside the wallet")
    parser.add_argument("--wallet-path", default=None, help="Override wallet root path if needed")
    parser.add_argument("--miner-hotkey", required=True, help="Target miner hotkey (signed_for)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Reverse proxy capacity endpoint")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    return parser.parse_args()


def epistula_sign_request(
    body: bytes,
    miner_hotkey: str,
    wallet: bt.wallet,
) -> Dict[str, str]:
    """Create a signed request body and Epistula headers."""

    hotkey = wallet.hotkey
    request_uuid = str(uuid.uuid4())
    timestamp_ms = int(time.time() * 1000)
    digest = hashlib.sha256(body).hexdigest()
    message = f"{digest}.{request_uuid}.{timestamp_ms}.{miner_hotkey}"
    signature = f"0x{hotkey.sign(message).hex()}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Epistula {signature}",
        "Epistula-Signature": signature,
        "Epistula-Timestamp": str(timestamp_ms),
        "Epistula-UUID": request_uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Signed-For": miner_hotkey,
    }

    return headers


def load_wallet(name: str, hotkey: str, path: str | None = None) -> bt.wallet:
    return bt.wallet(name=name, hotkey=hotkey, path=path)


def main() -> None:
    args = parse_args()

    wallet = load_wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
        path=args.wallet_path,
    )

    headers = epistula_sign_request(
        body=b"",
        miner_hotkey=args.miner_hotkey,
        wallet=wallet,
    )

    print(f"Requesting capacity from {args.url} as {headers['Epistula-Signed-By']}")
    print(f"Signing for miner hotkey {headers['Epistula-Signed-For']}")

    try:
        response = requests.get(args.url, headers=headers, timeout=args.timeout)
    except requests.RequestException as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    print(f"Status: {response.status_code} {response.reason}")
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
