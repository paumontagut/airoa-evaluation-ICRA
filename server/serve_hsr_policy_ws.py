#!/usr/bin/env python3
"""Serve SmolVLA policy as WebSocket server for the HSR client runtime."""

import argparse
import logging
import os
from pathlib import Path

from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SmolVLA policy as websocket server for HSR client")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to SmolVLA checkpoint directory")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--device", default="cuda", help="PyTorch device (cuda, cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    from smolvla_policy import SmolVLAPolicy

    policy = SmolVLAPolicy(checkpoint_dir=checkpoint_dir, device=args.device)

    metadata = {
        "model": "smolvla",
        "checkpoint_dir": checkpoint_dir,
        "server_host": args.host,
        "server_port": args.port,
    }

    logging.info("Serving SmolVLA checkpoint=%s on %s:%s", checkpoint_dir, args.host, args.port)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    main()
