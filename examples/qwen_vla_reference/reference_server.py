"""Reference Qwen-VLA ZMQ inference server for END-TO-END SERVICE-mode testing.

Hosts a :class:`ReferenceQwenVla` on GPU and speaks exactly the msgpack/ZMQ
envelope that :class:`~strands_robots.policies.qwen_vla.client.QwenVlaInferenceClient`
expects: ``ping``, ``get_action`` (-> (action, info)), ``reset`` (seed),
``reload`` (hot-swap checkpoint). Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).parent))
from reference_model import ReferenceQwenVla  # noqa: E402

from strands_robots.policies.qwen_vla.client import MsgSerializer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
logger = logging.getLogger("qwen_vla_server")


def serve(*, model_path: str | None, host: str, port: int, device: str, denoising_steps: int, data_config: str):
    model = ReferenceQwenVla(device=device)
    if model_path and Path(model_path).exists():
        model.load_checkpoint(model_path)
        logger.info("loaded weights from %s", model_path)
    else:
        logger.info("cold-start model (no weights at %s)", model_path)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    logger.info(
        "Qwen-VLA reference server ready on %s:%d (data_config=%s, steps=%d)", host, port, data_config, denoising_steps
    )

    try:
        while True:
            req = MsgSerializer.from_bytes(sock.recv())
            endpoint = req.get("endpoint")
            try:
                if endpoint == "ping":
                    sock.send(MsgSerializer.to_bytes({"pong": True}))
                elif endpoint == "reset":
                    opts = (req.get("data") or {}).get("options") or {}
                    model.reset(seed=opts.get("seed"))
                    sock.send(MsgSerializer.to_bytes({"ok": True}))
                elif endpoint == "reload":
                    ckpt = (req.get("data") or {}).get("checkpoint")
                    if ckpt and Path(ckpt).exists():
                        model.load_checkpoint(ckpt)
                        sock.send(MsgSerializer.to_bytes({"reloaded": ckpt}))
                        logger.info("hot-swapped checkpoint -> %s", ckpt)
                    else:
                        sock.send(MsgSerializer.to_bytes({"error": f"checkpoint not found: {ckpt}"}))
                elif endpoint == "get_action":
                    obs = (req.get("data") or {}).get("observation", {})
                    action = model.get_action(obs, denoising_steps=denoising_steps)
                    sock.send(MsgSerializer.to_bytes((action, {})))
                else:
                    sock.send(MsgSerializer.to_bytes({"error": f"unknown endpoint {endpoint}"}))
            except Exception as e:  # noqa: BLE001 - report wire errors to client
                sock.send(MsgSerializer.to_bytes({"error": str(e)}))
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--data-config", default="so100")
    args = ap.parse_args()
    serve(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        denoising_steps=args.denoising_steps,
        data_config=args.data_config,
    )
