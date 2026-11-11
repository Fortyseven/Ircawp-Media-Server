"""Verify a minimaxh3image transformer GGUF against the official Diffusers-format weights.

Reads only what it needs from ``transformer_diffusers/`` of the official repo
with HTTP range requests (the full copy is 66 GB):

1. every GGUF tensor name/shape matches ``MiniMaxH3ImageTransformer2DModel``
   (the per-block ``adaln_proj.linear`` weights being replaced by ``.table``);
2. spot-checked tensors (first/last block and the refiner) are bit-identical
   (unquantized GGUF) or, for a quantized GGUF, close after dequantization;
3. each ``adaln_proj.table`` equals the real ``MiniMaxH3ImageAdaLayerNormModulation``
   module run with the official weights at the seven timesteps.

Usage: verify_minimaxh3image_gguf.py models/minimax_h3_image_BF16.gguf [--blocks 0 49]
"""

import argparse
import json
import re
import struct
import sys
import time

import gguf
import httpx
import numpy as np
import torch

REPO = "MiniMaxAI/MiniMax-H3-Image"
FOLDER = "transformer_diffusers"
BASE = f"https://huggingface.co/{REPO}/resolve/main/{FOLDER}"
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


class RemoteSafetensors:
    """Range-request reader for a shard of the official repository."""

    def __init__(self, client):
        self.client = client
        self.headers = {}

    def _range(self, url, start, end, attempts=6):
        for attempt in range(attempts):
            try:
                r = self.client.get(url, headers={"Range": f"bytes={start}-{end}"}, follow_redirects=True)
                r.raise_for_status()
                if len(r.content) != end - start + 1:
                    raise httpx.TransportError("short range read")
                return r.content
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def tensor(self, shard, name):
        url = f"{BASE}/{shard}"
        if shard not in self.headers:
            size = struct.unpack("<Q", self._range(url, 0, 7))[0]
            self.headers[shard] = (size, json.loads(self._range(url, 8, 8 + size - 1)))
        size, header = self.headers[shard]
        meta = header[name]
        start, end = meta["data_offsets"]
        raw = bytearray(self._range(url, 8 + size + start, 8 + size + end - 1))
        return torch.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()


def gguf_tensor(reader_tensor):
    """Dequantized torch tensor (float32) for a GGUF tensor, in PyTorch dim order."""
    shape = tuple(int(x) for x in reversed(reader_tensor.shape))
    qtype = reader_tensor.tensor_type
    if qtype == gguf.GGMLQuantizationType.BF16:
        raw = torch.from_numpy(reader_tensor.data.copy().view(np.int16)).view(torch.bfloat16)
        return raw.reshape(shape).float()
    if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return torch.from_numpy(reader_tensor.data.copy()).reshape(shape).float()
    return torch.from_numpy(gguf.quants.dequantize(reader_tensor.data, qtype)).reshape(shape).float()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf_path")
    parser.add_argument("--blocks", type=int, nargs="*", default=[0, 49])
    parser.add_argument("--tolerance", type=float, default=0.0, help="max abs error allowed (0 = exact)")
    args = parser.parse_args(argv)

    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3ImageTransformer2DModel
    from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageAdaLayerNormModulation

    client = httpx.Client(timeout=300)
    config = client.get(f"{BASE}/config.json", follow_redirects=True).json()
    index = client.get(f"{BASE}/diffusion_pytorch_model.safetensors.index.json", follow_redirects=True).json()["weight_map"]
    remote = RemoteSafetensors(client)

    reader = gguf.GGUFReader(args.gguf_path)
    tensors = {t.name: t for t in reader.tensors}

    with init_empty_weights():
        model = MiniMaxH3ImageTransformer2DModel.from_config(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items() if ".adaln_proj.linear." not in k}
    hidden = config["hidden_size"]
    for n in range(config["num_layers"]):
        expected[f"transformer_blocks.{n}.adaln_proj.table"] = None  # shape checked below
    problems = []
    for name, shape in expected.items():
        if name not in tensors:
            problems.append(f"missing {name}")
        elif shape is not None and tuple(int(x) for x in reversed(tensors[name].shape)) != shape:
            problems.append(f"shape {name}: {tuple(reversed(tensors[name].shape))} != {shape}")
    extra = set(tensors) - set(expected) - {"adaln_timesteps"}
    problems += [f"unexpected {n}" for n in sorted(extra)]
    print(f"[1] names/shapes: {len(tensors)} GGUF tensors vs {len(expected)} expected -> "
          f"{'OK' if not problems else problems[:10]}")

    checks = []
    names = [n for n in expected if expected[n] is not None and (
        re.match(r"transformer_blocks\.(\d+)\.", n) is None
        or int(re.match(r"transformer_blocks\.(\d+)\.", n).group(1)) in args.blocks
        or n.startswith(("token_refiner.refiner_blocks.0.", "proj_in", "context_embedder", "norm_out", "pdd_proj_out.5"))
    )]
    worst = 0.0
    for name in names:
        official = remote.tensor(index[name] if name in index else None, name) if name in index else None
        if official is None:
            problems.append(f"{name} not in official index")
            continue
        error = (gguf_tensor(tensors[name]) - official.float()).abs().max().item()
        worst = max(worst, error)
        checks.append((name, error))
        if error > args.tolerance:
            problems.append(f"value {name}: max abs error {error}")
    print(f"[2] values: {len(checks)} tensors compared, worst max-abs error {worst:.3e} "
          f"(tolerance {args.tolerance}) -> {'OK' if worst <= args.tolerance else 'MISMATCH'}")

    timesteps = gguf_tensor(tensors["adaln_timesteps"])
    time_weights = {k: remote.tensor(index[k], k).float() for k in index if k.startswith("time_embedder.")}
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    proj = Timesteps(num_channels=config["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    embedder = TimestepEmbedding(in_channels=config["freq_dim"], time_embed_dim=config["time_embed_hidden_dim"],
                                 out_dim=config["time_embed_dim"])
    embedder.load_state_dict({k.removeprefix("time_embedder."): v for k, v in time_weights.items()})
    with torch.no_grad():
        temb = embedder(proj(timesteps))
    worst_table = 0.0
    for n in args.blocks:
        module = MiniMaxH3ImageAdaLayerNormModulation(config["time_embed_dim"], hidden).to(torch.bfloat16)
        module.linear.weight.data = remote.tensor(index[f"transformer_blocks.{n}.adaln_proj.linear.weight"],
                                                  f"transformer_blocks.{n}.adaln_proj.linear.weight")
        module.linear.bias.data = remote.tensor(index[f"transformer_blocks.{n}.adaln_proj.linear.bias"],
                                                f"transformer_blocks.{n}.adaln_proj.linear.bias")
        with torch.no_grad():
            parts = module(temb)  # six tensors of (T * 3, hidden), rows t * 3 + modality
        reference = torch.stack([p.view(len(timesteps), 3, hidden) for p in parts], dim=2)  # T,3,6,H
        table = gguf_tensor(tensors[f"transformer_blocks.{n}.adaln_proj.table"])
        table = table.view(len(timesteps), 2, 6, hidden)
        error = (table - reference[:, :2].float()).abs().max().item()
        worst_table = max(worst_table, error)
    print(f"[3] adaln tables for blocks {args.blocks}: worst max-abs error vs the real module {worst_table:.3e} "
          f"-> {'OK' if worst_table <= max(args.tolerance, 1e-6) else 'MISMATCH'}")
    print(f"    timesteps in GGUF: {[round(float(t), 6) for t in timesteps]}")
    if problems or worst_table > max(args.tolerance, 1e-6):
        print("FAILED:", problems[:20])
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
