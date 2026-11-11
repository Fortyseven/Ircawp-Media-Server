"""Convert the MiniMax-H3-Image transformer to an unquantized GGUF, then to Q4_K_S.

Two subcommands:

  convert   official safetensors shards -> models/minimax_h3_image_BF16.gguf
  quantize  that GGUF -> models/minimax_h3_image_Q4_K_S.gguf (llama-quantize)

Tensor names are the *Diffusers* names of ``MiniMaxH3ImageTransformer2DModel``
(the same renames, fused-QKV de-interleave and SwiGLU gate/value swap that
Diffusers' own ``convert_minimax_h3_to_diffusers.py`` applies), so the backend
can load them without further renaming.  The qwen2 architecture key and its
dimensions are spoofed only so llama-quantize accepts the file; they do not
describe the diffusion model.

Two deliberate departures from a plain tensor-for-tensor conversion:

* The six per-step output heads of ``pdd_fused_heads.safetensors`` are stored
  as ``pdd_proj_out.{i}.weight/bias`` so the GGUF is self-contained.
* Every block's ``adaln_proj`` (a 2688 -> 96768 linear, ~13B of the 33B
  parameters, and 2688 is not a multiple of 256 so Q4_K cannot hold it) is a
  pure function of the timestep embedding.  The checkpoint only ever runs at
  the seven timesteps {1 - sigma_i for the 6 PDD steps} + {0.999, the edit
  reference level}, so its output at those timesteps is computed here, with
  the official weights, and stored as ``transformer_blocks.{n}.adaln_proj.table``
  of shape (7, 2 * 6 * hidden) with the unused audio-modality rows dropped.
  ``adaln_timesteps`` holds the seven timestep values.  The weights themselves
  are NOT written: this is exact for the released schedule and unusable for any
  other.  The official safetensors remain the source for a different schedule.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np
import safetensors
import torch
import torch.nn.functional as F

REPO = "MiniMaxAI/MiniMax-H3-Image"
DEFAULT_BF16 = Path("models/minimax_h3_image_BF16.gguf")
DEFAULT_Q4 = Path("models/minimax_h3_image_Q4_K_S.gguf")
LLAMA_QUANTIZE = "/home/fortyseven/dev/_outside/llama.cpp/build/bin/llama-quantize"

# Official MiniMax-H3-Image transformer geometry (transformer/config.json).
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
FFN = 14336
NUM_LAYERS = 50
NUM_REFINER_LAYERS = 2
TIME_EMBED_DIM = 2688
FREQ_DIM = 256
MODALITIES = 3  # video/image, text, audio; the image model never uses audio
USED_MODALITIES = 2
REFERENCE_TIMESTEP = 0.999  # reference_noise_aug of the Diffusers pipeline

# Source tensors whose names start with these are float32 in the checkpoint.
# Anything else 2-D is bfloat16.
HIGH_PRECISION_TARGETS = (
    "proj_in.",
    "context_embedder.",
    "time_embedder.",
    "norm_out.linear.",
    "pdd_proj_out.",
)

# llama-quantize rules. First matching regex wins inside llama.cpp, so the
# specific rules come first and the default type is the positional argument.
QUANTIZE_TENSOR_TYPES = [
    "proj_in=f32",
    "context_embedder=f32",
    "time_embedder=f32",
    "norm_out=f32",
    "pdd_proj_out=f32",
    # The text-conditioning refiner is small and the most sensitive stage.
    "token_refiner=q8_0",
]


def resolve(repo, filename):
    local = Path(repo)
    if local.is_dir():
        return str(local / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


def reorder_interleaved_qkv(weight, heads=None, head_dim=None):
    """Per-head [q k v] row interleave -> [q_all; k_all; v_all] (no transpose)."""
    heads = HEADS if heads is None else heads
    head_dim = HEAD_DIM if head_dim is None else head_dim
    if weight.shape[0] != heads * 3 * head_dim:
        raise ValueError(f"fused qkv has {weight.shape[0]} rows")
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat(
        [t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0
    )


def swap_gate_value(fc1):
    """The checkpoint fuses [gate; value]; Diffusers' SwiGLU reads [value; gate]."""
    gate, value = fc1.chunk(2, dim=0)
    return torch.cat([value, gate], dim=0).contiguous()


def adaln_timesteps(sigmas):
    """The seven distinct timesteps the pipeline feeds the transformer."""
    steps = [float(torch.tensor(1.0 - s, dtype=torch.float32)) for s in sigmas[:-1]]
    return torch.tensor(steps + [REFERENCE_TIMESTEP], dtype=torch.float32)


def timestep_embeddings(timesteps, time_weights):
    """temb for each timestep, exactly as MiniMaxH3ImageTransformer2DModel computes it."""
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    proj = Timesteps(num_channels=FREQ_DIM, flip_sin_to_cos=True, downscale_freq_shift=0)
    embedder = TimestepEmbedding(
        in_channels=FREQ_DIM, time_embed_dim=time_weights["linear_1.weight"].shape[0],
        out_dim=time_weights["linear_2.weight"].shape[0],
    )
    embedder.load_state_dict(
        {
            "linear_1.weight": time_weights["linear_1.weight"],
            "linear_1.bias": time_weights["linear_1.bias"],
            "linear_2.weight": time_weights["linear_2.weight"],
            "linear_2.bias": time_weights["linear_2.bias"],
        }
    )
    with torch.no_grad():
        return embedder(proj(timesteps).to(torch.float32))


def adaln_table(temb, weight, bias):
    """Modulation of one block at each timestep: (T, 2 * 6 * hidden), float32.

    Mirrors MiniMaxH3ImageAdaLayerNormModulation: SiLU at the embedding's
    precision, then a bfloat16 linear. Rows of the audio modality are dropped.
    """
    with torch.no_grad():
        out = F.linear(F.silu(temb).to(torch.bfloat16), weight, bias)
    out = out.view(temb.shape[0], MODALITIES, weight.shape[0] // MODALITIES)
    return out[:, :USED_MODALITIES].reshape(temb.shape[0], -1).float().contiguous()


def read_time_embedder(shard_dir, index):
    names = {
        "time_embedder.proj_in.weight": "linear_1.weight",
        "time_embedder.proj_in.bias": "linear_1.bias",
        "time_embedder.proj_out.weight": "linear_2.weight",
        "time_embedder.proj_out.bias": "linear_2.bias",
    }
    out = {}
    for source, target in names.items():
        shard = Path(shard_dir) / index["weight_map"][source]
        with safetensors.safe_open(str(shard), framework="pt", device="cpu") as f:
            out[target] = f.get_tensor(source).to(torch.float32)
    return out


def rename(name):
    """Official transformer key -> Diffusers key (None = dropped)."""
    if name == "rope.inv_freq" or name.startswith("final_layer.video_out."):
        return None  # recomputed buffer / replaced by the per-step PDD heads
    if name.startswith("token_refiner.blocks."):
        name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif name.startswith("blocks."):
        name = name.replace("blocks.", "transformer_blocks.", 1)
    for old, new in (
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("video_patch_proj.", "proj_in."),
        ("condition_proj.", "context_embedder."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
    ):
        name = name.replace(old, new)
    return name.replace(".mlp.fc2.", ".ff.net.2.")


def convert_tensor(name, tensor):
    """Return [(diffusers_name, tensor)] for one source tensor (adaln handled apart)."""
    target = rename(name)
    if target is None:
        return []
    if target.endswith(".attn.qkv_proj.weight"):
        stacked = reorder_interleaved_qkv(tensor)
        prefix = target.removesuffix("qkv_proj.weight")
        inner = HEADS * HEAD_DIM
        q, k, v = (t.contiguous() for t in stacked.split(inner, dim=0))
        return [(f"{prefix}to_q.weight", q), (f"{prefix}to_k.weight", k), (f"{prefix}to_v.weight", v)]
    if target.endswith(".mlp.fc1.weight"):
        return [(target.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"), swap_gate_value(tensor))]
    return [(target, tensor)]


def target_dtype(name, tensor):
    """(torch tensor to write, GGML type) for the unquantized file."""
    if tensor.ndim < 2 or tensor.dtype == torch.float32 or name.startswith(HIGH_PRECISION_TARGETS):
        return tensor.to(torch.float32), gguf.GGMLQuantizationType.F32
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name}: unexpected dtype {tensor.dtype}")
    return tensor, gguf.GGMLQuantizationType.BF16


def to_array(tensor, qtype):
    if qtype == gguf.GGMLQuantizationType.BF16:
        return tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
    return tensor.contiguous().numpy()


def convert(repo, out, revision=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index_path = resolve(repo, "transformer/model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    with open(resolve(repo, "pdd_config.json"), encoding="utf-8") as f:
        pdd = json.load(f)
    sigmas = pdd["sigmas"]
    if len(sigmas) != pdd["num_inference_steps"] + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0:
        raise ValueError(f"unexpected pdd_config.json: {pdd}")

    shard_dir = Path(index_path).parent
    timesteps = adaln_timesteps(sigmas)
    temb = timestep_embeddings(timesteps, read_time_embedder(shard_dir, index))

    model_name = repo.rsplit("/", 1)[-1] if not Path(repo).is_dir() else Path(repo).name
    writer = gguf.GGUFWriter(str(out), "qwen2", use_temp_file=True)
    counts = {}

    def add(name, tensor):
        tensor, qtype = target_dtype(name, tensor)
        writer.add_tensor(name, to_array(tensor, qtype), raw_shape=tensor.shape, raw_dtype=qtype)
        counts[qtype.name] = counts.get(qtype.name, 0) + 1

    pending_adaln = {}
    try:
        writer.add_name(f"{model_name} BF16")
        writer.add_source_repo_url(f"https://huggingface.co/{repo}")
        if revision:
            writer.add_string("minimax_h3_image.source_revision", revision)
        writer.add_description(
            f"{model_name} transformer, Diffusers tensor names, bf16/f32, adaln_proj "
            "replaced by an exact per-timestep table; qwen2 metadata is spoofed only "
            "for llama-quantize compatibility."
        )
        writer.add_context_length(4096)
        writer.add_embedding_length(HIDDEN)
        writer.add_block_count(NUM_LAYERS)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_array("minimax_h3_image.sigmas", [float(s) for s in sigmas])

        for shard_name in dict.fromkeys(index["weight_map"].values()):
            with safetensors.safe_open(str(shard_dir / shard_name), framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    tensor = shard.get_tensor(name)
                    if name.startswith("blocks.") and ".adaln_proj.linear." in name:
                        block = name.split(".")[1]
                        kind = name.rsplit(".", 1)[-1]  # weight | bias
                        slot = pending_adaln.setdefault(block, {})
                        slot[kind] = tensor
                        if len(slot) == 2:
                            table = adaln_table(temb, slot["weight"], slot["bias"])
                            add(f"transformer_blocks.{block}.adaln_proj.table", table)
                            del pending_adaln[block]
                        continue
                    for target, converted in convert_tensor(name, tensor):
                        add(target, converted)
                    del tensor
        if pending_adaln:
            raise ValueError(f"incomplete adaln tensors for blocks {sorted(pending_adaln)}")

        add("adaln_timesteps", timesteps)
        with safetensors.safe_open(resolve(repo, "pdd_fused_heads.safetensors"), framework="pt", device="cpu") as heads:
            weight, bias = heads.get_tensor("video_out.weight"), heads.get_tensor("video_out.bias")
        for step in range(weight.shape[0]):
            add(f"pdd_proj_out.{step}.weight", weight[step].contiguous())
            add(f"pdd_proj_out.{step}.bias", bias[step].contiguous())

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    print(f"Wrote {sum(counts.values())} tensors {counts} to {out} ({out.stat().st_size} bytes)")


def quantize(src, dst, quant_type="Q4_K_S", llama_quantize=LLAMA_QUANTIZE, dry_run=False):
    command = [llama_quantize]
    for rule in QUANTIZE_TENSOR_TYPES:
        command += ["--tensor-type", rule]
    if dry_run:
        command.append("--dry-run")
    command += [str(src), str(dst), quant_type]
    print(" ".join(command))
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="safetensors -> BF16 GGUF")
    c.add_argument("--repo", default=REPO, help="HF repo id or local snapshot directory")
    c.add_argument("--revision", default=None, help="recorded in the GGUF metadata")
    c.add_argument("--out", type=Path, default=DEFAULT_BF16)
    q = sub.add_parser("quantize", help="BF16 GGUF -> Q4_K_S GGUF")
    q.add_argument("--src", type=Path, default=DEFAULT_BF16)
    q.add_argument("--out", type=Path, default=DEFAULT_Q4)
    q.add_argument("--type", default="Q4_K_S")
    q.add_argument("--llama-quantize", default=LLAMA_QUANTIZE)
    q.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "convert":
        convert(args.repo, args.out, args.revision)
    else:
        quantize(args.src, args.out, args.type, args.llama_quantize, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
